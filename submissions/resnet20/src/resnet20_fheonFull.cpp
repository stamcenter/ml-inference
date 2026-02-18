
/***********************************************************************************************************************
*
* @author: Nges Brian, Njungle
*
* MIT License
* Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State
University

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
********************************************************************************************************************/

#include "resnet20_fheon.h"
#include <iostream>
#include <sys/stat.h>

using namespace std;
using namespace lbcrypto;

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "./submissions/resnet20/weights/resnet20/"
#endif

Ctext convolution_block(FHEONHEController &fheonHEController,
                        FHEONANNController &fheonANNController, string layer,
                        Ctext encrytedVector, int &dataWidth, int &dataSize,
                        int kernelWidth, int padding, int striding,
                        int inputChannels, int outputChannels, int reluScale,
                        bool bootstrapState);
Ctext fc_layer_block(FHEONHEController &fheonHEController,
                     FHEONANNController &fheonANNController, string layer,
                     Ctext encrytedVector, int inputChannels,
                     int outputChannels, int rotPosition);
vector<Ctext> double_shortcut_convolution_block(
    FHEONHEController &fheonHEController,
    FHEONANNController &fheonANNController, string layer, Ctext &encrytedVector,
    int &dataWidth, int &dataSize, int inputChannels, int outputChannels);

Ctext resnet_block(FHEONHEController &fheonHEController,
                   FHEONANNController &fheonANNController, string layer,
                   Ctext encrytedVector, int &dataWidth, int &dataSize,
                   int inputChannels, int outputChannels, int reluScale,
                   int bootstrapState, bool shortcutConv,
                   PrivateKey<DCRTPoly> &sk);

Ctext resnet20(FHEONHEController &fheonHEController,
               CryptoContext<DCRTPoly> &context, Ctext encryptedInput,
               PrivateKey<DCRTPoly> &sk) {

  FHEONANNController fheonANNController(context);

  int img_depth = 3;
  int img_cols = 32;
  int dataWidth = img_cols;
  int kernelSize = 3;
  int padding = 1;
  int striding = 1;
  int avgpoolSize = 8;
  vector<int> channelValues = {16, 32, 64, 10};
  int rotPositions = 16;
  dataWidth = img_cols;
  int dataSize = img_depth * pow(img_cols, 2);
  int reluScale = 5;

  cout << "         [server] Starting encrypted ResNet20 inference" << endl;
  Ctext convData = convolution_block(
      fheonHEController, fheonANNController, "layer0_conv1", encryptedInput,
      dataWidth, dataSize, kernelSize, padding, striding, img_depth,
      channelValues[0], reluScale, false);

  dataSize = channelValues[0] * pow(dataWidth, 2);
  reluScale =
      fheonHEController.read_scaling_value_with_key(sk, convData, dataSize);
  cout << "         [server] Scaling value for ciphertext layer0_conv1 : "
       << reluScale << std::endl;
  convData = fheonANNController.he_relu(convData, reluScale, dataSize, 119);

  cout << "Layer 1" << endl;
  convData =
      resnet_block(fheonHEController, fheonANNController, "layer1_block1",
                   convData, dataWidth, dataSize, channelValues[0],
                   channelValues[0], reluScale, false, false, sk);
  convData =
      resnet_block(fheonHEController, fheonANNController, "layer1_block2",
                   convData, dataWidth, dataSize, channelValues[0],
                   channelValues[0], reluScale, true, false, sk);
  convData =
      resnet_block(fheonHEController, fheonANNController, "layer1_block3",
                   convData, dataWidth, dataSize, channelValues[0],
                   channelValues[0], reluScale, true, false, sk);

  cout << "Layer 2" << endl;
  convData =
      resnet_block(fheonHEController, fheonANNController, "layer2_block1",
                   convData, dataWidth, dataSize, channelValues[0],
                   channelValues[1], reluScale, true, true, sk);
  convData =
      resnet_block(fheonHEController, fheonANNController, "layer2_block2",
                   convData, dataWidth, dataSize, channelValues[1],
                   channelValues[1], reluScale, true, false, sk);
  convData =
      resnet_block(fheonHEController, fheonANNController, "layer2_block3",
                   convData, dataWidth, dataSize, channelValues[1],
                   channelValues[1], reluScale, true, false, sk);

  cout << "Layer 3" << endl;
  convData =
      resnet_block(fheonHEController, fheonANNController, "layer3_block1",
                   convData, dataWidth, dataSize, channelValues[1],
                   channelValues[2], reluScale, true, true, sk);
  convData =
      resnet_block(fheonHEController, fheonANNController, "layer3_block2",
                   convData, dataWidth, dataSize, channelValues[2],
                   channelValues[2], reluScale, true, false, sk);
  convData =
      resnet_block(fheonHEController, fheonANNController, "layer3_block3",
                   convData, dataWidth, dataSize, channelValues[2],
                   channelValues[2], reluScale, true, false, sk);

  cout << "Global Average Pooling" << endl;
  convData = fheonHEController.bootstrap_function(convData);
  convData = fheonANNController.he_globalavgpool(
      convData, dataWidth, channelValues[2], avgpoolSize, rotPositions);
  cout << "Classifier" << endl;
  convData = fc_layer_block(fheonHEController, fheonANNController, "layer_fc",
                            convData, channelValues[2], channelValues[3],
                            rotPositions);
  return convData;
}

Ctext convolution_block(FHEONHEController &fheonHEController,
                        FHEONANNController &fheonANNController, string layer,
                        Ctext encrytedVector, int &dataWidth, int &dataSize,
                        int kernelWidth, int padding, int striding,
                        int inputChannels, int outputChannels, int reluScale,
                        bool bootstrapState) {

  int width_sq = pow(dataWidth, 2);
  int encode_level = encrytedVector->GetLevel();
  vector<vector<Ptext>> kernelData;
  string dataPath = string(WEIGHTS_DIR) + layer;
  auto biasVector = load_bias(dataPath + "_bias.csv");
  auto rawKernel = load_weights(dataPath + "_weight.csv", outputChannels,
                                inputChannels, kernelWidth, kernelWidth);
  for (int i = 0; i < outputChannels; i++) {
    auto encodeKernel = fheonHEController.encode_kernel_optimized(
        rawKernel[i], width_sq, encode_level);
    kernelData.push_back(encodeKernel);
  }

  auto biasVectorEncoded =
      fheonHEController.encode_bais_input(biasVector, width_sq, encode_level);
  auto conv_data = fheonANNController.he_convolution_optimized(
      encrytedVector, kernelData, biasVectorEncoded, dataWidth, inputChannels,
      outputChannels, striding);

  kernelData.clear();
  kernelData.shrink_to_fit();
  biasVector.clear();
  return conv_data;
}

vector<Ctext> double_shortcut_convolution_block(
    FHEONHEController &fheonHEController,
    FHEONANNController &fheonANNController, string layer, Ctext &encrytedVector,
    int &dataWidth, int &dataSize, int inputChannels, int outputChannels) {

  string dataPath = string(WEIGHTS_DIR) + layer;
  int width_sq = pow(dataWidth, 2);
  int width_out_sq = pow((dataWidth / 2), 2);
  int kernelWidth = 3;
  int encode_level = encrytedVector->GetLevel();
  vector<vector<Ptext>> kernelData;
  vector<Ptext> shortcutkernelData;

  /*** convolution data */
  auto biasVector = load_bias(dataPath + "_conv1_bias.csv");
  auto rawKernel = load_weights(dataPath + "_conv1_weight.csv", outputChannels,
                                inputChannels, kernelWidth, kernelWidth);
  auto shortcutbiasVector = load_bias(dataPath + "_shortcut_bias.csv");
  auto shortcutrawKernel = load_fc_weights(dataPath + "_shortcut_weight.csv",
                                           outputChannels, inputChannels);
  for (int i = 0; i < outputChannels; i++) {
    auto encodeKernel = fheonHEController.encode_kernel_optimized(
        rawKernel[i], width_sq, encode_level);
    auto encodeWeights =
        fheonHEController.encode_bais_input(shortcutrawKernel[i], width_sq);
    kernelData.push_back(encodeKernel);
    shortcutkernelData.push_back(encodeWeights);
  }

  auto biasVectorEncoded =
      fheonHEController.encode_bais_input(biasVector, width_out_sq);
  auto shortcutbiasVectorEncoded =
      fheonHEController.encode_bais_input(shortcutbiasVector, width_out_sq);
  auto returnedCiphers =
      fheonANNController.he_convolution_and_shortcut_optimized(
          encrytedVector, kernelData, shortcutkernelData, biasVectorEncoded,
          shortcutbiasVectorEncoded, dataWidth, inputChannels, outputChannels);

  kernelData.clear();
  kernelData.shrink_to_fit();
  rawKernel.clear();
  rawKernel.shrink_to_fit();
  biasVector.clear();

  shortcutkernelData.clear();
  shortcutkernelData.shrink_to_fit();
  shortcutrawKernel.clear();
  shortcutrawKernel.shrink_to_fit();
  shortcutbiasVector.clear();
  return returnedCiphers;
}

Ctext resnet_block(FHEONHEController &fheonHEController,
                   FHEONANNController &fheonANNController, string layer,
                   Ctext encrytedVector, int &dataWidth, int &dataSize,
                   int inputChannels, int outputChannels, int reluScale,
                   int bootstrapState, bool shortcutConv,
                   PrivateKey<DCRTPoly> &sk) {

    int kernelWidth = 3;
    int padding = 1;
    int striding = 1;
    int polyDeg = 119;
    Ctext shortcut_convData = encrytedVector;
    Ctext convData;

    if (shortcutConv) {
        encrytedVector = fheonHEController.bootstrap_function(encrytedVector);
        auto doubleResults = double_shortcut_convolution_block(
            fheonHEController, fheonANNController, layer, encrytedVector, dataWidth,
            dataSize, inputChannels, outputChannels);
        dataWidth = dataWidth / 2;
        dataSize = (outputChannels * pow(dataWidth, 2));

        convData = doubleResults[0]->Clone();
        shortcut_convData = doubleResults[1]->Clone();
    } else {
        convData = convolution_block(
            fheonHEController, fheonANNController, layer + "_conv1", encrytedVector,
            dataWidth, dataSize, kernelWidth, padding, striding, inputChannels,
            outputChannels, reluScale, bootstrapState);
    }
    if (bootstrapState) {
        convData = fheonHEController.bootstrap_function(convData);
    }

    // reluScale = fheonHEController.read_scaling_value_with_key(sk, convData, dataSize);
    // cout << "         [server] Scaling value for conv1 ciphertext " << layer
    //     << " : " << reluScale << std::endl;

    convData = fheonANNController.he_relu(convData, reluScale, dataSize, polyDeg);
    auto second_convData = convolution_block(fheonHEController, fheonANNController, layer + "_conv2", convData,
                                            dataWidth, dataSize, kernelWidth, padding, striding, outputChannels,outputChannels, reluScale, bootstrapState);
    
    Ctext sum_convData = fheonANNController.he_sum_two_ciphertexts(second_convData, shortcut_convData);
    sum_convData = fheonHEController.bootstrap_function(sum_convData);

    //   reluScale = reluScale * 2;
    reluScale = fheonHEController.read_scaling_value_with_key(sk, sum_convData, dataSize);
    //   cout << "         [server] Scaling value for conv2 ciphertext " << layer
    //        << " : " << reluScale << std::endl;
    if(layer == "layer3_block2" || layer == "layer3_block3"){
            reluScale = 25; 
    }
    else{
        reluScale = reluScale * 2;
    }
    sum_convData = fheonANNController.he_relu(sum_convData, reluScale, dataSize, polyDeg);
    return sum_convData;
}

Ctext fc_layer_block(FHEONHEController &fheonHEController,
                     FHEONANNController &fheonANNController, string layer,
                     Ctext encrytedVector, int inputChannels,
                     int outputChannels, int rotPosition) {

  string dataPath = string(WEIGHTS_DIR) + layer;
  auto fc_biasVector = load_bias(dataPath + "_bias.csv");
  auto fc_rawKernelData =
      load_fc_weights(dataPath + "_weight.csv", outputChannels, inputChannels);
  vector<Ptext> fc_kernelData;
  for (int i = 0; i < outputChannels; i++) {
    auto encodeWeights = fheonHEController.encode_input(fc_rawKernelData[i]);
    fc_kernelData.push_back(encodeWeights);
  }
  Ptext encodedbaisVector = fheonHEController.encode_input(fc_biasVector);
  Ctext layer_data = fheonANNController.he_linear_optimized(
      encrytedVector, fc_kernelData, encodedbaisVector, inputChannels,
      outputChannels);
  fc_kernelData.clear();
  fc_kernelData.shrink_to_fit();
  fc_biasVector.clear();
  return layer_data;
}