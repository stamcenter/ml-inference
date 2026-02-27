
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

#include "lenet5_fheon.h"
#include <iostream>
#include <sys/stat.h>

using namespace std;
using namespace lbcrypto;

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "./../weights/lenet5/"
#endif

Ctext convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, 
					Ctext &encryptedInput,  int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int stride = 1);
Ctext fc_layer_block(FHEONHEController &fheonHEController,FHEONANNController &fheonANNController, string layer,
                     Ctext &encryptedInput, int inputSize, int outputSize, int rotPositions);

Ctext lenet5(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly> &context, Ctext &encryptedInput,
             string pubkey_dir, string sk_path) {

  string mk_file = "mk.bin";
  string l1_rk = "layer1_rk.bin";
  FHEONANNController fheonANNController(context);
  fheonHEController.harness_read_evaluation_keys(context, pubkey_dir, mk_file, l1_rk);

  int kernelWidth = 5;
  int poolSize = 2;
  int rotPositions = 16;
  vector<int> imgWidth = {28, 24, 12, 8, 4};
  vector<int> channels = {1, 6, 16, 256, 120, 84, 10};

  int reluScale = 10;
  int polyDegree = 59;
  vector<int> dataSizeVec;
  dataSizeVec.push_back((channels[1] * pow(imgWidth[1], 2)));
  dataSizeVec.push_back((channels[2] * pow(imgWidth[3], 2)));

  /***** The first Convolution Layer takes  image=(1,28,28), kernel=(6,1,5,5)
   * stride=1, pooling=0 output= (6,24,24) = 3456 vals */
  cout << "         [server] Layer 1" << endl;
  Ctext convData = convolution_block(fheonHEController, fheonANNController, "Conv1", encryptedInput, imgWidth[0], channels[0], 
							channels[1], kernelWidth);
  convData = fheonANNController.he_relu(convData, reluScale, dataSizeVec[0], polyDegree);
  convData = fheonANNController.he_avgpool_optimzed_with_multiple_channels(convData, imgWidth[1], channels[1], poolSize, poolSize);

  /***** Second convolution Layer input = (6,12,12), kernel=(16,6,5,5)
   * striding =1, padding = 0 output = (16,8,8) ***/
  cout << "         [server] Layer 2" << endl;
  string l2_rk = "layer2_rk.bin";
  fheonHEController.harness_read_evaluation_keys(context, pubkey_dir, mk_file, l2_rk);
  convData = convolution_block(fheonHEController, fheonANNController, "Conv2", convData, imgWidth[2], channels[1], 
								                  channels[2], kernelWidth);
  convData = fheonANNController.he_relu(convData, reluScale, dataSizeVec[1], polyDegree);
  context->EvalBootstrapSetup({4, 4});
  convData = fheonHEController.bootstrap_function(convData);
  convData = fheonANNController.he_avgpool_optimzed_with_multiple_channels(convData, imgWidth[3], channels[2], poolSize, poolSize);

  /*** fully connected layers */
  string l3_rk = "layer3_rk.bin";
  fheonHEController.harness_read_evaluation_keys(context, pubkey_dir, mk_file, l3_rk);

  cout << "         [server] FC 1" << endl;
  convData = fc_layer_block(fheonHEController, fheonANNController, "FC1", convData, channels[3], channels[4], rotPositions);
  convData = fheonHEController.bootstrap_function(convData);
  convData = fheonANNController.he_relu(convData, reluScale, channels[4], polyDegree);

  cout << "         [server] FC 2" << endl;
  convData = fc_layer_block(fheonHEController, fheonANNController, "FC2", convData, channels[4], channels[5], rotPositions);
  convData = fheonHEController.bootstrap_function(convData);
  convData = fheonANNController.he_relu(convData, reluScale, channels[5], polyDegree);

  cout << "         [server] FC 3" << endl;
  convData = fc_layer_block(fheonHEController, fheonANNController, "FC3", convData, channels[5], channels[6], rotPositions);

  return convData;
}

Ctext convolution_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer,
                        Ctext &encryptedInput, int inputWidth, int inputChannels, int outputChannels, int kernelWidth, int stride) {

	int widthSq = pow(inputWidth, 2);
	int outWidth = inputWidth - kernelWidth + 1;
	int outWidthSq = pow(outWidth, 2);
	string dataPath = string(WEIGHTS_DIR) + layer;

	auto biasVec = load_bias(dataPath + "_bias.csv");
	auto rawKernel = load_weights(dataPath + "_weight.csv", outputChannels, inputChannels, kernelWidth, kernelWidth);

	vector<vector<Ptext>> convKernelData;
	for (int i = 0; i < outputChannels; i++) {
		auto encodeKernel = fheonHEController.encode_kernel(rawKernel[i], widthSq);
		convKernelData.push_back(encodeKernel);
	}
	auto convBiasEncoded = fheonHEController.encode_bais_input(biasVec, outWidthSq);
	auto convData = fheonANNController.he_convolution(encryptedInput, convKernelData, convBiasEncoded, inputWidth,
									inputChannels, outputChannels, kernelWidth, 0, stride);

	// Clear memory
	for (auto &inner : convKernelData) {
		inner.clear();
	}
	convKernelData.clear();
	convKernelData.shrink_to_fit();
	biasVec.clear();
	rawKernel.clear();
	rawKernel.shrink_to_fit();

	return convData;
}

Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer,
                     Ctext &encryptedInput, int inputSize, int outputSize, int rotPositions) {

  string dataPath = string(WEIGHTS_DIR) + layer;
  auto biasVec = load_bias(dataPath + "_bias.csv");
  auto rawKernel = load_fc_weights(dataPath + "_weight.csv", outputSize, inputSize);

  vector<Ptext> fcKernelData;
  for (int i = 0; i < outputSize; i++) {
    auto encodeWeights = fheonHEController.encode_input(rawKernel[i]);
    fcKernelData.push_back(encodeWeights);
  }
  Ptext fcBiasVec = fheonHEController.encode_input(biasVec);
  Ctext fcData = fheonANNController.he_linear(encryptedInput, fcKernelData, fcBiasVec, inputSize, outputSize, rotPositions);

  // Clear memory
  fcKernelData.clear();
  fcKernelData.shrink_to_fit();
  biasVec.clear();
  rawKernel.clear();
  rawKernel.shrink_to_fit();

  return fcData;
}
