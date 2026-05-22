
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

#include "mlp_fheon.h"
#include <iostream>
#include <sys/stat.h>

using namespace std;
using namespace lbcrypto;

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "./../weights/mlp_fheon/"
#endif

Ctext fc_layer_block(FHEONHEController &fheonHEController, FHEONANNController &fheonANNController, string layer, 
                        Ctext &encryptedInput, int inputSize, int outputSize, int rotPositions);

Ctext mlp(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly> &context, Ctext &encryptedInput, PrivateKey<DCRTPoly>& sk) {
// Ctext mlp(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly> &context, Ctext &encryptedInput) {

	int rotPositions = 16;
	int polyDegree = 119;
    vector<int> channels = {784, 128, 64, 10};

    FHEONANNController fheonANNController(context);

    // cout << "         [server] FC1 layer" << endl;
    auto mlpData = fc_layer_block(fheonHEController, fheonANNController, "fc1", encryptedInput, channels[0], channels[1], rotPositions);
    
    int reluScale = 175; 
    // reluScale = fheonHEController.read_scaling_value_with_key(sk, mlpData, channels[1]); 
    // cout << "         [server] ReLU1 layer " << reluScale << endl;
    mlpData = fheonANNController.he_relu(mlpData, reluScale, channels[1], polyDegree);
    
    // cout << "         [server] FC2 layer" << endl;
    mlpData = fc_layer_block(fheonHEController, fheonANNController, "fc2", mlpData, channels[1], channels[2], rotPositions);
   
    reluScale = 300;
    // reluScale = fheonHEController.read_scaling_value_with_key(sk, mlpData, channels[2]);
    // cout << "         [server] ReLU2 with scale " << reluScale << endl;
    mlpData = fheonANNController.he_relu(mlpData, reluScale, channels[2], polyDegree);
    
    // cout << "         [server] FC3 layer" << endl;
    mlpData = fc_layer_block(fheonHEController, fheonANNController, "fc3", mlpData, channels[2], channels[3], rotPositions);

    return mlpData;
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