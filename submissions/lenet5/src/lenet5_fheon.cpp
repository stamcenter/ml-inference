
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

Ctext lenet5(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly> &context, 
				Ctext &encryptedInput, string pubkey_dir, string sk_path) {

	string mk_file = "mk.bin";
  	string l1_rk = "layer1_rk.bin";
	FHEONANNController fheonANNController(context);
	fheonHEController.harness_read_evaluation_keys(context, pubkey_dir, mk_file, l1_rk, sk_path);
  	fheonANNController.setContext(context);

	int kernelWidth = 5;
	int poolSize = 2;
	int rotPositions = 16;
	vector<int> imgWidth = {28, 24, 12, 8, 4};
	vector<int> channels = {1, 6, 16, 256, 120, 84, 10};

	/*******************************************************************
	 * Prepare Weights for the network
	 * ******************************************************************/
	string dataPath = WEIGHTS_DIR;

	/*** 1st Convolution */
	auto conv1_biasVec = load_bias(dataPath + "Conv1_bias.csv");
	auto conv1_rawKernel = load_weights(dataPath + "Conv1_weight.csv", channels[1], channels[0], kernelWidth, kernelWidth);
	int conv1WidthSq = pow(imgWidth[0], 2);
	vector<vector<Ptext>> conv1_kernelData;
	for (int i = 0; i < channels[1]; i++) {
		auto encodeKernel = fheonHEController.encode_kernel(conv1_rawKernel[i], conv1WidthSq);
		conv1_kernelData.push_back(encodeKernel);
	}
	auto conv1biasEncoded = fheonHEController.encode_bais_input(conv1_biasVec, (imgWidth[1] * imgWidth[1]));

	/*** 2nd Convolution */
	auto conv2_biasVec = load_bias(dataPath + "Conv2_bias.csv");
	auto conv2_rawKernel = load_weights(dataPath + "Conv2_weight.csv", channels[2], channels[1], kernelWidth, kernelWidth);
	int conv2WidthSq = pow(imgWidth[2], 2);
	vector<vector<Ptext>> conv2_kernelData;
	for (int i = 0; i < channels[2]; i++) {
		auto encodeKernel = fheonHEController.encode_kernel(conv2_rawKernel[i], conv2WidthSq);
		conv2_kernelData.push_back(encodeKernel);
	}
	auto conv2biasEncoded = fheonHEController.encode_bais_input(conv2_biasVec, (imgWidth[3] * imgWidth[3]));

	/*** 1st fc kernel and bias */
	auto fc1_biasVec = load_bias(dataPath + "FC1_bias.csv");
	auto fc1_rawKernel = load_fc_weights(dataPath + "FC1_weight.csv", channels[4], channels[3]);
	vector<Ptext> fc1_kernelData;
	for (int i = 0; i < channels[4]; i++) {
		auto encodeWeights = fheonHEController.encode_input(fc1_rawKernel[i]);
		fc1_kernelData.push_back(encodeWeights);
	}
	Ptext fc1baisVec = fheonHEController.encode_input(fc1_biasVec);

	/*** 2nd fc weights and bias */
	auto fc2_biasVec = load_bias(dataPath + "FC2_bias.csv");
	auto fc2_rawKernel = load_fc_weights(dataPath + "FC2_weight.csv", channels[5], channels[4]);
	vector<Ptext> fc2_kernelData;
	for (int i = 0; i < channels[5]; i++) {
		auto encodeWeights = fheonHEController.encode_input(fc2_rawKernel[i]);
		fc2_kernelData.push_back(encodeWeights);
	}
	Ptext fc2baisVec = fheonHEController.encode_input(fc2_biasVec);

	/*** 3rd fc weights and bias */
	auto fc3_biasVec = load_bias(dataPath + "FC3_bias.csv");
	auto fc3_rawKernel = load_fc_weights(dataPath + "FC3_weight.csv", channels[6], channels[5]);
	vector<Ptext> fc3_kernelData;
	for (int i = 0; i < channels[6]; i++) {
		auto encodeWeights = fheonHEController.encode_input(fc3_rawKernel[i]);
		fc3_kernelData.push_back(encodeWeights);
	}
	Ptext fc3baisVec = fheonHEController.encode_input(fc3_biasVec);

	/*************************************************************************************************
	 * Perform Encrypted Inference on the network
	 * ***********************************************************************************************/
	/*************************************************************************************************/
	int reluScale = 10;
	int polyDegree = 59;
	vector<int> dataSizeVec;
	dataSizeVec.push_back((channels[1] * pow(imgWidth[1], 2)));
	dataSizeVec.push_back((channels[2] * pow(imgWidth[3], 2)));
	/**********************************************************************************************/

	/***** The first Convolution Layer takes  image=(1,28,28), kernel=(6,1,5,5)
	 * stride=1, pooling=0 output= (6,24,24) = 3456 vals */
	auto convData = fheonANNController.he_convolution(encryptedInput, conv1_kernelData, conv1biasEncoded, imgWidth[0], channels[0], channels[1], kernelWidth);
	convData = fheonANNController.he_relu(convData, reluScale, dataSizeVec[0], polyDegree);
	convData = fheonANNController.he_avgpool_optimzed(convData, imgWidth[1], channels[1], poolSize, poolSize);

	/***** Second convolution Layer input = (6,12,12), kernel=(16,6,5,5)
	 * striding =1, padding = 0 output = (16,8,8) ***/
	string l2_rk = "layer2_rk.bin";
  	fheonHEController.harness_read_evaluation_keys(context, pubkey_dir, mk_file,l2_rk, sk_path);
	fheonANNController.setContext(context);
	convData = fheonANNController.he_convolution(convData, conv2_kernelData, conv2biasEncoded, imgWidth[2], channels[1], channels[2], kernelWidth);
	convData = fheonANNController.he_relu(convData, reluScale, dataSizeVec[1],polyDegree);
	convData = fheonHEController.bootstrap_function(convData);
	convData = fheonANNController.he_avgpool_optimzed(convData, imgWidth[3], channels[2], poolSize, poolSize);

	/*** fully connected layers */
	string l3_rk = "layer3_rk.bin";
	fheonHEController.harness_read_evaluation_keys(context, pubkey_dir, mk_file, l3_rk, sk_path);
	fheonANNController.setContext(context);
	convData = fheonANNController.he_linear(convData, fc1_kernelData, fc1baisVec, channels[3], channels[4], rotPositions);
	convData = fheonHEController.bootstrap_function(convData);
	convData = fheonANNController.he_relu(convData, reluScale, channels[4], polyDegree);
	convData = fheonANNController.he_linear(convData, fc2_kernelData, fc2baisVec, channels[4], channels[5], rotPositions);
	convData = fheonHEController.bootstrap_function(convData);
	convData = fheonANNController.he_relu(convData, reluScale, channels[5], polyDegree);
	convData = fheonANNController.he_linear(convData, fc3_kernelData, fc3baisVec,  channels[5], channels[6], rotPositions);
	return convData;
}
