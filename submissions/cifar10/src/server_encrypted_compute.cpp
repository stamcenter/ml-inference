// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "encryption_utils.h"
#include "params.h"
#include "vgg11_fheon.h"
#include "utils.h"
#include <chrono>

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "./submissions/cifar10/weights/vgg11/"
#endif

using namespace lbcrypto;

/** reads the trained weights from the CSV files of weightsDir (<layer>_weight.csv and <layer>_bias.csv of every layer) */
static VGG11Weights load_vgg11_weights(const std::string &weightsDir) {
	VGG11Weights weights;
	for (const auto &conv : vgg11Convs) {
		weights.conv[conv.name] = load_weights(weightsDir + conv.name + "_weight.csv", conv.outputChannels, conv.inputChannels,
		                                       kernelWidth, kernelWidth);
		weights.bias[conv.name] = load_bias(weightsDir + conv.name + "_bias.csv");
	}
	auto add_linear = [&](const std::string &layer, int outputs, int inputs) {
		weights.linear[layer] = load_fc_weights(weightsDir + layer + "_weight.csv", outputs, inputs);
		weights.bias[layer] = load_bias(weightsDir + layer + "_bias.csv");
	};
	add_linear("fc1", fcWidth, vgg11Convs.back().outputChannels);
	add_linear("fc2", fcWidth, fcWidth);
	add_linear("fc3", numClasses, fcWidth);
	return weights;
}

int main(int argc, char *argv[]) {

	if (argc < 2 || !std::isdigit(argv[1][0])) {
		std::cout << "Usage: " << argv[0] << " instance-size \n";
		std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
		return 0;
	}
	vgg11_init_fheon();
	auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
	InstanceParams prms(size);

	// The context, the public key and the relinearization key of the client (no secret key: the server never decrypts).
	// The bootstrapping and rotation keys of every stage are loaded by the model, stage by stage
	std::cout << "         [server] Loading keys" << std::endl;
	CryptoContext<DCRTPoly> context;
	FHEONHEController fheonHEController(context);
	FHEONHEController::HEConfig keysConfig;
	keysConfig.keysFolder = prms.pubkeydir().string();
	fheonHEController.load_context(keysConfig, true);
	fheonHEController.set_bootstrap_correction_factor(bootstrapCorrectionFactor);
	context = fheonHEController.getContext();
	ANNController fheonANNController(context);
	/** the trained weights, read once from the CSV files of the weights folder */
	VGG11Weights weights = load_vgg11_weights(WEIGHTS_DIR);

	fs::create_directories(prms.ctxtdowndir());
	std::cout << "         [server] run encrypted CIFAR10 inference" << std::endl;
	Ctext ctxt;
	for (size_t i = 0; i < prms.getBatchSize(); ++i) {
		auto input_ctxt_path = prms.ctxtupdir() / ("cipher_input_" + std::to_string(i) + ".bin");
		if (!Serial::DeserializeFromFile(input_ctxt_path, ctxt, SerType::BINARY)) {
			throw std::runtime_error("Failed to get ciphertexts from " + input_ctxt_path.string());
		}
		auto start = std::chrono::high_resolution_clock::now();
		auto ctxtResult = vgg11_infer(fheonHEController, fheonANNController, weights, ctxt);
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		std::cout << "         [server] Execution time for ciphertext " << i
				<< " : " << duration.count() / 1000.0 << " seconds" << std::endl;
		auto result_ctxt_path = prms.ctxtdowndir() / ("cipher_result_" + std::to_string(i) + ".bin");
		Serial::SerializeToFile(result_ctxt_path, ctxtResult, SerType::BINARY);
	}
	return 0;
}
