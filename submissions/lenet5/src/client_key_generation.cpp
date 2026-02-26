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
#include "utils.h"
#include "encryption_utils.h"
#include "lenet5_fheon.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

vector<uint32_t> levelBudget = {4, 4};
vector<uint32_t> bsgsDim = {0, 0};
int numSlots = 1 << 12;
int ringDim = 1 << 13;

CryptoContextT generate_crypto_context() {

	int dcrtBits = 46;
	int firstMod = 50;
	int modelDepth = 11;
	int digitSize = 4;
	lbcrypto::SecretKeyDist secretKeyDist = lbcrypto::SPARSE_TERNARY;
	int circuitDepth = modelDepth + lbcrypto::FHECKKSRNS::GetBootstrapDepth(
										levelBudget, secretKeyDist);

	CCParamsT parameters;
	parameters.SetMultiplicativeDepth(circuitDepth);
	// parameters.SetSecurityLevel(HEStd_128_classic);
	parameters.SetSecurityLevel(HEStd_NotSet);
	parameters.SetRingDim(ringDim);
	parameters.SetBatchSize(numSlots);
	parameters.SetScalingModSize(dcrtBits);
	parameters.SetFirstModSize(firstMod);
	parameters.SetNumLargeDigits(digitSize);
	parameters.SetScalingTechnique(FLEXIBLEAUTO);
	parameters.SetSecretKeyDist(secretKeyDist);

	CryptoContextT context = GenCryptoContext(parameters);
	context->Enable(PKE);
	context->Enable(KEYSWITCH);
	context->Enable(LEVELEDSHE);
	context->Enable(ADVANCEDSHE);
	context->Enable(FHE);
	return context;
}


CryptoContextT generate_mult_rot_key(CryptoContextT context,
                                     PrivateKeyT secretKey) {

	context->EvalMultKeyGen(secretKey);
	vector<int> rotPositions = {
		-2880, -2304, -1728, -1152, -960, -896, -864, -832, -768, -720, -704,
		-640,  -576,  -552,  -528,  -512, -504, -480, -456, -448, -432, -408,
		-384,  -360,  -336,  -320,  -312, -288, -264, -256, -240, -224, -216,
		-208,  -192,  -176,  -168,  -160, -144, -128, -120, -112, -104, -96,
		-88,   -80,   -72,   -64,   -56,  -48,  -40,  -32,  -24,  -16,  -15,
		-14,   -13,   -12,   -11,   -10,  -9,   -8,   -1,   1,    2,    3,
		4,     5,     6,     7,     8,    9,    10,   11,   12,   13,   14,
		15,    16,    24,    28,    36,   48,   64,   144,  432,  576,  784};
	context->EvalRotateKeyGen(secretKey, rotPositions);
	return context;
}


void generate_rotation_keys(CryptoContextT context, PrivateKeyT secretKey,
                            vector<int> channels, int dataset_size) {

	FHEONHEController fheonHEController(context);
	FHEONANNController fheonANNController(context);

	int kernelWidth = 5;
	int poolSize = 2;
	int Stride = 1;
	int paddingLen = 0;
	int rotPositions = 16;
	vector<int> inputWidth = {28, 24, 12, 8, 4};
	auto size = static_cast<InstanceSize>(dataset_size);
  	InstanceParams prms(size);
	
	//** generate rotation keys*/
	auto conv1_keys = fheonANNController.generate_convolution_rotation_positions(inputWidth[0], channels[0], channels[1],  kernelWidth, paddingLen, Stride);
	auto avg1_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(inputWidth[1], channels[1],  poolSize, poolSize, false, "single_channel");
	auto conv2_keys = fheonANNController.generate_convolution_rotation_positions(inputWidth[2], channels[1], channels[2], kernelWidth, paddingLen, Stride);
	auto avg2_keys = fheonANNController.generate_avgpool_optimized_rotation_positions(inputWidth[3],channels[2], poolSize, poolSize, false, "single_channel");
	auto fc_keys = fheonANNController.generate_linear_rotation_positions(channels[4], rotPositions);

	/************************************************************************************************
	 */
	vector<vector<int>> rkeys_layer1, rkeys_layer2, rkeys_layer3;
	rkeys_layer1.push_back(conv1_keys);
	rkeys_layer1.push_back(avg1_keys);

	rkeys_layer2.push_back(conv2_keys);
	rkeys_layer2.push_back(avg2_keys);

	rkeys_layer3.push_back(fc_keys);

	/********************************************************************************************************************************************/
	/*** join all keys and generate unique values only */
	vector<int> serkeys_layer1 = serialize_rotation_keys(rkeys_layer1);
	vector<int> serkeys_layer2 = serialize_rotation_keys(rkeys_layer2);
	vector<int> serkeys_layer3 = serialize_rotation_keys(rkeys_layer3);

	cout << "Layer 1 keys (" << serkeys_layer1.size() << ") " << serkeys_layer1 << endl;
	cout << "Layer 2 keys (" << serkeys_layer2.size() << ") " << serkeys_layer2 << endl;
	cout << "Layer 3 keys (" << serkeys_layer3.size() << ") " << serkeys_layer3 << endl;

	ofstream layer1_file(prms.pubkeydir() / "layer1_rk.bin", ios::out | ios::binary);
	ofstream layer2_file(prms.pubkeydir() / "layer2_rk.bin", ios::out | ios::binary);
	ofstream layer3_file(prms.pubkeydir() / "layer3_rk.bin", ios::out | ios::binary);

	// Generate and serialize the different rotation keys needed by the lenet-5 model
	fheonHEController.harness_generate_bootstrapping_and_rotation_keys(context, secretKey, serkeys_layer1, layer1_file);
	fheonHEController.harness_clear_bootstrapping_and_rotation_keys(context);

	fheonHEController.harness_generate_bootstrapping_and_rotation_keys(context, secretKey, serkeys_layer2, layer2_file);
	fheonHEController.harness_clear_bootstrapping_and_rotation_keys(context);

	fheonHEController.harness_generate_bootstrapping_and_rotation_keys(context, secretKey, serkeys_layer3, layer3_file);
	fheonHEController.harness_clear_bootstrapping_and_rotation_keys(context);
	cout << "All keys generated" << endl;
	/********************************************************************************************************************************************/
}

int main(int argc, char *argv[]) {

	if (argc < 2 || !isdigit(argv[1][0])) {
		cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
		cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
		return 0;
	}
	int dataset_size = stoi(argv[1]);
	auto size = static_cast<InstanceSize>(dataset_size);
	InstanceParams prms(size);

	// Step 1: Setup CryptoContext
	auto cryptoContext = generate_crypto_context();

	// Step 2: Key Generation
	// cout << "Starting KeyGen..." << endl;
	auto keyPair = cryptoContext->KeyGen();
	cryptoContext->EvalMultKeyGen(keyPair.secretKey);
	cryptoContext->EvalSumKeyGen(keyPair.secretKey);

	// Step 3: Serialize cryptocontext and keys
	fs::create_directories(prms.pubkeydir());
	if (!Serial::SerializeToFile(prms.pubkeydir() / "cc.bin", cryptoContext, SerType::BINARY) ||
		!Serial::SerializeToFile(prms.pubkeydir() / "pk.bin", keyPair.publicKey, SerType::BINARY)) {
		throw runtime_error("Failed to write keys to " + prms.pubkeydir().string());
	}
	ofstream emult_file(prms.pubkeydir() / "mk.bin", ios::out | ios::binary);
	if (!emult_file.is_open() || !cryptoContext->SerializeEvalMultKey(emult_file, SerType::BINARY)) {
		throw runtime_error("Failed to write mult keys to " + prms.pubkeydir().string());
	}

	vector<int> channels = {1, 6, 16, 256, 120, 84, 10};
	generate_rotation_keys(cryptoContext, keyPair.secretKey, channels, dataset_size);

	// cout << "Eval Keys serialized. Serializing Secret Key..." << endl;
	fs::create_directories(prms.seckeydir());
	if (!Serial::SerializeToFile(prms.seckeydir() / "sk.bin", keyPair.secretKey, SerType::BINARY)) {
		throw runtime_error("Failed to write keys to " + prms.seckeydir().string());
	}

	return 0;
}