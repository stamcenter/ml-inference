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
#include "mlp_fheon.h"
#include "utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

// Parameters are now defined in mlp_fheon.h via config struct

CryptoContextT generate_crypto_context() {

    lbcrypto::SecretKeyDist secretKeyDist = lbcrypto::SPARSE_TERNARY;
    int circuitDepth = config.modelDepth + lbcrypto::FHECKKSRNS::GetBootstrapDepth(
                                        config.levelBudget, secretKeyDist);

    CCParamsT parameters;
    parameters.SetMultiplicativeDepth(circuitDepth);
      parameters.SetSecurityLevel(HEStd_128_classic);
    // parameters.SetSecurityLevel(HEStd_NotSet);
    // parameters.SetRingDim(config.ringDim);
    // parameters.SetBatchSize(config.numSlots);
    parameters.SetScalingModSize(config.dcrtBits);
    parameters.SetFirstModSize(config.firstMod);
    parameters.SetNumLargeDigits(config.digitSize);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetSecretKeyDist(secretKeyDist);

    CryptoContextT context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    context->Enable(FHE);

    cout << "Context built, generating keys..." << endl;
    cout << endl
        << "dcrtBits: " << config.dcrtBits << " -- firstMod: " << config.firstMod << endl
        << "Ciphertexts depth: " << circuitDepth
        << ", available multiplications: " << config.modelDepth - 2 << endl;
    return context;
}

CryptoContextT generate_mult_rot_key(CryptoContextT context, PrivateKeyT secretKey) {

    context->EvalMultKeyGen(secretKey);
    vector<int> rotPositions = {
            -768, -752, -736, -720, -704, -688, -672, -656, -640, -624, -608, -592, -576, -560, -544, -528, -512, -496, -480, -464, -448, 
            -432, -416, -400, -384, -368, -352, -336, -320, -304, -288, -272, -256, -240, -224, -208, -192, -176, -160, -144, -128, -112, 
            -96, -80, -64, -48, -32, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1
    };
    context->EvalRotateKeyGen(secretKey, rotPositions);
    return context;
}

void generate_rotation_keys(FHEONHEController &fheonHEController, CryptoContextT context, PrivateKeyT secretKey,
                            vector<int> channels, int dataset_size) {

    FHEONANNController fheonANNController(context);

    auto size = static_cast<InstanceSize>(dataset_size);
    InstanceParams prms(size);
    ofstream layer1_file(prms.pubkeydir() / "rk.bin", ios::out | ios::binary);
    vector<vector<int>> rotation_keys;
    int rotPositions = 16;
    auto rotation_positions = fheonANNController.generate_linear_rotation_positions(channels[0], rotPositions);
    cout << "This is the rotation positions (" << rotation_positions.size() << "): " << rotation_positions << endl;
    // fheonHEController.generate_rotation_keys(rotation_positions, "rotation_keys.bin",  true);
    fheonHEController.harness_generate_rotation_keys(context, secretKey, rotation_positions, layer1_file , true);
    /**************************************************************************************************/
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

    FHEONHEController fheonHEController(cryptoContext);

    // Step 2: Key Generation
    // cout << "Starting KeyGen..." << endl;
    auto keyPair = cryptoContext->KeyGen();
    cryptoContext->EvalMultKeyGen(keyPair.secretKey);
    cryptoContext->EvalSumKeyGen(keyPair.secretKey);

    double logPQ = fheonHEController.getlogPQ(keyPair.publicKey->GetPublicElements()[0]);
    cout << "log PQ = " << logPQ << std::endl;
    cout << "Cyclotomic Order: " << cryptoContext->GetCyclotomicOrder() << endl;
    cout << "Ring dimension: " << (cryptoContext->GetCyclotomicOrder() / 2)
        << endl;
    cout << "Num Slots     : " << (cryptoContext->GetCyclotomicOrder() / 4)
        << endl;
    cout << endl;

    // Step 3: Serialize cryptocontext and keys
    fs::create_directories(prms.pubkeydir());
    if (!Serial::SerializeToFile(prms.pubkeydir() / "cc.bin", cryptoContext, SerType::BINARY) ||
        !Serial::SerializeToFile(prms.pubkeydir() / "pk.bin", keyPair.publicKey, SerType::BINARY)) {
        throw runtime_error("Failed to write keys to " + prms.pubkeydir().string());
    }
    ofstream emult_file(prms.pubkeydir() / "mk.bin", ios::out | ios::binary);
    if (!emult_file.is_open() ||
        !cryptoContext->SerializeEvalMultKey(emult_file, SerType::BINARY)) {
        throw runtime_error("Failed to write mult keys to " +
                            prms.pubkeydir().string());
    }

    vector<int> channels = {784, 128, 64, 10};
    generate_rotation_keys(fheonHEController, cryptoContext, keyPair.secretKey, channels, dataset_size);

    // cout << "Eval Keys serialized. Serializing Secret Key..." << endl;
    fs::create_directories(prms.seckeydir());
    if (!Serial::SerializeToFile(prms.seckeydir() / "sk.bin", keyPair.secretKey, SerType::BINARY)) {
        throw runtime_error("Failed to write keys to " + prms.seckeydir().string());
    }
    return 0;
}
