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
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

/** CKKS parameters of the MLP circuit (its depth follows the levels in mlp_fheon.h): a leveled chain, no bootstrapping */
static FHEONHEController::HEConfig mlp_he_config(const string &keysFolder) {
    FHEONHEController::HEConfig heinitConfig;
    heinitConfig.ringDim = inputSlotsLog2 + 1;
    heinitConfig.numSlots = inputSlotsLog2;
    /** the whole circuit, plus one tower so the logits end above the first modulus */
    heinitConfig.mlevelBootstrap = circuitLevels + 1;
    heinitConfig.levelBudget = {3, 3};
    if (performanceMode) {
        /** the parameters of MLPDiagonal */
        heinitConfig.dcrtBits = 32;
        heinitConfig.firstMod = 36;
        heinitConfig.numDigits = 1;
    } else {
        /** 128-bit security: the ring optimization takes the smallest ring the standard accepts, with the largest
         *  scale down to minScaleBits (the first modulus keeps its 6 bits above the scale) */
        heinitConfig.dcrtBits = 46;
        heinitConfig.firstMod = 52;
        heinitConfig.numDigits = 4;
        heinitConfig.minScaleBits = 23;
    }
    heinitConfig.securityLevel = mlpSecurityLevel;
    heinitConfig.serialize = true;
    heinitConfig.keysFolder = keysFolder;
    /** no keys with the context: the key file holds exactly the keys of the layers (the diagonal primitives never call EvalSum) */
    heinitConfig.contextBootstrapKeys = false;
    heinitConfig.contextSumKeys = false;
    return heinitConfig;
}

int main(int argc, char *argv[]) {

    if (argc < 2 || !isdigit(argv[1][0])) {
        cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    mlp_init_fheon();
    auto size = static_cast<InstanceSize>(stoi(argv[1]));
    InstanceParams prms(size);
    fs::create_directories(prms.pubkeydir());
    fs::create_directories(prms.seckeydir());

    // Step 1: leveled CryptoContext, key pair and relinearization key of the MLP, serialized to the public key
    // directory (crypto-context.bin, public-key.bin, mult-keys.bin and the parameters of the context)
    CryptoContext<DCRTPoly> context;
    FHEONHEController fheonHEController(context);
    fheonHEController.generate_leveled_context(mlp_he_config(prms.pubkeydir().string()));
    context = fheonHEController.getContext();
    ANNController fheonANNController(context);

    // Step 2: a single key file with the rotations of the three linear layers
    vector<int> rotationKeys = mlp_rotation_keys(fheonANNController);
    cout << "Rotation positions (" << rotationKeys.size() << "): " << rotationKeys << endl;
    fheonHEController.generate_rotation_keys(rotationKeys, mlpKeyFile, true, false);

    // Step 3: the secret key stays with the client
    fs::path secretKey = fs::path(fheonHEController.secret_key_file);
    fs::copy_file(secretKey, prms.seckeydir() / secretKey.filename(), fs::copy_options::overwrite_existing);
    fs::remove(secretKey);
    cout << "All keys generated" << endl;
    return 0;
}
