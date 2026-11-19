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
#include "vgg11_fheon.h"

#include "utils.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

/** CKKS parameters of VGG11Diagonal: at 128 bits OpenFHE picks the ring dimension of the modulus chain and the 2^14
 *  slots are sparsely packed in it */
static FHEONHEController::HEConfig vgg11_he_config(const string &keysFolder) {
    FHEONHEController::HEConfig heinitConfig;
    heinitConfig.ringDim = 15;
    heinitConfig.numSlots = 14;
    /** after a bootstrapping: a ReLU and a convolution or linear layer (reluLevels + 1), or a pooling and a convolution */
    heinitConfig.mlevelBootstrap = reluLevels + 2;
    heinitConfig.dcrtBits = 48;
    heinitConfig.firstMod = 52;
    heinitConfig.numDigits = 4;
    /** the level budget of the ResNet-20 submission (degree-119 ReLUs, ring 2^16 at 128 bits); VGG11Diagonal uses {4, 4}
     *  with degree-27 ReLUs */
    heinitConfig.levelBudget = {3, 3};
    heinitConfig.securityLevel = vgg11SecurityLevel;
    heinitConfig.serialize = true;
    heinitConfig.keysFolder = keysFolder;
    /** no keys with the context: the stage key files hold exactly the bootstrapping and rotation keys of their layers
     *  (the diagonal primitives never call EvalSum) */
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

    vgg11_init_fheon();
    auto size = static_cast<InstanceSize>(stoi(argv[1]));
    InstanceParams prms(size);
    fs::create_directories(prms.pubkeydir());
    fs::create_directories(prms.seckeydir());

    // Step 1: CryptoContext, key pair and relinearization key of VGG11Diagonal, serialized to the public key
    // directory (crypto-context.bin, public-key.bin, mult-keys.bin and the parameters of the context)
    CryptoContext<DCRTPoly> context;
    FHEONHEController fheonHEController(context);
    fheonHEController.generate_context(vgg11_he_config(prms.pubkeydir().string()));
    fheonHEController.set_bootstrap_correction_factor(bootstrapCorrectionFactor);
    context = fheonHEController.getContext();
    ANNController fheonANNController(context);

    // Step 2: one key file per stage, generated one after the other: the bootstrapping keys for the slot count of the
    // stage and the rotations of its layers (the last stage also holds the fully connected layers). The server loads
    // one stage at a time
    vector<int> stageSlots = vgg11_stage_slots();
    if (stageSlots[numStages-1] != fcWidth) {
        throw runtime_error("the fully connected layers must have the slot count of the last stage");
    }
    for (int stage = 0; stage < numStages; stage++) {
        vector<int> rotationKeys = vgg11_stage_rotation_keys(fheonANNController, stage);
        cout << "Stage " << stage+1 << " (" << stageSlots[stage] << " slots) keys (" << rotationKeys.size() << ") " << rotationKeys << endl;
        fheonHEController.generate_bootstrapping_and_rotation_keys(rotationKeys, stageSlots[stage], vgg11_stage_key_file(stage), true, false);
        fheonHEController.clear_context(stageSlots[stage]);
    }

    // Step 3: the secret key stays with the client
    fs::path secretKey = fs::path(fheonHEController.secret_key_file);
    fs::copy_file(secretKey, prms.seckeydir() / secretKey.filename(), fs::copy_options::overwrite_existing);
    fs::remove(secretKey);
    cout << "All keys generated" << endl;
    return 0;
}
