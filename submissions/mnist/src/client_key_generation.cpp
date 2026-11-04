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
#include "lenet5_fheon.h"

#include "utils.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

/** CKKS parameters of the LeNet-5 circuit (its depth follows the levels in lenet5_fheon.h) */
static FHEONHEController::HEConfig lenet5_he_config(const string &keysFolder) {
    FHEONHEController::HEConfig heinitConfig;
    bool useBootstrapping = lenet5_uses_bootstrapping();
    heinitConfig.ringDim = 13;
    heinitConfig.numSlots = 12;
    /** at a standard security level the ring holds more slots (2^15 at 128 bits): conv2 then runs its 16 channels as
     *  16 parallel groups instead of 4 groups of 4 interleaved channels (see the layout in lenet5_fheon.cpp) */
    heinitConfig.fillSlots = true;
    heinitConfig.mlevelBootstrap = useBootstrapping ? max(postBootstrapLevels + 2, preBootstrapLevels + 1 - bootstrapDepth)
                                                    : preBootstrapLevels + postBootstrapLevels + 1;
    heinitConfig.dcrtBits = 46;
    heinitConfig.firstMod = 50;
    heinitConfig.numDigits = 4;
    heinitConfig.levelBudget = {3, 3};
    /** standard security levels: a leveled chain (no bootstrapping) with a small scale. The ring optimization brings it
     *  to 2^15 at 128 bits: the chain stays precise down to a 22-bit scale, the first modulus keeps 6 bits above the
     *  scale for the logits of the last level */
    if (!useBootstrapping) {
        heinitConfig.firstMod = heinitConfig.dcrtBits + 6;
        heinitConfig.minScaleBits = 23;
    }
    heinitConfig.securityLevel = lenet5SecurityLevel;
    heinitConfig.serialize = true;
    heinitConfig.keysFolder = keysFolder;
    /** no keys with the context: the key file holds exactly the keys of the layers (the diagonal primitives never call EvalSum) */
    heinitConfig.contextBootstrapKeys = false;
    heinitConfig.contextSumKeys = false;
    return heinitConfig;
}

/** generates the context and keys of lenet5_he_config (bootstrapping or leveled context) and serializes them */
static void lenet5_generate_context(FHEONHEController &fheonHEController, const string &keysFolder) {
    auto heinitConfig = lenet5_he_config(keysFolder);
    if (lenet5_uses_bootstrapping()) {
        fheonHEController.generate_context(heinitConfig);
    } else {
        fheonHEController.generate_leveled_context(heinitConfig);
    }
}

/** rotations of every layer of LeNet-5, for a context of numSlots slots */
static vector<int> lenet5_rotation_keys(ANNController &fheonANNController, int numSlots) {
    LeNet5Layouts layouts = lenet5_layouts(fheonANNController, numSlots);
    FHEONLayout fcKeysLayout, fc3KeysLayout;
    fcKeysLayout.inputSlots = numSlots;
    fc3KeysLayout.inputSlots = fcDiagonals;
    vector<vector<int>> rotation_keys;
    rotation_keys.push_back(fheonANNController.generate_convolution_rotation_positions(imgWidth[0], channels[0], channels[1], kernelWidth, 0, 1, layouts.conv1));
    rotation_keys.push_back(fheonANNController.generate_avgpool_rotation_positions(imgWidth[1], 2, 2, channels[1], layouts.pool1));
    rotation_keys.push_back(fheonANNController.generate_convolution_rotation_positions(imgWidth[2], channels[1], channels[2], kernelWidth, 0, 1, layouts.conv2));
    rotation_keys.push_back(fheonANNController.generate_linear_rotation_positions(channels[4], 0, fcKeysLayout));
    rotation_keys.push_back(fheonANNController.generate_linear_rotation_positions(channels[6], 0, fc3KeysLayout));
    return serialize_rotation_keys(rotation_keys);
}

int main(int argc, char *argv[]) {

    if (argc < 2 || !isdigit(argv[1][0])) {
        cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

    lenet5_init_fheon();
    auto size = static_cast<InstanceSize>(stoi(argv[1]));
    InstanceParams prms(size);
    fs::create_directories(prms.pubkeydir());
    fs::create_directories(prms.seckeydir());

    // Step 1: CryptoContext, key pair and relinearization key of LeNet5Diagonal, serialized to the public key
    // directory (crypto-context.bin, public-key.bin, mult-keys.bin, sum-keys.bin and the parameters of the context)
    CryptoContext<DCRTPoly> context;
    FHEONHEController fheonHEController(context);
    lenet5_generate_context(fheonHEController, prms.pubkeydir().string());
    context = fheonHEController.getContext();
    ANNController fheonANNController(context);

    // Step 2: a single key file: the rotations of every layer and, with the performance-efficient parameters, the
    // bootstrapping keys of the 128 slots of fc2 (generated first: the file holds every key in memory)
    if (lenet5_uses_bootstrapping()) {
        fheonHEController.generate_bootstrapping_keys_for_slots(fcDiagonals);
    }
    vector<int> rotationKeys = lenet5_rotation_keys(fheonANNController, fheonHEController.num_slots);
    cout << "Rotation positions (" << rotationKeys.size() << "): " << rotationKeys << endl;
    fheonHEController.generate_rotation_keys(rotationKeys, lenet5KeyFile, true, false);

    // Step 3: the secret key stays with the client
    fs::path secretKey = fs::path(fheonHEController.secret_key_file);
    fs::copy_file(secretKey, prms.seckeydir() / secretKey.filename(), fs::copy_options::overwrite_existing);
    fs::remove(secretKey);
    cout << "All keys generated" << endl;
    return 0;
}
