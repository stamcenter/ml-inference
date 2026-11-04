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

int main(int argc, char *argv[]) {

    if (argc < 2 || !isdigit(argv[1][0])) {
        cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
        cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }

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
