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
#include "params.h"
#include "utils.h"
#include <chrono>

using namespace lbcrypto;

int main(int argc, char *argv[]) {

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size \n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    InstanceParams prms(size);

    // The context, the public key, the relinearization key and the rotation keys of the client (no secret key: the
    // server never decrypts)
    std::cout << "         [server] Loading keys" << std::endl;
    CryptoContext<DCRTPoly> context;
    FHEONHEController fheonHEController(context);
    fheonHEController.load_context(lenet5_he_config(prms.pubkeydir().string()), true);
    context = fheonHEController.getContext();
    ANNController fheonANNController(context);
    fheonHEController.load_rotation_keys(lenet5KeyFile);
    /** the bootstrapping precomputations of the 128 slots of fc2 (the keys are in the key file) */
    if (lenet5_uses_bootstrapping()) {
        fheonHEController.prepare_bootstrapping(fcDiagonals);
    }
    LeNet5 lenet5(fheonHEController, fheonANNController);

    fs::create_directories(prms.ctxtdowndir());
    std::cout << "         [server] run encrypted MNIST inference" << std::endl;
    Ctext ctxt;
    for (size_t i = 0; i < prms.getBatchSize(); ++i) {
        auto input_ctxt_path = prms.ctxtupdir() / ("cipher_input_" + std::to_string(i) + ".bin");
        if (!Serial::DeserializeFromFile(input_ctxt_path, ctxt, SerType::BINARY)) {
            throw std::runtime_error("Failed to get ciphertexts from " + input_ctxt_path.string());
        }
        auto start = std::chrono::high_resolution_clock::now();
        auto ctxtResult = lenet5.infer(ctxt);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "         [server] Execution time for ciphertext " << i
                  << " : " << duration.count() / 1000.0 << " seconds" << std::endl;
        auto result_ctxt_path = prms.ctxtdowndir() / ("cipher_result_" + std::to_string(i) + ".bin");
        Serial::SerializeToFile(result_ctxt_path, ctxtResult, SerType::BINARY);
    }
    return 0;
}
