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

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "./submissions/mnist/weights/lenet5/"
#endif

using namespace lbcrypto;

/** reads the trained weights from the CSV files of weightsDir (Conv1_weight.csv, Conv1_bias.csv, ..., FC3_bias.csv) */
static LeNet5Weights load_lenet5_weights(const std::string &weightsDir) {
    LeNet5Weights weights;
    weights.conv1Weight = load_weights(weightsDir + "Conv1_weight.csv", channels[1], channels[0], kernelWidth, kernelWidth);
    weights.conv1Bias = load_bias(weightsDir + "Conv1_bias.csv");
    weights.conv2Weight = load_weights(weightsDir + "Conv2_weight.csv", channels[2], channels[1], kernelWidth, kernelWidth);
    weights.conv2Bias = load_bias(weightsDir + "Conv2_bias.csv");
    weights.fc1Weight = load_fc_weights(weightsDir + "FC1_weight.csv", channels[4], channels[3]);
    weights.fc1Bias = load_bias(weightsDir + "FC1_bias.csv");
    weights.fc2Weight = load_fc_weights(weightsDir + "FC2_weight.csv", channels[5], channels[4]);
    weights.fc2Bias = load_bias(weightsDir + "FC2_bias.csv");
    weights.fc3Weight = load_fc_weights(weightsDir + "FC3_weight.csv", channels[6], channels[5]);
    weights.fc3Bias = load_bias(weightsDir + "FC3_bias.csv");
    return weights;
}

int main(int argc, char *argv[]) {

    if (argc < 2 || !std::isdigit(argv[1][0])) {
        std::cout << "Usage: " << argv[0] << " instance-size \n";
        std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
        return 0;
    }
    lenet5_init_fheon();
    auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
    InstanceParams prms(size);

    // The context, the public key, the relinearization key and the rotation keys of the client (no secret key: the
    // server never decrypts)
    std::cout << "         [server] Loading keys" << std::endl;
    CryptoContext<DCRTPoly> context;
    FHEONHEController fheonHEController(context);
    FHEONHEController::HEConfig keysConfig;
    keysConfig.keysFolder = prms.pubkeydir().string();
    fheonHEController.load_context(keysConfig, true);
    context = fheonHEController.getContext();
    ANNController fheonANNController(context);
    fheonHEController.load_rotation_keys(lenet5KeyFile);
    /** the bootstrapping precomputations of the 128 slots of fc2 (the keys are in the key file) */
    if (lenet5_uses_bootstrapping()) {
        fheonHEController.prepare_bootstrapping(fcDiagonals);
    }
    /** the trained weights, read from the CSV files of the weights folder */
    LeNet5 lenet5(fheonHEController, fheonANNController, load_lenet5_weights(WEIGHTS_DIR));

    fs::create_directories(prms.ctxtdowndir());
    std::cout << "         [server] run encrypted MNIST inference" << std::endl;
    Ctext ctxt;
    for (size_t i = 0; i < prms.getBatchSize(); ++i) {

        cout << endl<< "         [server] Processing Input " << i << std::endl;
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
