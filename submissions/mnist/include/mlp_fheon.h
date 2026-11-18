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
#ifndef MLP_FHEON_H_
#define MLP_FHEON_H_

#include "FHEONANNDiagonalController.h"
#include "FHEONHEController.h"
#include "openfhe.h"

using namespace std;
using namespace lbcrypto;

/** The MLP of FHEON (HEDiagonalSingleModels/MLPDiagonal) with the diagonal primitives: the architecture, the slot
 *  layout of every layer, the levels of the circuit and the encrypted inference. The context and keys are generated in
 *  client_key_generation, the weights are read from the CSV files of the weights folder by server_encrypted_compute.
 *
 *  Both settings use a leveled chain (no bootstrapping) and sparse packing on 1024 slots.
 *  128-bit security (default): the ring optimization of FHEONHEController picks the smallest ring and the largest
 *  scale the standard accepts for the depth of the circuit.
 *  Performance-efficient parameters (performanceMode = true): the parameters of MLPDiagonal, ring 2^11. */
using ANNController = ANNDiagonalController;

/** false: 128-bit security (HEStd_128_classic, uniform ternary secret), the benchmark setting.
 *  true: performance-efficient parameters (HEStd_NotSet, sparse secret), not a standard security level.
 *  Rebuild and rerun key generation after changing it: the client and the server must agree on it */
const bool performanceMode = false;
/** security level of the CKKS context in bits (0 = performance efficient), see FHEONHEController::HEConfig */
const int mlpSecurityLevel = performanceMode ? 0 : 128;

/** degree of the ReLU approximation: 6 levels at degree 59, 5 at 27 and 4 at 13. The linear layers scale the inputs
 *  of the ReLUs, so no ReLU spends a level on scaling */
const int reluDegree = 27;
const int chebyshevLevels = (reluDegree > 27) ? 6 : (reluDegree > 13) ? 5 : 4;
/** fc1 + ReLU + fc2 + ReLU + fc3 */
const int circuitLevels = 3 + 2*chebyshevLevels;

/** 784 -> 128 -> 64 -> 10 */
const vector<int> channels = {784, 128, 64, 10};
/** log2 of the slots of the input ciphertext: the 784 pixels, rounded up to a power of two */
const int inputSlotsLog2 = 10;
/** slots of the input ciphertext of every layer: each linear layer returns its outputs in a ciphertext with as many
 *  slots as it has diagonals (its outputs rounded up to a power of two) */
const vector<int> fcSlots = {1 << inputSlotsLog2, 128, 64};

/** FHEON log verbosity: 0 silent, 1 layer names, 2 timings and memory of every operation */
const int fheonVerboseLevel = 0;

/** sets the FHEON verbosity and starts the clock of the "Total" timings: call once at the start of each stage */
inline void mlp_init_fheon() {
    utils::verbose_level = fheonVerboseLevel;
    utils::program_start_time = std::chrono::steady_clock::now();
}

/** the key file of the rotation keys the client generates and the server loads */
const string mlpKeyFile = "mlp.bin";

/** level the input is encrypted at: the whole circuit ends with 3 towers */
int mlp_input_level(const CryptoContext<DCRTPoly> &context);

/** rotations of the three linear layers */
vector<int> mlp_rotation_keys(ANNController &fheonANNController);

/** the plaintext weights of the MLP, as trained (the CSV files of the weights folder) */
struct MLPWeights {
    vector<vector<double>> fc1Weight, fc2Weight, fc3Weight;
    vector<double> fc1Bias, fc2Bias, fc3Bias;
};

/** the network with its weights encoded once for all the images */
class MLP {
public:
    MLP(FHEONHEController &fheonHEController, ANNController &fheonANNController, MLPWeights weights);
    /** encrypted inference of one image, without decryption: the 10 logits in the first slots */
    Ctext infer(const Ctext &encryptedImage);

private:
    FHEONHEController &fheonHEController;
    ANNController &fheonANNController;
    vector<Ptext> fc1Weights, fc2Weights, fc3Weights;
    Ptext fc1Bias, fc2Bias, fc3Bias;
};

#endif // ifndef MLP_FHEON_H_
