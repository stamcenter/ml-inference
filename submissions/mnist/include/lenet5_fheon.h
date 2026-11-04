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
#ifndef LENET5_FHEON_H_
#define LENET5_FHEON_H_

#include "FHEONANNDiagonalController.h"
#include "FHEONHEController.h"
#include "openfhe.h"

using namespace std;
using namespace lbcrypto;

/** LeNet-5 with the diagonal primitives of FHEON (FHEON's HEDiagonalSingleModels/LeNet5Diagonal): the architecture,
 *  the slot layout of every layer, the levels of the circuit and the encrypted inference. The context and keys are
 *  generated in client_key_generation, the weights are read from the CSV files of the weights folder by server_encrypted_compute.
 *
 *  128-bit security (default): a leveled chain (no bootstrapping) with a small scale, ring 2^15, every slot of the ring
 *  used (conv2 runs its 16 channels in parallel).
 *  Performance-efficient parameters (performanceMode = true): ring 2^13, one bootstrapping on the 128 slots of fc2. */
using ANNController = ANNDiagonalController;

/** false: 128-bit security (HEStd_128_classic, uniform ternary secret), the benchmark setting.
 *  true: performance-efficient parameters (HEStd_NotSet, sparse secret), not a standard security level.
 *  Rebuild and rerun key generation after changing it: the client and the server must agree on it */
const bool performanceMode = false;
/** security level of the CKKS context in bits (0 = performance efficient), see FHEONHEController::HEConfig */
const int lenet5SecurityLevel = performanceMode ? 0 : 128;

/** degree of the ReLU approximation: 6 levels at degree 59, 5 at 27 and 4 at 13 (plus one when the ReLU scales its input) */
const int reluDegree = 27;
const int chebyshevLevels = (reluDegree > 27) ? 6 : (reluDegree > 13) ? 5 : 4;
const int reluLevels = chebyshevLevels + 1;
/** conv1 + ReLU + conv2 + ReLU + fc1 + ReLU + fc2, and the last ReLU + fc3 */
const int preBootstrapLevels = 3*chebyshevLevels + 6;
const int postBootstrapLevels = chebyshevLevels + 1;
/** depth of the bootstrapping with the level budget {3, 3} */
const int bootstrapDepth = 16;
const int bootstrapIterations = 1;
/** the fully connected outputs, rounded up to a power of two */
const int fcDiagonals = 128;

const int kernelWidth = 5;
const vector<int> imgWidth = {28, 24, 12, 8, 4};
const vector<int> channels = {1, 6, 16, 256, 120, 84, 10};

/** FHEON log verbosity: 0 silent, 1 layer names, 2 timings and memory of every operation */
const int fheonVerboseLevel = 0;

/** sets the FHEON verbosity and starts the clock of the "Total" timings: call once at the start of each stage */
inline void lenet5_init_fheon() {
    utils::verbose_level = fheonVerboseLevel;
    utils::program_start_time = std::chrono::steady_clock::now();
}

/** the key file of the rotation (and bootstrapping) keys the client generates and the server loads */
const string lenet5KeyFile = "lenet5.bin";

/** one bootstrapping at the performance-efficient level, leveled at the standard levels */
inline bool lenet5_uses_bootstrapping() { return performanceMode; }

/** level the input is encrypted at: the circuit before the bootstrapping (the whole circuit when leveled) ends with 3 towers */
int lenet5_input_level(const CryptoContext<DCRTPoly> &context);

/** slot layouts of the layers for a context of numSlots slots (see lenet5_fheon.cpp). The rotation keys of the client
 *  and the weight encoding of the server both follow them */
struct LeNet5Layouts {
    FHEONLayout conv1, conv2, pool1;
    vector<vector<int>> fc1Positions;
};
LeNet5Layouts lenet5_layouts(ANNController &fheonANNController, int numSlots);

/** the plaintext weights of LeNet-5, as trained (the CSV files of the weights folder) */
struct LeNet5Weights {
    vector<vector<vector<vector<double>>>> conv1Weight, conv2Weight;
    vector<vector<double>> fc1Weight, fc2Weight, fc3Weight;
    vector<double> conv1Bias, conv2Bias, fc1Bias, fc2Bias, fc3Bias;
};

/** the network with its weights encoded once for all the images */
class LeNet5 {
public:
    LeNet5(FHEONHEController &fheonHEController, ANNController &fheonANNController, LeNet5Weights weights);
    /** encrypted inference of one image, without decryption: the 10 logits in the first slots */
    Ctext infer(const Ctext &encryptedImage);

private:
    FHEONHEController &fheonHEController;
    ANNController &fheonANNController;
    FHEONLayout conv1Layout, pool1Layout, conv2Layout, fc3Layout;
    vector<vector<Ptext>> conv1Kernel, conv2Kernel;
    vector<Ptext> fc1Weights, fc2Weights, fc3Weights;
    Ptext conv1Bias, conv2Bias, fc1Bias, fc2Bias, fc3Bias;
    /** fc3 is encoded at the level the first inference reaches it (fc3Level) */
    vector<vector<double>> fc3RawWeights;
    vector<double> fc3RawBias;
    int fc3Level = -1;
};

#endif // ifndef LENET5_FHEON_H_
