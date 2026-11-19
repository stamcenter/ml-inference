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


#ifndef VGG11_FHEON_H_
#define VGG11_FHEON_H_

#include "FHEONANNDiagonalController.h"
#include "FHEONHEController.h"
#include "openfhe.h"

#include <map>

using namespace std;
using namespace lbcrypto;

/** VGG-11 with the diagonal primitives of FHEON (FHEON's HEDiagonalSingleModels/VGG11Diagonal): the architecture, the
 *  ReLU scales, the stages and their key files, the level of the input and the encrypted inference. The context and
 *  keys are generated in client_key_generation, the weights are read from the CSV files of the weights folder by
 *  server_encrypted_compute. */
using ANNController = ANNDiagonalController;

/** false: 128-bit security (HEStd_128_classic), the benchmark setting.
 *  true: performance-efficient parameters (HEStd_NotSet, sparse secret), not a standard security level.
 *  Rebuild and rerun key generation after changing it: the client and the server must agree on it */
const bool performanceMode = false;
/** security level of the CKKS context in bits (0 = performance efficient), see FHEONHEController::HEConfig */
const int vgg11SecurityLevel = performanceMode ? 0 : 128;

/** FHEON log verbosity: 0 silent, 1 layer names, 2 timings and memory of every operation */
const int fheonVerboseLevel = 0;

/** sets the FHEON verbosity and starts the clock of the "Total" timings: call once at the start of each stage */
inline void vgg11_init_fheon() {
    utils::verbose_level = fheonVerboseLevel;
    utils::program_start_time = std::chrono::steady_clock::now();
}

/** degree of the ReLU approximation. VGG11Diagonal uses 27 with the scale of every ReLU read from the decrypted
 *  activations of the image; with fixed scales (the server never decrypts) the approximation error builds up over the
 *  10 ReLUs and degree 119 is needed (see vgg11_fheon.cpp) */
const int reluDegree = 119;
/** levels of a ReLU: the Chebyshev evaluation (4 at degree 13, 5 at 27, 6 at 59, 7 at 119) and the scaling of its input */
const int reluLevels = ((reluDegree > 59) ? 7 : (reluDegree > 27) ? 6 : (reluDegree > 13) ? 5 : 4) + 1;
/** several slot counts are bootstrapped (one per stage): they share the same correction factor */
const uint32_t bootstrapCorrectionFactor = 9;
const int bootstrapIterations = 1;

const int imgCols = 32;
const int imgDepth = 3;
const int kernelWidth = 3;
const int numClasses = 10;
/** width of fc1 and fc2, and slot count of the fully connected layers */
const int fcWidth = 1024;
/** bootstrapping below 1024 slots is not accurate: smaller stages are zero-padded to 1024 slots */
const int minimumSlots = 1024;

/** a 3x3 "same" convolution of the network, with the 2x2 average pooling that follows it (if any) */
struct VGG11Conv {
    string name;
    int inputChannels;
    int outputChannels;
    bool poolAfter;
};
const vector<VGG11Conv> vgg11Convs = {
    {"conv1", imgDepth, 16, true},
    {"conv2", 16, 32, true},
    {"conv3", 32, 64, false},
    {"conv4", 64, 64, true},
    {"conv5", 64, 128, false},
    {"conv6", 128, 128, true},
    {"conv7", 128, 128, false},
    {"conv8", 128, 128, false},
};
/** a stage per width (32, 16, 8, 4, 2): every pooling ends a stage */
const int numStages = 5;

/** stage of every convolution */
vector<int> vgg11_layer_stages();
/** width of every stage */
vector<int> vgg11_stage_widths();
/** slot count of every stage: channels * width^2 of its convolutions, at least minimumSlots (16384, 8192, 4096, 2048, 1024) */
vector<int> vgg11_stage_slots();
/** key file of a stage: its bootstrapping keys and the rotations of its layers (the last stage holds the fully connected layers) */
string vgg11_stage_key_file(int stage);
/** rotations of a stage: its convolutions, the pooling that ends it, or the fully connected layers (last stage) */
vector<int> vgg11_stage_rotation_keys(ANNController &fheonANNController, int stage);
/** level the input is encrypted at: the circuit before the first bootstrapping (conv1 and its ReLU) ends with 3 towers */
int vgg11_input_level(const CryptoContext<DCRTPoly> &context);

/** the plaintext weights of VGG-11, as trained (the CSV files of the weights folder), by layer name
 *  ("conv1", ..., "conv8", "fc1", "fc2", "fc3") */
struct VGG11Weights {
    map<string, vector<vector<vector<vector<double>>>>> conv;
    map<string, vector<vector<double>>> linear;
    map<string, vector<double>> bias;
};

/** encrypted inference of one image, without decryption: the 10 logits in the first slots. Every stage has its own key
 *  file (its bootstrapping keys and the rotations of its layers): the model loads the keys of each stage before running it */
Ctext vgg11_infer(FHEONHEController &fheonHEController, ANNController &fheonANNController, VGG11Weights &weights,
                  const Ctext &encryptedImage);

#endif // ifndef VGG11_FHEON_H_
