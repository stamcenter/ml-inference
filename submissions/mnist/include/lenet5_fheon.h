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

/** LeNet-5 with the diagonal primitives of FHEON (FHEON's HEDiagonalSingleModels/LeNet5Diagonal). The client (key
 *  generation, encryption) and the server (inference) share the circuit below: the parameters of the context, the
 *  rotation keys, and the level of the input.
 *
 *  Performance-efficient parameters (cmake -DSECURITY_LEVEL=0): ring 2^13, one bootstrapping on the 128 slots of fc2.
 *  Standard levels (128 bits): a leveled chain (no bootstrapping) with a small scale, ring 2^15 at 128 bits, every slot
 *  of the ring used (conv2 runs its 16 channels in parallel). */
using ANNController = ANNDiagonalController;

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

/** one bootstrapping at the performance-efficient level, leveled at the standard levels */
inline bool lenet5_uses_bootstrapping() { return FHEON_SECURITY_LEVEL == 0; }

FHEONHEController::HEConfig lenet5_he_config(const string &keysFolder);
/** generates the context and keys of lenet5_he_config (bootstrapping or leveled context) and serializes them */
void lenet5_generate_context(FHEONHEController &fheonHEController, const string &keysFolder);
/** rotations of every layer, for a context of numSlots slots */
vector<int> lenet5_rotation_keys(ANNController &fheonANNController, int numSlots);
const string lenet5KeyFile = "lenet5.bin";
/** level the input is encrypted at: the circuit before the bootstrapping (the whole circuit when leveled) ends with 3 towers */
int lenet5_input_level(const CryptoContext<DCRTPoly> &context);

/** the network with its weights encoded once for all the images */
class LeNet5 {
public:
    LeNet5(FHEONHEController &fheonHEController, ANNController &fheonANNController);
    /** encrypted inference of one image, without decryption: the 10 logits in the first slots */
    Ctext infer(const Ctext &encryptedImage);

private:
    FHEONHEController &fheonHEController;
    ANNController &fheonANNController;
    FHEONLayout conv1Layout, conv2Layout, pool1Layout, fc1Layout, fc2Layout, fc3Layout;
    vector<vector<Ptext>> conv1_kernelData, conv2_kernelData;
    Ptext conv1biasEncoded, conv2biasEncoded;
    vector<Ptext> fc1_kernelData, fc2_kernelData, fc3_kernelData;
    Ptext fc1baisVector, fc2baisVector, fc3baisVector;
    vector<vector<double>> fc3_rawKernelData;
    vector<double> fc3_biasVector;
    int fc3Level = -1;
};

#endif // ifndef LENET5_FHEON_H_
