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

#include "FHEONANNController.h"
#include "FHEONHEController.h"
#include "openfhe.h"

using namespace std;
using namespace lbcrypto;

struct LeNetConfig {
  vector<uint32_t> levelBudget = {4, 4};
  vector<uint32_t> bsgsDim = {0, 0};
  int ringDim = 1 << 13;
  int numSlots = 1 << 12;
  int dcrtBits = 42;
  int firstMod = 46;
  int modelDepth = 11;
  int digitSize = 4;
};

inline LeNetConfig config;

// using CiphertextT = ConstCiphertext<DCRTPoly>;

Ctext lenet5(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly> &v0,
             Ctext &v1, string pubkey_dir);

// Ctext lenet5(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly>
// &v0, Ctext v1);

#endif // ifndef LENET5_FHEON_H_