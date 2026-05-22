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

#include "fheon/FHEONANNController.h"
#include "fheon/FHEONHEController.h"
#include "openfhe.h"
#include <cstdint>
#include <vector>

using namespace std;
using namespace lbcrypto;

struct MLPConfig {
  vector<uint32_t> levelBudget = {3, 3};
  vector<uint32_t> bsgsDim = {0, 0};
  int ringDim = 1 << 11;
  int numSlots = 1 << 10;
  int dcrtBits = 26;
  int firstMod = 30;
  int modelDepth = 7;
  int digitSize = 3;
};

inline MLPConfig config;

// Ctext mlp(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly> &v0, Ctext &v1);
Ctext mlp(FHEONHEController &fheonHEController, CryptoContext<DCRTPoly> &context, Ctext &encryptedInput, PrivateKey<DCRTPoly>& sk);

#endif // ifndef MLP_FHEON_H_
