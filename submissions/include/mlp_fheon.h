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
  vector<uint32_t> levelBudget = {
      1, 1}; // MLP doesn't use bootstrapping but harness might expect it
  vector<uint32_t> bsgsDim = {0, 0};
  int ringDim = 0; // GenCryptoContext will Decide
  int numSlots = 1 << 12;
  int dcrtBits = 50;
  int firstMod = 60;
  int modelDepth = 9;
  int digitSize = 3;
};

inline MLPConfig config;

Ctext mlp(CryptoContext<DCRTPoly> &v0, Ctext &v1);

#endif // ifndef MLP_FHEON_H_
