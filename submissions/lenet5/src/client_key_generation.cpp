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
#include "utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

vector<uint32_t> levelBudget = {4, 4};
vector<uint32_t> bsgsDim = {0, 0};
int numSlots = 1 << 12;

CryptoContextT generate_crypto_context() {

  int ringDim = 1 << 13;
  int dcrtBits = 46;
  int firstMod = 50;
  int modelDepth = 12;
  int digitSize = 4;
  lbcrypto::SecretKeyDist secretKeyDist = lbcrypto::SPARSE_TERNARY;
  int circuitDepth = modelDepth + lbcrypto::FHECKKSRNS::GetBootstrapDepth(
                                      levelBudget, secretKeyDist);

  CCParamsT parameters;
  parameters.SetMultiplicativeDepth(circuitDepth);
  parameters.SetSecurityLevel(HEStd_NotSet);
  parameters.SetRingDim(ringDim);
  parameters.SetScalingModSize(dcrtBits);
  parameters.SetFirstModSize(firstMod);
  parameters.SetNumLargeDigits(digitSize);
  parameters.SetBatchSize(numSlots);
  parameters.SetScalingTechnique(FLEXIBLEAUTO);
  parameters.SetSecretKeyDist(secretKeyDist);

  CryptoContextT context = GenCryptoContext(parameters);
  context->Enable(PKE);
  context->Enable(KEYSWITCH);
  context->Enable(LEVELEDSHE);
  context->Enable(ADVANCEDSHE);
  context->Enable(FHE);
  return context;
}

CryptoContextT generate_mult_rot_key(CryptoContextT context,
                                     PrivateKeyT secretKey) {

  context->EvalMultKeyGen(secretKey);
  vector<int> rotPositions = {
      -2880, -2304, -1728, -1152, -960, -896, -864, -832, -768, -720, -704,
      -640,  -576,  -552,  -528,  -512, -504, -480, -456, -448, -432, -408,
      -384,  -360,  -336,  -320,  -312, -288, -264, -256, -240, -224, -216,
      -208,  -192,  -176,  -168,  -160, -144, -128, -120, -112, -104, -96,
      -88,   -80,   -72,   -64,   -56,  -48,  -40,  -32,  -24,  -16,  -15,
      -14,   -13,   -12,   -11,   -10,  -9,   -8,   -1,   1,    2,    3,
      4,     5,     6,     7,     8,    9,    10,   11,   12,   13,   14,
      15,    16,    24,    28,    36,   48,   64,   144,  432,  576,  784};
  context->EvalRotateKeyGen(secretKey, rotPositions);
  return context;
}

int main(int argc, char *argv[]) {

  if (argc < 2 || !isdigit(argv[1][0])) {
    cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
    cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
    return 0;
  }
  auto size = static_cast<InstanceSize>(stoi(argv[1]));
  InstanceParams prms(size);

  // Step 1: Setup CryptoContext
  auto cryptoContext = generate_crypto_context();

  // Step 2: Key Generation
  // cout << "Starting KeyGen..." << endl;
  auto keyPair = cryptoContext->KeyGen();
  // cout << "KeyGen done. Starting EvalMultKeyGen..." << endl;
  cryptoContext = generate_mult_rot_key(cryptoContext, keyPair.secretKey);
  // cout << "EvalMultKeyGen done." << endl;

  // Step 4: Bootstrap key generation
  cryptoContext->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots);
  cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
  // cout << "Bootstrap KeyGen done." << endl;

  cryptoContext->EvalSumKeyGen(keyPair.secretKey);

  // Step 3: Serialize cryptocontext and keys
  fs::create_directories(prms.pubkeydir());
  // cout << "Serializing CC and PK..." << endl;

  if (!Serial::SerializeToFile(prms.pubkeydir() / "cc.bin", cryptoContext,
                               SerType::BINARY) ||
      !Serial::SerializeToFile(prms.pubkeydir() / "pk.bin", keyPair.publicKey,
                               SerType::BINARY)) {
    throw runtime_error("Failed to write keys to " + prms.pubkeydir().string());
  }
  // cout << "CC and PK serialized. Serializing Eval Keys..." << endl;
  ofstream emult_file(prms.pubkeydir() / "mk.bin", ios::out | ios::binary);
  ofstream erot_file(prms.pubkeydir() / "rk.bin", ios::out | ios::binary);
  if (!emult_file.is_open() || !erot_file.is_open() ||
      !cryptoContext->SerializeEvalMultKey(emult_file, SerType::BINARY) ||
      !cryptoContext->SerializeEvalAutomorphismKey(erot_file,
                                                   SerType::BINARY)) {
    throw runtime_error("Failed to write eval keys to " +
                        prms.pubkeydir().string());
  }
  // cout << "Eval Keys serialized. Serializing Secret Key..." << endl;

  fs::create_directories(prms.seckeydir());
  if (!Serial::SerializeToFile(prms.seckeydir() / "sk.bin", keyPair.secretKey,
                               SerType::BINARY)) {
    throw runtime_error("Failed to write keys to " + prms.seckeydir().string());
  }
  // cout << "Secret Key serialized." << endl;

  return 0;
}