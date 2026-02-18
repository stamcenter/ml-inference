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

vector<uint32_t> levelBudget = {3, 3};
vector<uint32_t> bsgsDim = {0, 0};
int numSlots = 1 << 14;

CryptoContextT generate_crypto_context() {

  int ringDim = 1 << 15;
  int dcrtBits = 50;
  int firstMod = 54;
  int modelDepth = 11;
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
      -15360, -14336, -13312, -12288, -11520, -11264, -10240, -9216, -8192,
      -7936,  -7680,  -7424,  -7168,  -6912,  -6656,  -6400,  -6144, -5952,
      -5888,  -5632,  -5376,  -5120,  -4864,  -4608,  -4352,  -4096, -4032,
      -3968,  -3904,  -3840,  -3776,  -3712,  -3648,  -3584,  -3520, -3456,
      -3392,  -3328,  -3264,  -3200,  -3136,  -3072,  -3008,  -2944, -2880,
      -2816,  -2752,  -2688,  -2624,  -2560,  -2496,  -2432,  -2368, -2304,
      -2240,  -2176,  -2112,  -2048,  -1984,  -1920,  -1856,  -1792, -1728,
      -1664,  -1600,  -1536,  -1472,  -1408,  -1344,  -1280,  -1216, -1152,
      -1088,  -1024,  -960,   -896,   -832,   -768,   -704,   -640,  -576,
      -512,   -448,   -384,   -320,   -256,   -192,   -128,   -64,   -48,
      -32,    -16,    -8,     -1,     1,      2,      3,      4,     5,
      6,      7,      8,      9,      10,     11,     12,     13,    14,
      15,     16,     24,     32,     48,     64,     256,    1024};
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
  // cryptoContext->EvalBootstrapSetup(levelBudget, bsgsDim, numSlots);
  // cryptoContext->EvalBootstrapKeyGen(keyPair.secretKey, numSlots);
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