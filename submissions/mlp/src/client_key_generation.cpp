#include "encryption_utils.h"
#include "mlp_fheon.h"
#include "utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

CryptoContextT generate_crypto_context() {
  CCParamsT parameters;
  parameters.SetMultiplicativeDepth(config.modelDepth);
  parameters.SetSecurityLevel(HEStd_NotSet);
  parameters.SetRingDim(config.ringDim);
  parameters.SetBatchSize(config.numSlots);
  parameters.SetScalingModSize(config.dcrtBits);
  parameters.SetFirstModSize(config.firstMod);
  parameters.SetNumLargeDigits(config.digitSize);
  parameters.SetScalingTechnique(FLEXIBLEAUTO);

  CryptoContextT context = GenCryptoContext(parameters);
  context->Enable(PKE);
  context->Enable(KEYSWITCH);
  context->Enable(LEVELEDSHE);

  cout << "Context built, generating keys..." << endl;
  return context;
}

void generate_rotation_keys(FHEONHEController &fheonHEController,
                            CryptoContextT context, PrivateKeyT secretKey,
                            int dataset_size) {

  // MLP currently uses a raw implementation that requires direct keys for all
  // rotations from 1 to 1023. We keep this for compatibility while aligning the
  // code structure.
  vector<int> rkeys;
  for (int i = 1; i < 1024; i++) {
    rkeys.push_back(i);
  }

  auto size = static_cast<InstanceSize>(dataset_size);
  InstanceParams prms(size);

  cout << "Generating " << rkeys.size() << " rotation keys..." << endl;

  ofstream rk_file(prms.pubkeydir() / "rk.bin", ios::out | ios::binary);

  // Serialize keys using the harness method for consistency.
  // For MLP, bootstrapping is not used, so the harness method will just
  // serialize rotation keys.
  fheonHEController.harness_generate_bootstrapping_and_rotation_keys(
      context, secretKey, rkeys, rk_file, false);

  cout << "All keys generated" << endl;
}

int main(int argc, char *argv[]) {

  if (argc < 2 || !isdigit(argv[1][0])) {
    cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
    cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
    return 0;
  }

  int dataset_size = stoi(argv[1]);
  auto size = static_cast<InstanceSize>(dataset_size);
  InstanceParams prms(size);

  // Step 1: Setup CryptoContext
  auto cryptoContext = generate_crypto_context();

  // Step 2: Key Generation
  auto keyPair = cryptoContext->KeyGen();
  cryptoContext->EvalMultKeyGen(keyPair.secretKey);

  FHEONHEController fheonHEController(cryptoContext);

  // Step 3: Serialize cryptocontext and keys
  fs::create_directories(prms.pubkeydir());

  if (!Serial::SerializeToFile(prms.pubkeydir() / "cc.bin", cryptoContext,
                               SerType::BINARY) ||
      !Serial::SerializeToFile(prms.pubkeydir() / "pk.bin", keyPair.publicKey,
                               SerType::BINARY)) {
    throw runtime_error("Failed to write keys to " + prms.pubkeydir().string());
  }

  ofstream emult_file(prms.pubkeydir() / "mk.bin", ios::out | ios::binary);
  if (!emult_file.is_open() ||
      !cryptoContext->SerializeEvalMultKey(emult_file, SerType::BINARY)) {
    throw runtime_error("Failed to write mult keys to " +
                        prms.pubkeydir().string());
  }

  /*** work on rotation keys */
  generate_rotation_keys(fheonHEController, cryptoContext, keyPair.secretKey,
                         dataset_size);

  fs::create_directories(prms.seckeydir());
  if (!Serial::SerializeToFile(prms.seckeydir() / "sk.bin", keyPair.secretKey,
                               SerType::BINARY)) {
    throw runtime_error("Failed to write keys to " + prms.seckeydir().string());
  }

  return 0;
}