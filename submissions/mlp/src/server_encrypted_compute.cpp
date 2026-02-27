#include "encryption_utils.h"
#include "mlp_fheon.h"
#include "params.h"
#include "utils.h"
#include <chrono>

using namespace lbcrypto;

int main(int argc, char *argv[]) {

  if (argc < 2 || !std::isdigit(argv[1][0])) {
    std::cout << "Usage: " << argv[0] << " instance-size \n";
    std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
    return 0;
  }
  auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
  InstanceParams prms(size);

  CryptoContext<DCRTPoly> cc = read_crypto_context(prms);
  PublicKey<DCRTPoly> pk = read_public_key(prms);

  std::cout << "         [server] Loading keys" << std::endl;

  Ctext ctxt;
  fs::create_directories(prms.ctxtdowndir());
  std::cout << "         [server] run encrypted MNIST inference" << std::endl;

  FHEONHEController fheonHEController(cc);
  std::string pubkey_dir = prms.pubkeydir().string() + "/";
  std::string mk_file = "mk.bin";
  std::string rk_file = "rk.bin";

  // Load evaluation keys using the harness method for consistency across models
  fheonHEController.harness_read_evaluation_keys(cc, pubkey_dir, mk_file,
                                                 rk_file);

  for (size_t i = 0; i < prms.getBatchSize(); ++i) {
    auto input_ctxt_path =
        prms.ctxtupdir() / ("cipher_input_" + std::to_string(i) + ".bin");
    if (!Serial::DeserializeFromFile(input_ctxt_path, ctxt, SerType::BINARY)) {
      throw std::runtime_error("Failed to get ciphertexts from " +
                               input_ctxt_path.string());
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto ctxtResult = mlp(cc, ctxt);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << "         [server] Execution time for ciphertext " << i
              << " : " << duration.count() << " seconds" << std::endl;
    auto result_ctxt_path =
        prms.ctxtdowndir() / ("cipher_result_" + std::to_string(i) + ".bin");
    Serial::SerializeToFile(result_ctxt_path, ctxtResult, SerType::BINARY);
  }

  return 0;
}
