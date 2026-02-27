#include "encryption_utils.h"
#include "mlp_fheon.h"
#include "utils.h"

using namespace lbcrypto;

int main(int argc, char *argv[]) {

  if (argc < 2 || !std::isdigit(argv[1][0])) {
    std::cout << "Usage: " << argv[0] << " instance-size [--count_only]\n";
    std::cout << "  Instance-size: 0-SINGLE, 1-SMALL, 2-MEDIUM, 3-LARGE\n";
    return 0;
  }
  auto size = static_cast<InstanceSize>(std::stoi(argv[1]));
  InstanceParams prms(size);

  CryptoContext<DCRTPoly> cc = read_crypto_context(prms);

  // Step 2: Read public key
  PublicKey<DCRTPoly> pk = read_public_key(prms);

  std::vector<Sample> dataset;
  load_dataset(dataset, prms.test_input_file().c_str(), MNIST_DIM);
  if (dataset.empty()) {
    throw std::runtime_error("No data found in " +
                             prms.test_input_file().string());
  }
  // Step 2: Encrypt inputs
  if (dataset.size() != prms.getBatchSize()) {
    throw std::runtime_error("Dataset size does not match instance size");
  }

  Ctext ctxt;
  fs::create_directories(prms.ctxtupdir());
  for (size_t i = 0; i < dataset.size(); ++i) {
    auto *input = dataset[i].image;
    std::vector<float> input_vector(input, input + NORMALIZED_DIM);
    // Normalization removed for MLP
    ctxt = input_encrypt(cc, input_vector, pk);
    auto ctxt_path =
        prms.ctxtupdir() / ("cipher_input_" + std::to_string(i) + ".bin");
    Serial::SerializeToFile(ctxt_path, ctxt, SerType::BINARY);
  }

  return 0;
}
