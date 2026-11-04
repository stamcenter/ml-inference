
# Workload Implementation – ML Inference Benchmarks

This submissions implements the LeNet-5 privacy-preserving encrypted machine learning inference model built on [FHEON](https://fheon.pqcsecure.org/) for benchmarking.

## 1. Project Structure

The submissions directory is organized as follows:

- **`submissions/`**: Entry Directory for submissions from Harness.
- **`fheon/`**: The core FHEON framework implementation.
- **`common/`**: Shared utilities for encryption helpers, key management, and data loading.
- **`include/`**: Unified header files for the framework and models.
- **`lenet5/`**: the model source, weights and documentation .

---


## 2. FHEON Implementation

### Introduction to FHEON

**FHEON** is a configurable framework designed to facilitate the implementation of privacy-preserving neural network inference using Fully Homomorphic Encryption (FHE). Built on top of [OpenFHE](https://openfhe-development.readthedocs.io), FHEON provides high-level abstractions for common deep learning components while optimizing for the unique constraints of homomorphic computation.

For more information, visit the [official website](https://fheon.pqcsecure.org/), explore the [source code](https://github.com/stamcenter/fheon), or read the [research paper](https://arxiv.org/abs/2510.03996).

### Model: LeNet-5 (diagonal primitives)

The LeNet-5 model of FHEON's `HEDiagonalSingleModels/LeNet5Diagonal`, built on the diagonal primitives (`FHEONANNDiagonalController`).

- **Architecture**: conv1 (5x5, 1->6), ReLU, average pooling (2x2), conv2 (5x5, 6->16), ReLU, average pooling (fused into fc1), fc1 (256->120), ReLU, fc2 (120->84), ReLU, fc3 (84->10).
- **Layout**: no layer compacts its output; the convolutions run on a strided layout that keeps the 28-slot rows of the image, and the fully connected layers are evaluated over diagonals (baby-step giant-step).
- **ReLU**: Chebyshev approximation of degree 27 with **fixed scales** (11, 9, 8, 10). The server never decrypts: every scale bounds the ReLU input of every image. The fully connected scales are the ones of LeNet5Diagonal; the convolution scales are the largest pre-activations of the plaintext network over the MNIST training and test sets, +5%. Plaintext simulation of the circuit over the 10000 test images: 99.39% (exact ReLU 99.27%).
- **Keys**: the context generates no bootstrapping or EvalSum keys; a single key file (`rotation_keys_lenet5.bin`) holds the rotations of the layers (and, with the performance parameters, the bootstrapping keys of the 128 slots of fc2).
- **Implementation**: `submissions/mnist/src/lenet5_fheon.cpp` (shared by `client_key_generation`, `client_encode_encrypt_input` and `server_encrypted_compute`).

---

## 3. Security Level

The security level is chosen when the submission is built: `cmake -DSECURITY_LEVEL=128` (the default) or `-DSECURITY_LEVEL=0` for the performance-efficient parameters. At 128 bits the LeNet-5 model satisfies the **128-bit security level** of the [Homomorphic Encryption Standard v1.1](https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf) (`HEStd_128_classic`, uniform ternary secret): OpenFHE picks the ring dimension from log QP of the chain.

### CKKS Parameters: LeNet-5, 128 bits (leveled, no bootstrapping)
- **Ciphertexts depth**: 28 (the whole network, no bootstrapping)
- **Scale / first modulus**: 26 / 32 bits, 8 key-switching digits
- **log Q / log QP**: 759 / 879 (the standard allows log QP <= 881 at this ring dimension)
- **Ring dimension**: 32768
- **Number of Slots**: 16384 (every slot of the ring; conv2 runs its 16 channels in parallel)

### Performance-efficient parameters (`-DSECURITY_LEVEL=0`)

Ring dimension 8192 (4096 slots), sparse ternary secret, scale 46 bits and one bootstrapping on the 128 slots of fc2. Not a standard security level.

---

## Technical Details

### Execution Paths
The inference executables typically expect external weights and keys provided at runtime:
- **Weights**: `submissions/<model_name>/weights/`
- **Keys**: Generated and managed via model-specific `client_key_generation` utilities.