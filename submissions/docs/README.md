# Workload Implementation – ML Inference Benchmarks

This submissions implements various privacy-preserving encrypted machine learning inference models for benchmarking.

## Project Structure

The submissions directory is organized as follows:

- **`submissions/`**: Contains specific model implementations (MLP, LeNet-5, ResNet-20).
- **`fheon/`**: The core FHEON framework implementation.
- **`common/`**: Shared utilities for encryption helpers, key management, and data loading.
- **`include/`**: Unified header files for the framework and models.

---

## 1. OpenFHE Implementation

The following baseline model is implemented directly using [OpenFHE](https://openfhe-development.readthedocs.io) primitives, serving as a primary reference for encrypted arithmetic performance.

### MLP (Direct OpenFHE)

- **Description**: A Multi-Layer Perceptron (MLP) fully connected network.
- **Dataset**: MNIST
- **Implementation**: `submissions/mlp/src/mlp_openfhe.cpp`
- **Key Details**: Features hardcoded weights optimized for standard CKKS operations.

---

## 2. FHEON Implementation

### Introduction to FHEON

**FHEON** is a configurable framework designed to facilitate the implementation of privacy-preserving neural network inference using Fully Homomorphic Encryption (FHE). Built on top of OpenFHE, FHEON provides high-level abstractions for common deep learning components while optimizing for the unique constraints of homomorphic computation.

For more information, visit the [official website](https://fheon.pqcsecure.org/), explore the [source code](https://github.com/stamcenter/fheon), or read the [research paper](https://arxiv.org/abs/2510.03996).

### Model: LeNet-5

A modular implementation of the classic LeNet-5 model using the FHEON framework.

- **Architecture**: Convolution layers (5x5 kernel, stride 1), Average Pooling (2x2, stride 2), and Polynomial Approximation for ReLU.
- **Dataset**: MNIST
- **Optimization**: Modular blocks for Convolution and Fully Connected layers with efficient data management (clears intermediate weights/biases immediately after use).
- **Implementation**: `submissions/lenet5/src/lenet5_fheon.cpp`

### Model: ResNet-20

A deep residual network targeting CIFAR-10 built using FHEON.

- **Architecture**: Initial convolution, three stages of ResNet blocks with shortcuts, and Global Average Pooling.
- **Dataset**: CIFAR-10
- **Bootstrapping**: Strategic integration of CKKS bootstrapping to maintain circuit depth.
- **Implementation**: `submissions/resnet20/src/resnet20_fheon.cpp`

---

## 3. Security Level

Both the LeNet-5 and ResNet-20 models are configured to satisfy the **128-bit security level** using the standardized parameters for CKKS as defined in the [Homomorphic Encryption Standard v1.1](https://homomorphicencryption.org/wp-content/uploads/2018/11/HomomorphicEncryptionStandardv1.1.pdf).


### CKKS Parameters: LeNet-5
- **Ciphertexts depth**: 29
- **log PQ**: 1624
- **Cyclotomic Order**: 131072
- **Ring dimension**: 65536
- **Number of Slots**: 32768


### CKKS Parameters: ResNet-20
- **Ciphertexts depth**: 29
- **log PQ**: 1862
- **Cyclotomic Order**: 262144
- **Ring dimension**: 131072
- **Number of Slots**: 65536


### Optimized Performance

The `client_key_generation` utilities also provide equivalent ring dimensions and slot counts that target smaller security levels. These configurations are designed to significantly improve computation speed and reduce memory overhead, consistent with the performance benchmarks presented in the FHEON research paper.

---

## Technical Details

### Execution Paths
The inference executables typically expect external weights and keys provided at runtime:
- **Weights**: `submissions/<model_name>/weights/`
- **Keys**: Generated and managed via model-specific `client_key_generation` utilities.