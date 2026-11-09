# MNIST Submission

This project implements an encrypted LeNet-5 inference pipeline for the MNIST benchmark using the FHEON framework.

## Model architecture

The workload uses a LeNet-5-style CNN adapted for fully homomorphic encrypted inference:

- `conv1`: 5x5, 1 -> 6
- ReLU
- average pooling
- `conv2`: 5x5, 6 -> 16
- ReLU
- average pooling
- `fc1`: 256 -> 120
- ReLU
- `fc2`: 120 -> 84
- ReLU
- `fc3`: 84 -> 10

This is implemented using FHEON diagonal primitives and optimized for encrypted execution over the MNIST dataset.

## Structure

- `src/` — model logic and benchmark executables
- `fheon/` — FHEON controller implementations
- `include/` — headers and configuration
- `weights/lenet5/` — trained model weights
- `build/` — generated CMake artifacts

## Build

```bash
cd submissions/mnist
cmake -S . -B build
cmake --build build
```

## Benchmark flow

1. key generation
2. model preprocessing
3. input preprocessing
4. input encryption
5. encrypted inference
6. decryption
7. postprocessing

## Security

The default configuration targets 128-bit security, and a faster performance-oriented mode is available through the project settings.

## Notes

This submission is designed to work with the repository's ML-inference harness and follows the standard MNIST benchmark pipeline.
