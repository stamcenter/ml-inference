# MNIST Submission

This project implements an encrypted MLP inference pipeline for the MNIST benchmark using the FHEON framework
(FHEON's `HEDiagonalSingleModels/MLPDiagonal`).

## Model architecture

A three-layer fully connected network, the same shape as the harness plaintext model (`harness/mnist/model.py`):

- `fc1`: 784 -> 128
- ReLU
- `fc2`: 128 -> 64
- ReLU
- `fc3`: 64 -> 10

Every layer is evaluated with the FHEON diagonal primitives (baby-step giant-step rotations, one level per layer).
Each layer returns its outputs in the first slots of a ciphertext with as many slots as it has diagonals:

| Layer | Input slots | Diagonals |
|-------|-------------|-----------|
| `fc1` | 1024        | 128       |
| `fc2` | 128         | 64        |
| `fc3` | 64          | 16        |

ReLU is a degree-27 Chebyshev approximation on [-1, 1]. The ReLU scales are fixed (10 after `fc1`, 13 after `fc2`):
the largest pre-activations of the plaintext network over the MNIST training and test sets are 8.6 and 11.7, so every
ReLU input stays within [-1, 1]. `fc1` and `fc2` produce their outputs already divided by the scale, and the next layer
multiplies it back, so no ReLU spends a level on scaling and the server never decrypts.

Plaintext simulation of the circuit over the 10000 MNIST test images: 98.26% (98.21% with the exact ReLU).

## Structure

- `src/` — model logic (`mlp_fheon.cpp`) and benchmark executables
- `fheon/` — FHEON controller implementations
- `include/` — headers and configuration (`mlp_fheon.h`)
- `weights/mlp/` — trained model weights (FHEON's `basicmlp` weights)
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

The default configuration targets 128-bit security (`HEStd_128_classic`). The circuit is leveled (13 levels, no
bootstrapping), and the ring optimization of `FHEONHEController` picks ring dimension 2^14 with a 24-bit scale and
8 key-switching digits (log QP = 425, the standard allows 438). The 784 pixels are packed into 1024 of the 8192 slots.

Setting `performanceMode = true` in `include/mlp_fheon.h` uses the parameters of MLPDiagonal instead: ring 2^11,
32-bit scale, not a standard security level. Rebuild and rerun key generation after changing it.

| Setting      | Ring | Keys   | Encrypted inference per image | `small` accuracy |
|--------------|------|--------|-------------------------------|------------------|
| 128-bit      | 2^14 | 708 MB | 1.6 s                         | 98/100           |
| Performance  | 2^11 | 17 MB  | 0.11 s                        | 98/100           |

## Notes

This submission is designed to work with the repository's ML-inference harness and follows the standard MNIST benchmark
pipeline. Run it with `--dataset mnist --model mlp`.
