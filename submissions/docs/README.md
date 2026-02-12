
## Workload implementation – ml inference
--------------------------------------

This submission contains two models:
1.  **LeNet-5**: Built with the [FHEON](https://fheon.pqcsecure.org/) Framework.
2.  **MLP**: Built directly on [OpenFHE](https://openfhe-development.readthedocs.io).

## Model 1: LeNet-5 (FHEON)

### Model architecture
This submission is based on the classic LeNet-5 model.
The model architecture is as shown below:
- The convolution layers are configured with a `5x5` kernel window, padding of `0` and stride of `1` layer.
- The Average Pooling layers are configured with a stride of `2`.
- The activation layer, using Approx-RELU based on polynomial appox configured with a polynomial degree of `119`
- The first FC layer maps 256x120
- The second FC layer maps 120x84
- The third FC layer maps 84x10 output labels.


### Build details
- **Weights**: Placed in `weights/lenet5` folder. 
- **Source Code**: FHEON source is in `fheon` folder, and headers in `include/fheon`.
- **Implementation**: The LeNet-5 model is implemented in `lenet5/src/lenet5_fheon.cpp`.
- **Key Generation**: `lenet5/src/client_key_generation.cpp` is modified to support the required crypto context.

## Model 2: MLP (OpenFHE)

### Model architecture
The MLP model is a fully connected network implemented directly using OpenFHE primitives.

### Build details
- **Implementation**: The MLP model is implemented in `mlp/src/mlp_openfhe.cpp` and `mlp/src/server_encrypted_compute.cpp`.
- **Key Generation**: `mlp/src/client_key_generation.cpp` handles the crypto context generation.

## Common Utilities
Common utility source files for both models (e.g., encryption helpers, data loading) are located in `common/src`.

## Building
The `CMakeLists.txt` files in `lenet5` and `mlp` folders are used to build and link the respective executables.