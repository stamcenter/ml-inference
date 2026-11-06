## Workload implementation – ml inference
--------------------------------------

The submission is built with the [FHEON](https://fheon.pqcsecure.org/) Framework.

<<<<<<< HEAD:submission/docs/README.md
## Model architecture  and changes
This submission is based on the classic LeNet-5 model.
We assume that server has access to training model weights exported as CSV files and placed in the  `weights/lenet5` folder. 
The model architecture is as shown below:
- The convolution layers are configured with a `5x5` kernel window, padding of `0` and stride of `1` layer.
- The Average Pooling layers are configured with a stride of `2`.
- The activation layer, using Approx-RELU based on polynomial appox configured with a polynomial degree of `119`
- The first FC layer maps 256x120
- The second FC layer maps 120x84
- The third FC layer maps 84x10 output labels.

=======
## Model architecture changes
We assume that server has access to training data and the model architecture has been modified as follows
- The first FC layer, of size 512x784
- The activation layer, using Approx-RELU based on polynomial appox

This network is trained to get a plaintext accuracy of approximately 95%.
>>>>>>> upstream/main:submissions/mlp/docs/README.md

## FHEON details
FHEON is a configurable framework for developing privacy-preserving convolutional neural networks (CNNs) under homomorphic encryption (HE). FHEON adopts the Residue Number System (RNS) variant of CKKS as implemented in OpenFHE providing implementations of different neural network layers such as convolution, pooling, FCs and activiation functions

<<<<<<< HEAD:submission/docs/README.md
See further details of FHEON on https://arxiv.org/abs/2510.03996

## Build details
Weights are placed in `weights/lenet5` folder. 
The FHEON source code is placed in `fheonsrc` folder. 
The FHEON header files are placed in the `include` folder. 
The LeNet-5 model developed is in the `lenet5_fheon.cpp` file.
The `client_key_generation.cpp` file was modified to support the required crypto context.
All required rotation keys for the `lenet5` model were inlined. 
The `CMakeLists.txt` file is used to build and link the FHEON library
=======
See further details of compilation [here](https://github.com/google/heir/blob/3ed0da33f81984b32a32f8490e2de1f07ed14c03/tests/Examples/openfhe/ckks/mnist/BUILD#L6)
>>>>>>> upstream/main:submissions/mlp/docs/README.md
