## Workload implementation – ml inference
--------------------------------------

The submission is built with the [FHEON](https://fheon.pqcsecure.org/) Framework.

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


## FHEON details
FHEON is a configurable framework for developing privacy-preserving convolutional neural networks (CNNs) under homomorphic encryption (HE). FHEON adopts the Residue Number System (RNS) variant of CKKS as implemented in OpenFHE providing implementations of different neural network layers such as convolution, pooling, FCs and activiation functions

See further details of FHEON on https://arxiv.org/abs/2510.03996

## Build details
Weights are placed in `weights/lenet5` folder. 
The FHEON source code is placed in `fheonsrc` folder. 
The FHEON header files are placed in the `include` folder. 
The LeNet-5 model developed is in the `lenet5_fheon.cpp` file.
The `client_key_generation.cpp` file was modified to support the required crypto context.
All required rotation keys for the `lenet5` model were inlined. 
The `CMakeLists.txt` file is used to build and link the FHEON library