/***********************************************************************************************************************
 *
 * @author: Nges Brian, Njungle
 *
 * MIT License
 * Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State University
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 **********************************************************************************************************************/

#include "lenet5_fheon.h"

#include <iostream>

/** both convolutions are "valid" (no padding) with stride 1; both poolings are 2x2 with stride 2 */
const int convStride = 1;
const int convPadding = 0;
const int poolWidth = 2;

/** 
 *  LeNet5Diagonal does for the two convolutions. Every scale bounds |x| at its ReLU for every image. The fully connected
 *  scales are the ones of LeNet5Diagonal (largest pre-activations over the MNIST test set 6.8 and 9.3; 6.9 and 9.6 over
 *  the training set). The convolution scales are the largest pre-activations of the plaintext network over the MNIST
 *  training and test sets (9.7 and 8.4), +5%, rounded up. Plaintext simulation of the circuit (degree 27, these scales)
 *  over the 10000 test images: 99.39% (exact ReLU 99.27%) */
const double conv1ReluScale = 11;
const double conv2ReluScale = 9;
const double fc1ReluScale = 8;
const double fc2ReluScale = 10;

int lenet5_input_level(const CryptoContext<DCRTPoly> &context) {
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    return numTowers - (preBootstrapLevels + 2 + (lenet5_uses_bootstrapping() ? 0 : postBootstrapLevels));
}

/*************************************************** Layout of the network ********************************************
 * No layer compacts its output: the rows keep the 28-slot stride of the image all the way.
 *  conv1 : (1,28,28) -> (6,24,24), channel c at 672*c, value (y,x) at 28*y + x
 *  pool1 : window sums left in place, value (Y,X) at 56*Y + 2*X (the 1/4 is in the conv2 weights)
 *  conv2 : (6,12,12) -> (16,8,8), 4 groups of 4 channels at 672*g; the 4 channels of a group are
 *          interleaved in the gaps left by pool1, value (y,x) at 56*y + 2*x + {0, 1, 28, 29}.
 *          With 16384 slots or more (standard security levels): 16 groups of 1 channel, channel c at 672*c
 *  pool2 : fused into fc1, which reads the 4 slots of every window (the 1/4 is in the fc1 weights)
 *  fc1, fc2, fc3 : diagonals (outputs rounded up to a power of two), outputs in the first slots of ciphertexts of
 *          128, 128 and 16 slots
 **********************************************************************************************************************/
LeNet5Layouts lenet5_layouts(ANNController &fheonANNController, int numSlots) {
    LeNet5Layouts layouts;
    int rowStride = imgWidth[0];
    int colStride = 1;
    int channelStride = imgWidth[1]*rowStride;
    /** multiplexed conv2: its 16 output channels are packed as numGroups groups of channelStride slots, the channels of a
     *  group interleaved in the gaps pool1 leaves (sub shifts). With the slots of a standard security level all 16
     *  channels run in parallel, one per group */
    bool parallelConv2 = numSlots >= channels[2]*channelStride;
    vector<int> conv2SubShifts = parallelConv2 ? vector<int>{0} : vector<int>{0, colStride, rowStride, rowStride + colStride};
    int channelsPerGroup = conv2SubShifts.size();

    layouts.conv1.rowStride = rowStride;
    layouts.conv1.colStride = colStride;
    layouts.conv1.channelStride = channelStride;
    layouts.conv1.subShifts = {0};

    layouts.pool1.rowStride = rowStride;
    layouts.pool1.colStride = colStride;

    layouts.conv2.rowStride = poolWidth*rowStride;
    layouts.conv2.colStride = poolWidth*colStride;
    layouts.conv2.channelStride = channelStride;
    layouts.conv2.subShifts = conv2SubShifts;
    layouts.conv2.numGroups = channels[2]/channelsPerGroup;
    /** the pooling returns window sums: the next layer takes the 1/4 in its weights */
    layouts.conv2.weightScale = fheonANNController.avgpool_weight_scale(poolWidth);

    /** slots holding every input of fc1: the 2x2 window of (channel, Y, X) in the conv2 output */
    layouts.fc1Positions.resize(channels[3]);
    for (int ch = 0; ch < channels[2]; ch++) {
        int groupStart = (ch/channelsPerGroup)*channelStride + conv2SubShifts[ch % channelsPerGroup];
        for (int row = 0; row < imgWidth[3]; row++) {
            for (int col = 0; col < imgWidth[3]; col++) {
                int input = ch*imgWidth[4]*imgWidth[4] + (row/poolWidth)*imgWidth[4] + (col/poolWidth);
                layouts.fc1Positions[input].push_back(groupStart + row*layouts.conv2.rowStride + col*layouts.conv2.colStride);
            }
        }
    }
    return layouts;
}

/*************************************************** Weight encoding *************************************************/

LeNet5::LeNet5(FHEONHEController &heController, ANNController &annController, LeNet5Weights weights)
    : fheonHEController(heController), fheonANNController(annController) {
    int numSlots = fheonHEController.num_slots;
    LeNet5Layouts layouts = lenet5_layouts(fheonANNController, numSlots);
    conv1Layout = layouts.conv1;
    pool1Layout = layouts.pool1;
    conv2Layout = layouts.conv2;
    cout << "         [server] conv2 layout: " << conv2Layout.numGroups << " groups of " << conv2Layout.subShifts.size()
         << " channel(s) on " << numSlots << " slots" << endl;

    /** the levels are fixed by the circuit: conv1 on the fresh ciphertext, 1 level per layer, reluLevels per ReLU that
     *  scales its input and chebyshevLevels per ReLU whose input is already scaled */
    int conv1Level = lenet5_input_level(fheonHEController.getContext());
    int conv2Level = conv1Level + 1 + reluLevels;
    int fc1Level = conv2Level + 1 + reluLevels;
    int fc2Level = fc1Level + 1 + chebyshevLevels;

    conv1Kernel = fheonANNController.encode_convolution_kernel(fheonHEController, weights.conv1Weight, imgWidth[0], numSlots,
                                                               conv1Level, convStride, convPadding, conv1Layout);
    conv1Bias = fheonANNController.encode_convolution_bias(fheonHEController, weights.conv1Bias, imgWidth[1], numSlots,
                                                           conv1Level, conv1Layout);

    conv2Kernel = fheonANNController.encode_convolution_kernel(fheonHEController, weights.conv2Weight, imgWidth[2], numSlots,
                                                               conv2Level, convStride, convPadding, conv2Layout);
    conv2Bias = fheonANNController.encode_convolution_bias(fheonHEController, weights.conv2Bias, imgWidth[3], numSlots,
                                                           conv2Level, conv2Layout);

    /** the fully connected layers produce their outputs already divided by the scale of their ReLU, and the next layer
     *  multiplies it back (ReLU(x / s) = ReLU(x) / s): the ReLUs save the level of the scaling, and the bootstrapping
     *  after fc2 works on values within [-1, 1] */
    FHEONLayout fc1Layout;
    fc1Layout.inputPositions = layouts.fc1Positions;
    fc1Layout.weightScale = fheonANNController.avgpool_weight_scale(poolWidth)/fc1ReluScale;
    for (auto &bias : weights.fc1Bias) { bias /= fc1ReluScale; }
    fc1Weights = fheonANNController.encode_linear_weights(fheonHEController, weights.fc1Weight, numSlots, fc1Level, fc1Layout);
    fc1Bias = fheonANNController.encode_linear_bias(fheonHEController, weights.fc1Bias, numSlots, fc1Level);

    FHEONLayout fc2Layout;
    fc2Layout.weightScale = fc1ReluScale/fc2ReluScale;
    for (auto &bias : weights.fc2Bias) { bias /= fc2ReluScale; }
    fc2Weights = fheonANNController.encode_linear_weights(fheonHEController, weights.fc2Weight, fcDiagonals, fc2Level, fc2Layout);
    fc2Bias = fheonANNController.encode_linear_bias(fheonHEController, weights.fc2Bias, fcDiagonals, fc2Level);

    /** fc3 is encoded on the first inference, once the level after the bootstrapping is known */
    fc3Layout.weightScale = fc2ReluScale;
    fc3RawWeights = std::move(weights.fc3Weight);
    fc3RawBias = std::move(weights.fc3Bias);
}

/*************************************************** Encrypted inference *********************************************/

Ctext LeNet5::infer(const Ctext &encryptedImage) {
    Ctext x = encryptedImage;

    /** conv1: (1,28,28) -> (6,24,24), ReLU, pool1 -> (6,12,12) */
    cout << "         [server] Layer 1" << endl;
    x = fheonANNController.he_convolution(x, conv1Kernel, conv1Bias, imgWidth[0], channels[0], channels[1], kernelWidth,
                                          convPadding, convStride, conv1Layout);
    x = fheonANNController.he_relu(x, conv1ReluScale, x->GetSlots(), reluDegree);
    x = fheonANNController.he_avgpool(x, imgWidth[1], channels[1], poolWidth, poolWidth, pool1Layout);

    /** conv2: (6,12,12) -> (16,8,8), ReLU; pool2 is part of fc1 */
    cout << "         [server] Layer 2" << endl;
    x = fheonANNController.he_convolution(x, conv2Kernel, conv2Bias, imgWidth[2], channels[1], channels[2], kernelWidth,
                                          convPadding, convStride, conv2Layout);
    x = fheonANNController.he_relu(x, conv2ReluScale, x->GetSlots(), reluDegree);

    /** fc1: 256 -> 120, ReLU (input already scaled) */
    cout << "         [server] FC 1" << endl;
    x = fheonANNController.he_linear(x, fc1Weights, fc1Bias, channels[3], channels[4], 0);
    x = fheonANNController.he_relu(x, 1, x->GetSlots(), reluDegree);

    /** fc2: 120 -> 84, bootstrapping at the performance-efficient level, ReLU (input already scaled) */
    cout << "         [server] FC 2" << endl;
    x = fheonANNController.he_linear(x, fc2Weights, fc2Bias, channels[4], channels[5], 0);
    if (lenet5_uses_bootstrapping()) {
        x = fheonHEController.bootstrap_function(x, bootstrapIterations);
    }
    x = fheonANNController.he_relu(x, 1, x->GetSlots(), reluDegree);

    /** fc3: 84 -> 10 logits */
    cout << "         [server] FC 3" << endl;
    int level = fheonANNController.convolution_kernel_level(x, imgWidth[0]);
    if (level != fc3Level) {
        fc3Level = level;
        fc3Weights = fheonANNController.encode_linear_weights(fheonHEController, fc3RawWeights, fcDiagonals, fc3Level, fc3Layout);
        fc3Bias = fheonANNController.encode_linear_bias(fheonHEController, fc3RawBias, fcDiagonals, fc3Level);
    }
    return fheonANNController.he_linear(x, fc3Weights, fc3Bias, channels[5], channels[6], 0);
}
