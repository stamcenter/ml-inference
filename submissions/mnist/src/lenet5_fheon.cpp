
/***********************************************************************************************************************
*
* @author: Nges Brian, Njungle
*
* MIT License
* Copyright (c) 2025 Secure, Trusted and Assured Microelectronics, Arizona State
University

* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:

* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.

* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
********************************************************************************************************************/

#include "lenet5_fheon.h"
#include <iostream>
#include <sys/stat.h>

using namespace std;
using namespace lbcrypto;

#ifndef WEIGHTS_DIR
#define WEIGHTS_DIR "./submissions/mnist/weights/lenet5/"
#endif

/** fixed ReLU scales: the server never decrypts, so the scales cannot be read from the activations as FHEON's
 *  LeNet5Diagonal does for the two convolutions. Every scale bounds |x| at its ReLU for every image. The fully connected
 *  scales are the ones of LeNet5Diagonal (largest pre-activations over the MNIST test set 6.8 and 9.3; 6.9 and 9.6 over
 *  the training set). The convolution scales are the largest pre-activations of the plaintext network over the MNIST
 *  training and test sets (9.7 and 8.4), +5%, rounded up. Plaintext simulation of the circuit (degree 27, these scales)
 *  over the 10000 test images: 99.39% (exact ReLU 99.27%) */
const double conv1ReluScale = 11;
const double conv2ReluScale = 9;
const int fc1ReluScale = 8;
const int fc2ReluScale = 10;

FHEONHEController::HEConfig lenet5_he_config(const string &keysFolder) {
    FHEONHEController::HEConfig heinitConfig;
    bool useBootstrapping = lenet5_uses_bootstrapping();
    heinitConfig.ringDim = 13;
    heinitConfig.numSlots = 12;
    /** at a standard security level the ring holds more slots (2^15 at 128 bits): conv2 then runs its 16 channels as
     *  16 parallel groups instead of 4 groups of 4 interleaved channels (see the layout below) */
    heinitConfig.fillSlots = true;
    heinitConfig.mlevelBootstrap = useBootstrapping ? max(postBootstrapLevels + 2, preBootstrapLevels + 1 - bootstrapDepth)
                                                    : preBootstrapLevels + postBootstrapLevels + 1;
    heinitConfig.dcrtBits = 46;
    heinitConfig.firstMod = 50;
    heinitConfig.numDigits = 4;
    heinitConfig.levelBudget = {3, 3};
    /** standard security levels: a leveled chain (no bootstrapping) with a small scale. The ring optimization brings it
     *  to 2^15 at 128 bits: the chain stays precise down to a 22-bit scale, the first modulus keeps 6 bits above the
     *  scale for the logits of the last level */
    if (!useBootstrapping) {
        heinitConfig.firstMod = heinitConfig.dcrtBits + 6;
        heinitConfig.minScaleBits = 23;
    }
    heinitConfig.serialize = true;
    heinitConfig.keysFolder = keysFolder;
    /** no keys with the context: the key file holds exactly the keys of the layers (the diagonal primitives never call EvalSum) */
    heinitConfig.contextBootstrapKeys = false;
    heinitConfig.contextSumKeys = false;
    return heinitConfig;
}

void lenet5_generate_context(FHEONHEController &fheonHEController, const string &keysFolder) {
    auto heinitConfig = lenet5_he_config(keysFolder);
    if (lenet5_uses_bootstrapping()) {
        fheonHEController.generate_context(heinitConfig);
    } else {
        fheonHEController.generate_leveled_context(heinitConfig);
    }
}

int lenet5_input_level(const CryptoContext<DCRTPoly> &context) {
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    return numTowers - (preBootstrapLevels + 2 + (lenet5_uses_bootstrapping() ? 0 : postBootstrapLevels));
}

/*************************************************** Layout of the network ************************************************************
 * No layer compacts its output: the rows keep the 28-slot stride of the image all the way.
 *  conv1 : (1,28,28) -> (6,24,24), channel c at 672*c, value (y,x) at 28*y + x
 *  pool1 : window sums left in place, value (Y,X) at 56*Y + 2*X (the 1/4 is in the conv2 weights)
 *  conv2 : (6,12,12) -> (16,8,8), 4 groups of 4 channels at 672*g; the 4 channels of a group are
 *          interleaved in the gaps left by pool1, value (y,x) at 56*y + 2*x + {0, 1, 28, 29}.
 *          With 16384 slots or more (standard security levels): 16 groups of 1 channel, channel c at 672*c
 *  pool2 : fused into fc1, which reads the 4 slots of every window (the 1/4 is in the fc1 weights)
 *  fc1, fc2, fc3 : diagonals (their outputs rounded up to a power of two), outputs in the first slots of ciphertexts of 128, 128 and 16 slots
 **************************************************************************************************************************************/
struct LeNet5Layouts {
    FHEONLayout conv1, conv2, pool1;
    vector<vector<int>> fc1Positions;
};

static LeNet5Layouts lenet5_layouts(ANNController &fheonANNController, int numSlots) {
    LeNet5Layouts layouts;
    int rowStride = imgWidth[0];
    int colStride = 1;
    int channelStride = imgWidth[1]*rowStride;
    /** multiplexed conv2: its 16 output channels are packed as numGroups groups of channelStride slots, the channels of a
     *  group interleaved in the gaps pool1 leaves (sub shifts). With the slots of a standard security level all 16 channels
     *  run in parallel, one per group */
    bool parallelConv2 = numSlots >= channels[2]*channelStride;
    vector<int> conv2SubShifts = parallelConv2 ? vector<int>{0} : vector<int>{0, colStride, rowStride, rowStride + colStride};
    int conv2Groups = channels[2]/conv2SubShifts.size();

    layouts.conv1.rowStride = rowStride;
    layouts.conv1.colStride = colStride;
    layouts.conv1.channelStride = channelStride;
    layouts.conv1.subShifts = {0};
    layouts.conv2.rowStride = 2*rowStride;
    layouts.conv2.colStride = 2*colStride;
    layouts.conv2.channelStride = channelStride;
    layouts.conv2.subShifts = conv2SubShifts;
    layouts.conv2.numGroups = conv2Groups;
    /** the pooling returns window sums: the next layer takes the 1/4 in its weights */
    layouts.conv2.weightScale = fheonANNController.avgpool_weight_scale(2);
    layouts.pool1.rowStride = rowStride;
    layouts.pool1.colStride = colStride;

    /** slots holding every input of fc1: the 2x2 window of (channel, Y, X) in the conv2 output */
    layouts.fc1Positions.resize(channels[3]);
    for (int ch = 0; ch < channels[2]; ch++) {
        int groupStart = (ch/conv2SubShifts.size())*channelStride + conv2SubShifts[ch % conv2SubShifts.size()];
        for (int row = 0; row < imgWidth[3]; row++) {
            for (int col = 0; col < imgWidth[3]; col++) {
                int input = ch*imgWidth[4]*imgWidth[4] + (row/2)*imgWidth[4] + (col/2);
                layouts.fc1Positions[input].push_back(groupStart + row*2*rowStride + col*2*colStride);
            }
        }
    }
    return layouts;
}

vector<int> lenet5_rotation_keys(ANNController &fheonANNController, int numSlots) {
    LeNet5Layouts layouts = lenet5_layouts(fheonANNController, numSlots);
    FHEONLayout fcKeysLayout, fc3KeysLayout;
    fcKeysLayout.inputSlots = numSlots;
    fc3KeysLayout.inputSlots = fcDiagonals;
    vector<vector<int>> rotation_keys;
    rotation_keys.push_back(fheonANNController.generate_convolution_rotation_positions(imgWidth[0], channels[0], channels[1], kernelWidth, 0, 1, layouts.conv1));
    rotation_keys.push_back(fheonANNController.generate_avgpool_rotation_positions(imgWidth[1], 2, 2, channels[1], layouts.pool1));
    rotation_keys.push_back(fheonANNController.generate_convolution_rotation_positions(imgWidth[2], channels[1], channels[2], kernelWidth, 0, 1, layouts.conv2));
    rotation_keys.push_back(fheonANNController.generate_linear_rotation_positions(channels[4], 0, fcKeysLayout));
    rotation_keys.push_back(fheonANNController.generate_linear_rotation_positions(channels[6], 0, fc3KeysLayout));
    return serialize_rotation_keys(rotation_keys);
}

/*************************************************** Weights of the network **************************************************/

LeNet5::LeNet5(FHEONHEController &heController, ANNController &annController)
    : fheonHEController(heController), fheonANNController(annController) {
    int numSlots = fheonHEController.num_slots;
    LeNet5Layouts layouts = lenet5_layouts(fheonANNController, numSlots);
    conv1Layout = layouts.conv1;
    conv2Layout = layouts.conv2;
    pool1Layout = layouts.pool1;
    cout << "         [server] conv2 layout: " << conv2Layout.numGroups << " groups of " << conv2Layout.subShifts.size()
         << " channel(s) on " << numSlots << " slots" << endl;

    /** the levels are fixed by the circuit: conv1 on the fresh ciphertext, 1 level per layer, reluLevels per ReLU that
     *  scales its input and chebyshevLevels per ReLU whose input is already scaled */
    int conv1Level = lenet5_input_level(fheonHEController.getContext());
    int conv2Level = conv1Level + 1 + reluLevels;
    int fc1Level = conv2Level + 1 + reluLevels;
    int fc2Level = fc1Level + 1 + (reluLevels - 1);
    string dataPath = string(WEIGHTS_DIR);

    auto conv1_biasVector = load_bias(dataPath+"Conv1_bias.csv");
    auto conv1_rawKernelData = load_weights(dataPath+"Conv1_weight.csv", channels[1], channels[0], kernelWidth, kernelWidth);
    conv1_kernelData = fheonANNController.encode_convolution_kernel(fheonHEController, conv1_rawKernelData, imgWidth[0], numSlots, conv1Level, 1, 0, conv1Layout);
    conv1biasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, conv1_biasVector, imgWidth[1], numSlots, conv1Level, conv1Layout);

    auto conv2_rawKernelData = load_weights(dataPath+"Conv2_weight.csv", channels[2], channels[1], kernelWidth, kernelWidth);
    auto conv2_biasVector = load_bias(dataPath+"Conv2_bias.csv");
    conv2_kernelData = fheonANNController.encode_convolution_kernel(fheonHEController, conv2_rawKernelData, imgWidth[2], numSlots, conv2Level, 1, 0, conv2Layout);
    conv2biasEncoded = fheonANNController.encode_convolution_bias(fheonHEController, conv2_biasVector, imgWidth[3], numSlots, conv2Level, conv2Layout);

    /** the fully connected layers produce their outputs already divided by the scale of their ReLU, and the next layer
     *  multiplies it back (ReLU(x / s) = ReLU(x) / s): the ReLUs save the level of the scaling, and the bootstrapping
     *  after fc2 works on values within [-1, 1] */
    auto fc1_biasVector = load_bias(dataPath+"FC1_bias.csv");
    auto fc1_rawKernelData = load_fc_weights(dataPath+"FC1_weight.csv", channels[4], channels[3]);
    for (auto &bias : fc1_biasVector) { bias /= fc1ReluScale; }
    fc1Layout.inputPositions = layouts.fc1Positions;
    fc1Layout.weightScale = fheonANNController.avgpool_weight_scale(2)/fc1ReluScale;
    fc1_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc1_rawKernelData, numSlots, fc1Level, fc1Layout);
    fc1baisVector = fheonANNController.encode_linear_bias(fheonHEController, fc1_biasVector, numSlots, fc1Level);

    auto fc2_biasVector = load_bias(dataPath+"FC2_bias.csv");
    auto fc2_rawKernelData = load_fc_weights(dataPath+"FC2_weight.csv", channels[5], channels[4]);
    for (auto &bias : fc2_biasVector) { bias /= fc2ReluScale; }
    fc2Layout.weightScale = static_cast<double>(fc1ReluScale)/fc2ReluScale;
    fc2_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc2_rawKernelData, fcDiagonals, fc2Level, fc2Layout);
    fc2baisVector = fheonANNController.encode_linear_bias(fheonHEController, fc2_biasVector, fcDiagonals, fc2Level);

    /** fc3 is encoded once the level after the bootstrapping is known */
    fc3_biasVector = load_bias(dataPath+"FC3_bias.csv");
    fc3_rawKernelData = load_fc_weights(dataPath+"FC3_weight.csv", channels[6], channels[5]);
    fc3Layout.weightScale = fc2ReluScale;
}

Ctext LeNet5::infer(const Ctext &encryptedImage) {
    Ctext input = encryptedImage;
    /***** conv1: image=(1,28,28), kernel=(6,1,5,5), stride=1, padding=0, output=(6,24,24) */
    cout << "         [server] Layer 1" << endl;
    auto convData = fheonANNController.he_convolution(input, conv1_kernelData, conv1biasEncoded, imgWidth[0], channels[0], channels[1], kernelWidth, 0, 1, conv1Layout);
    convData = fheonANNController.he_relu(convData, conv1ReluScale, convData->GetSlots(), reluDegree);
    convData = fheonANNController.he_avgpool(convData, imgWidth[1], channels[1], 2, 2, pool1Layout);

    /***** conv2: input=(6,12,12), kernel=(16,6,5,5), stride=1, padding=0, output=(16,8,8) */
    cout << "         [server] Layer 2" << endl;
    convData = fheonANNController.he_convolution(convData, conv2_kernelData, conv2biasEncoded, imgWidth[2], channels[1], channels[2], kernelWidth, 0, 1, conv2Layout);
    convData = fheonANNController.he_relu(convData, conv2ReluScale, convData->GetSlots(), reluDegree);

    /*** fully connected layers; the second pooling is part of the first one */
    cout << "         [server] FC 1" << endl;
    convData = fheonANNController.he_linear(convData, fc1_kernelData, fc1baisVector, channels[3], channels[4], 0);
    convData = fheonANNController.he_relu(convData, 1, convData->GetSlots(), reluDegree);

    cout << "         [server] FC 2" << endl;
    convData = fheonANNController.he_linear(convData, fc2_kernelData, fc2baisVector, channels[4], channels[5], 0);
    if (lenet5_uses_bootstrapping()) {
        convData = fheonHEController.bootstrap_function(convData, bootstrapIterations);
    }
    convData = fheonANNController.he_relu(convData, 1, convData->GetSlots(), reluDegree);

    cout << "         [server] FC 3" << endl;
    int levelNow = fheonANNController.convolution_kernel_level(convData, imgWidth[0]);
    if (levelNow != fc3Level) {
        fc3Level = levelNow;
        fc3_kernelData = fheonANNController.encode_linear_weights(fheonHEController, fc3_rawKernelData, fcDiagonals, fc3Level, fc3Layout);
        fc3baisVector = fheonANNController.encode_linear_bias(fheonHEController, fc3_biasVector, fcDiagonals, fc3Level);
    }
    return fheonANNController.he_linear(convData, fc3_kernelData, fc3baisVector, channels[5], channels[6], 0);
}
