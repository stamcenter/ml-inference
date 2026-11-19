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

#include "vgg11_fheon.h"

/** convolutions: 3x3, padding 1, stride 1 */
const int convPadding = 1;
const int convStride = 1;
const int poolWidth = 2;

/** Fixed ReLU scales. VGG11Diagonal divides every ReLU input by its largest value, read from the decrypted activations
 *  of the image, which the server cannot do. reluScales[r] bounds |x| at ReLU r for every image (the Chebyshev
 *  approximation diverges outside [-scale, scale]): the largest pre-activations of the plaintext network over the CIFAR-10
 *  training and test sets (60000 images), +5%, rounded up to 0.1.
 *  The ciphertexts hold the true activations (as in the ResNet-20 submission): every ReLU scales its input by 1/scale and
 *  multiplies its output back. Keeping the activations normalized within [0, 1] instead (VGG11Diagonal) leaves values of
 *  about 0.1 at the bootstrappings, whose error then dominates: logits off by up to 1.
 *  Plaintext simulation of the circuit (degree 119, these scales) over the 10000 test images: 86.37% (exact ReLU 87.56%;
 *  degree 27: 15%, degree 59: 69%), largest ReLU input 0.887 of its scale */
static const vector<double> reluScales = {
    14.3, 22.6, 12.8, 9.6, 6.1, 7.6, 5.3, 10.2,     // conv1 - conv8
    13.1, 15.6};                                    // fc1, fc2

vector<int> vgg11_layer_stages() {
    vector<int> stages;
    int stage = 0;
    for (const auto &conv : vgg11Convs) {
        stages.push_back(stage);
        if (conv.poolAfter) { stage++; }
    }
    return stages;
}

vector<int> vgg11_stage_widths() {
    vector<int> widths;
    for (int stage = 0; stage < numStages; stage++) { widths.push_back(imgCols >> stage); }
    return widths;
}

vector<int> vgg11_stage_slots() {
    vector<int> widths = vgg11_stage_widths(), layerStages = vgg11_layer_stages(), slots(numStages, minimumSlots);
    for (size_t i = 0; i < vgg11Convs.size(); i++) {
        int stage = layerStages[i];
        slots[stage] = max(slots[stage], vgg11Convs[i].outputChannels*widths[stage]*widths[stage]);
    }
    return slots;
}

string vgg11_stage_key_file(int stage) {
    return "stage" + to_string(stage+1) + ".bin";
}

/** a bootstrapping is followed by a ReLU and a convolution or linear layer (reluLevels + 1 levels), or by a pooling and a
 *  convolution. When the two channel stages of the pooling downsampling do not fit in those levels, a single stage is used */
static FHEONLayout vgg11_pool_layout(int inputWidth, int inputChannels, int outputSlots) {
    FHEONLayout layout;
    layout.outputSlots = outputSlots;
    int twoStageLevels = static_cast<int>(log2(inputWidth/2)) + 1 + 2 + 1;
    layout.channelGroup = (twoStageLevels > reluLevels + 1) ? inputChannels : 0;
    return layout;
}

vector<int> vgg11_stage_rotation_keys(ANNController &fheonANNController, int stage) {
    vector<int> widths = vgg11_stage_widths(), slots = vgg11_stage_slots(), layerStages = vgg11_layer_stages();
    int width = widths[stage];
    vector<vector<int>> keys;
    for (size_t i = 0; i < vgg11Convs.size(); i++) {
        if (layerStages[i] != stage) continue;
        const auto &conv = vgg11Convs[i];
        FHEONLayout convLayout;
        convLayout.inputSlots = slots[stage];
        keys.push_back(fheonANNController.generate_convolution_rotation_positions(width, conv.inputChannels, conv.outputChannels,
                                                                                 kernelWidth, convPadding, convStride, convLayout));
        if (conv.poolAfter) {
            keys.push_back(fheonANNController.generate_avgpool_rotation_positions(width, poolWidth, poolWidth, conv.outputChannels,
                                                                                 vgg11_pool_layout(width, conv.outputChannels, slots[stage+1])));
        }
    }
    /** the fully connected layers run with the keys of the last stage, which has their slot count */
    if (stage == numStages - 1) {
        FHEONLayout fcLayout;
        fcLayout.inputSlots = fcWidth;
        keys.push_back(fheonANNController.generate_linear_rotation_positions(fcWidth, 0, fcLayout));
        keys.push_back(fheonANNController.generate_linear_rotation_positions(numClasses, 0, fcLayout));
    }
    return serialize_rotation_keys(keys);
}

int vgg11_input_level(const CryptoContext<DCRTPoly> &context) {
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    return max(1, numTowers - (1 + reluLevels + 1 + 3));
}

/*************************************************** Layers *********************************************************/

/** the layer being run, in the log of the server (independent of the FHEON verbosity) */
static void print_layer(const string &layer) {
    cout << "         [server] " << layer << endl;
}

/** 3x3 convolution with the weights of layer (multiplied by weightScale) encoded at the level of the input */
static Ctext convolution_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, VGG11Weights &weights,
                               const VGG11Conv &conv, Ctext &x, int dataWidth, double weightScale) {
    print_layer(conv.name);
    int numSlots = x->GetSlots();
    int level = fheonANNController.convolution_kernel_level(x, dataWidth);
    FHEONLayout layout;
    layout.weightScale = weightScale;
    auto kernel = fheonANNController.encode_convolution_kernel(fheonHEController, weights.conv.at(conv.name), dataWidth, numSlots,
                                                               level, convStride, convPadding, layout);
    auto bias = fheonANNController.encode_convolution_bias(fheonHEController, weights.bias.at(conv.name), dataWidth, numSlots, level);
    return fheonANNController.he_convolution(x, kernel, bias, dataWidth, conv.inputChannels, conv.outputChannels, kernelWidth,
                                            convPadding);
}

/** linear layer on the fcWidth slots, its weights multiplied by weightScale */
static Ctext linear_block(FHEONHEController &fheonHEController, ANNController &fheonANNController, VGG11Weights &weights,
                          const string &layer, Ctext &x, int inputSize, int outputSize, const vector<vector<int>> &inputPositions,
                          double weightScale) {
    print_layer(layer);
    FHEONLayout layout;
    layout.inputPositions = inputPositions;
    layout.weightScale = weightScale;
    int numSlots = x->GetSlots();
    int level = fheonANNController.convolution_kernel_level(x, 1);
    auto fcWeights = fheonANNController.encode_linear_weights(fheonHEController, weights.linear.at(layer), numSlots, level, layout);
    Ptext fcBias = fheonANNController.encode_linear_bias(fheonHEController, weights.bias.at(layer), numSlots, level);
    return fheonANNController.he_linear(x, fcWeights, fcBias, inputSize, outputSize, 0, layout);
}

/** ReLU r with its fixed scale */
static Ctext relu_block(ANNController &fheonANNController, Ctext &x, int relu) {
    return fheonANNController.he_relu(x, reluScales[relu], x->GetSlots(), reluDegree);
}

/*************************************************** Encrypted inference *********************************************/

/** access to the context registry of OpenFHE (AddContext is protected) */
struct ContextRegistry : CryptoContextFactory<DCRTPoly> {
    static void register_only(const CryptoContext<DCRTPoly> &context) {
        ReleaseAllContexts();
        AddContext(context);
    }
};

/** the keys of a stage in place of the keys in memory, and the bootstrapping precomputations of its slot count */
static void use_stage_keys(FHEONHEController &fheonHEController, int stage) {
    vector<int> stageSlots = vgg11_stage_slots();
    fheonHEController.clear_context(stageSlots[(stage + numStages - 1) % numStages]);
    fheonHEController.read_evaluation_keys(fheonHEController.getContext(), vgg11_stage_key_file(stage));
    /** loading the keys releases the crypto contexts of OpenFHE: register the context of the server again, so that the
     *  next ciphertext read from a file belongs to it */
    ContextRegistry::register_only(fheonHEController.getContext());
    fheonHEController.prepare_bootstrapping(stageSlots[stage]);
}

/*************************************************** Layout of the network ********************************************
 * Every convolution is a 3x3 "same" convolution evaluated over channel diagonals; its channels sit in consecutive blocks
 * of width^2 slots. A pooling halves the width and repacks its output on the slots of the next stage, whose ciphertexts
 * (and bootstrappings) have fewer slots. The global average pooling (2x2) is part of fc1. fc1, fc2 and fc3 run on 1024
 * slots with the keys of the last stage.
 **********************************************************************************************************************/
Ctext vgg11_infer(FHEONHEController &fheonHEController, ANNController &fheonANNController, VGG11Weights &weights,
                  const Ctext &encryptedImage) {
    vector<int> widths = vgg11_stage_widths(), slots = vgg11_stage_slots();
    Ctext x = encryptedImage;
    int stage = 0;
    use_stage_keys(fheonHEController, stage);
    /** the factor of a pooling (1/4 for the window sums) goes into the weights of the next convolution */
    double weightScale = 1.0;
    for (size_t i = 0; i < vgg11Convs.size(); i++) {
        const auto &conv = vgg11Convs[i];
        int width = widths[stage];
        x = convolution_block(fheonHEController, fheonANNController, weights, conv, x, width, weightScale);
        weightScale = 1.0;
        /** conv1 and its ReLU run on the fresh ciphertext */
        if (i > 0) {
            x = fheonHEController.bootstrap_function(x, bootstrapIterations);
        }
        x = relu_block(fheonANNController, x, i);
        if (conv.poolAfter) {
            x = fheonHEController.bootstrap_function(x, bootstrapIterations);
            x = fheonANNController.he_avgpool(x, width, conv.outputChannels, poolWidth, poolWidth,
                                              vgg11_pool_layout(width, conv.outputChannels, slots[stage+1]));
            weightScale = fheonANNController.avgpool_weight_scale(poolWidth);
            stage++;
            use_stage_keys(fheonHEController, stage);
        }
    }

    /** global average pooling + fully connected layers, on ciphertexts of fcWidth slots */
    int lastWidth = widths[stage];
    int lastChannels = vgg11Convs.back().outputChannels;
    vector<vector<int>> fc1Positions(lastChannels), fcPositions(fcWidth);
    for (int ch = 0; ch < lastChannels; ch++) {
        for (int i = 0; i < lastWidth*lastWidth; i++) { fc1Positions[ch].push_back(ch*lastWidth*lastWidth + i); }
    }
    for (int i = 0; i < fcWidth; i++) { fcPositions[i] = {i}; }
    x->SetSlots(fcWidth);

    int convReLUs = vgg11Convs.size();
    x = linear_block(fheonHEController, fheonANNController, weights, "fc1", x, lastChannels, fcWidth, fc1Positions,
                     1.0/(lastWidth*lastWidth));
    x = fheonHEController.bootstrap_function(x, bootstrapIterations);
    x = relu_block(fheonANNController, x, convReLUs);
    x = linear_block(fheonHEController, fheonANNController, weights, "fc2", x, fcWidth, fcWidth, fcPositions,
                     1.0);
    x = fheonHEController.bootstrap_function(x, bootstrapIterations);
    x = relu_block(fheonANNController, x, convReLUs+1);
    return linear_block(fheonHEController, fheonANNController, weights, "fc3", x, fcWidth, numClasses, fcPositions, 1.0);
}
