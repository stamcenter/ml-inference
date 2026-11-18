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

#include "mlp_fheon.h"

#include <iostream>

/** Fixed ReLU scales: MLPDiagonal reads the scale of each ReLU from the decrypted pre-activations, which the server
 *  cannot do. Every scale bounds |x| at its ReLU for every image: the largest pre-activations of the plaintext network
 *  over the MNIST training and test sets are 8.6 (fc1) and 11.7 (fc2), +5%, rounded up. Plaintext simulation of the
 *  circuit (degree 27, these scales) over the 10000 test images: 98.26% (exact ReLU 98.21%), largest ReLU input 0.95 */
const double fc1ReluScale = 10;
const double fc2ReluScale = 13;

int mlp_input_level(const CryptoContext<DCRTPoly> &context) {
    int numTowers = context->GetCryptoParameters()->GetElementParams()->GetParams().size();
    return numTowers - (circuitLevels + 2);
}

vector<int> mlp_rotation_keys(ANNController &fheonANNController) {
    vector<vector<int>> rotation_keys;
    for (int layer = 0; layer < 3; layer++) {
        FHEONLayout keysLayout;
        keysLayout.inputSlots = fcSlots[layer];
        rotation_keys.push_back(fheonANNController.generate_linear_rotation_positions(channels[layer + 1], 0, keysLayout));
    }
    return serialize_rotation_keys(rotation_keys);
}

/*************************************************** Weight encoding *************************************************
 *  fc1 : 784 -> 128, on the 1024-slot input,  128 diagonals
 *  fc2 : 128 -> 64,  on a 128-slot ciphertext, 64 diagonals
 *  fc3 : 64 -> 10,   on a 64-slot ciphertext,  16 diagonals
 * fc1 and fc2 produce their outputs already divided by the scale of their ReLU, and the next layer multiplies it back
 * (ReLU(x / s) = ReLU(x) / s): the ReLUs evaluate their approximation on [-1, 1] without a scaling level.
 **********************************************************************************************************************/
MLP::MLP(FHEONHEController &heController, ANNController &annController, MLPWeights weights)
    : fheonHEController(heController), fheonANNController(annController) {
    /** the levels are fixed by the circuit: 1 level per layer and chebyshevLevels per ReLU */
    int fc1Level = mlp_input_level(fheonHEController.getContext());
    int fc2Level = fc1Level + 1 + chebyshevLevels;
    int fc3Level = fc2Level + 1 + chebyshevLevels;

    FHEONLayout fc1Layout;
    fc1Layout.weightScale = 1.0/fc1ReluScale;
    for (auto &bias : weights.fc1Bias) { bias /= fc1ReluScale; }
    fc1Weights = fheonANNController.encode_linear_weights(fheonHEController, weights.fc1Weight, fcSlots[0], fc1Level, fc1Layout);
    fc1Bias = fheonANNController.encode_linear_bias(fheonHEController, weights.fc1Bias, fcSlots[0], fc1Level);

    FHEONLayout fc2Layout;
    fc2Layout.weightScale = fc1ReluScale/fc2ReluScale;
    for (auto &bias : weights.fc2Bias) { bias /= fc2ReluScale; }
    fc2Weights = fheonANNController.encode_linear_weights(fheonHEController, weights.fc2Weight, fcSlots[1], fc2Level, fc2Layout);
    fc2Bias = fheonANNController.encode_linear_bias(fheonHEController, weights.fc2Bias, fcSlots[1], fc2Level);

    FHEONLayout fc3Layout;
    fc3Layout.weightScale = fc2ReluScale;
    fc3Weights = fheonANNController.encode_linear_weights(fheonHEController, weights.fc3Weight, fcSlots[2], fc3Level, fc3Layout);
    fc3Bias = fheonANNController.encode_linear_bias(fheonHEController, weights.fc3Bias, fcSlots[2], fc3Level);
}

/*************************************************** Encrypted inference *********************************************/

Ctext MLP::infer(const Ctext &encryptedImage) {
    Ctext x = encryptedImage;

    /** fc1: 784 -> 128, ReLU (input already scaled) */
    x = fheonANNController.he_linear(x, fc1Weights, fc1Bias, channels[0], channels[1], 0);
    x = fheonANNController.he_relu(x, 1, x->GetSlots(), reluDegree);

    /** fc2: 128 -> 64, ReLU (input already scaled) */
    x = fheonANNController.he_linear(x, fc2Weights, fc2Bias, channels[1], channels[2], 0);
    x = fheonANNController.he_relu(x, 1, x->GetSlots(), reluDegree);

    /** fc3: 64 -> 10 logits */
    return fheonANNController.he_linear(x, fc3Weights, fc3Bias, channels[2], channels[3], 0);
}
