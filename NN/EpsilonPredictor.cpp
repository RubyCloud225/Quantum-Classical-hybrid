#include "EpsilonPredictor.hpp"
#include <vector>

EpsilonPredictor::EpsilonPredictor(int input_channels, int output_size) {
    nn_.addConvolutionalLayer(input_channels, 64, 3, 1); // conv layer
    nn_.addReluLayer(); // Activation
    nn_.addPoolingLayer(2, 2, 2, 2, 0); // pooling layer using max pooling
    nn_.addConvolutionalLayer(64, 128, 3, 1); // second conv layer
    nn_.addReluLayer(); // activation
    nn_.addPoolingLayer(2, 2, 2, 2, 0); // pooling layer
    nn_.addFlatten(); // Flatten the output
    nn_.addLayer(FullyConnectedLayer(128 * 7 * 7, output_size)); // output layer
}

std::vector<double> EpsilonPredictor::predict(const std::vector<double>& x_t, int t) {
    std::vector<double> input = x_t;
    return nn_.forward(input, t);
}