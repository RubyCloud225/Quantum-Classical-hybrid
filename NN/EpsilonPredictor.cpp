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
    nn_.addFullyConnectedLayer(128 * 7 * 7, output_size); // output layer
}

std::vector<int> EpsilonPredictor::predictEpilson(const std::vector<double>& x_t, int t) {
    // Wrap x_t to make it a 2D vector of doubles
    std::vector<std::vector<double>> input = {x_t}; 

    // Call the forward method of the neural network
    auto raw_result = nn_.forward(input); // Assuming nn_.forward returns a 3D vector of doubles

    // Flatten the 3D result and convert to integers
    std::vector<int> result;
    for (const auto& matrix : raw_result) { // Iterate over each 2D matrix
        for (const auto& row : matrix) {    // Iterate over each row
            for (double val : row) {   // Iterate over each value in the row
                result.push_back(static_cast<int>(val)); // Convert double to int and add to result
            }
        }
    }
    return result;
}