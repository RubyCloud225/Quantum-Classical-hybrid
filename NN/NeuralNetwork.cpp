#include "NeuralNetwork.hpp"

// Constructor
NeuralNetwork::NeuralNetwork() {
    // You can initialize any parameters or settings here if needed
}

// Add a convolutional layer to the network
void NeuralNetwork::addConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride) {
    layers_.emplace_back(input_channels, output_channels, kernel_size, stride);
}

// Forward pass through the network
std::vector<std::vector<std::vector<double>>> NeuralNetwork::forward(const std::vector<std::vector<double>>& input) {
    // Wrap the input in a 3D vector
    std::vector<std::vector<std::vector<double>>> current_input(1, input); // 1 channel

    // Pass the input through each layer
    for (auto& layer : layers_) {
        current_input = layer.forwardPass(current_input);
    }

    return current_input; // Return the final output
}