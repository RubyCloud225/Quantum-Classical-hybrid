#include "NeuralNetwork.hpp"

// Constructor
NeuralNetwork::NeuralNetwork() {
    // You can initialize any parameters or settings here if needed
}

// Add a convolutional layer to the network
void NeuralNetwork::addConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride) {
    ConvolutionalLayer convLayer(input_channels, output_channels, kernel_size, stride);
    convLayers_.push_back(convLayer);
}

void NeuralNetwork::addReluLayer() {
    ReLu relu;
    reluLayers_.push_back(relu);
}

void NeuralNetwork::addPoolingLayer(int inputHeight, int inputWidth, int poolSize, int stride, int padding) {
    PoolingLayer poolingLayer(inputHeight, inputWidth, poolSize, stride, padding);
    poolingLayers_.push_back(poolingLayer);
}

// Forward pass through the network
std::vector<std::vector<std::vector<double>>> NeuralNetwork::forward(const std::vector<std::vector<double>>& input) {
    // Wrap the input in a 3D vector
    std::vector<std::vector<std::vector<double>>> currentOutput(1, input); // 1 channel

    // ConvolutionalLayer
    for (const auto& convLayer : convLayers_) {
        currentOutput = convLayer.forwardPass(currentOutput);
    }
    for (const auto& relu : reluLayers_) {
        currentOutput = relu.forward(currentOutput);
    }
    for (const auto& poolingLayer : poolingLayers_) {
        currentOutput = poolingLayer.forward(currentOutput);
    }

    return currentOutput; // Return the final output
}