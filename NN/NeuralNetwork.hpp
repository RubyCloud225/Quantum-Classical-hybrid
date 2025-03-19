#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include "ConvolutionalLayer.hpp"

class NeuralNetwork {
    public:
    NeuralNetwork();
    void addConvolutionalLayer(int inputChannels, int outputChannels, int kernelSize, int stride);
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<double>>& input);

    private:
    std::vector<ConvolutionalLayer> layers_; // vector to hold the layers
};

#endif