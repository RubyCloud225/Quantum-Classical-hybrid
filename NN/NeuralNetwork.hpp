#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include "ConvolutionalLayer.hpp"
#include "ReluLayer.hpp"
#include "PoolingLayer.hpp"
#include "Flatten.hpp"

class NeuralNetwork {
    public:
    NeuralNetwork();
    void addConvolutionalLayer(int inputChannels, int outputChannels, int kernelSize, int stride);
    void addReluLayer();
    void addPoolingLayer(int inputHeight, int inputWidth, int outputHeight, int outputWidth, int stride);
    std::vector<std::vector<std::vector<double>>> forward(const std::vector<std::vector<double>>& input);
    void addFlatten();
  

    private:
    std::vector<ConvolutionalLayer> convLayers_; // vector to hold the layers
    std::vector<ReLu> reluLayers_; // vector to hold the ReLU layers
    std::vector<PoolingLayer> poolingLayers_; // vector to hold the pooling layers
    std::vector<Flatten> flatten_;
};

#endif