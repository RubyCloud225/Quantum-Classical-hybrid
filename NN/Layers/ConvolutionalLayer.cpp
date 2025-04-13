#include "ConvolutionalLayer.hpp"
#include <vector>

// Constructor
ConvolutionalLayer::ConvolutionalLayer(int input_channels, int output_channels, int kernel_size, int stride)
    : input_channels_(input_channels), output_channels_(output_channels), kernel_size_(kernel_size), stride_(stride) {
    initializeWeights();
}

// Initialize the weights of the layer
void ConvolutionalLayer::initializeWeights() {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0 / sqrt(input_channels_));
    kernels_.resize(output_channels_);
    for (int i = 0; i < output_channels_; i++) {
        kernels_[i].resize(input_channels_);
        for (int j = 0; j < input_channels_; j++) {
            kernels_[i][j].resize(kernel_size_ * kernel_size_); // Corrected to 1D representation
            for (int k = 0; k < kernel_size_ * kernel_size_; k++) {
                kernels_[i][j][k] = distribution(generator);
            }
        }
    }
    biases_.resize(output_channels_);
    for (int i = 0; i < output_channels_; i++) {
        biases_[i] = 0.0; // Initialize biases to zero
    }
}

// Forward pass
std::vector<std::vector<std::vector<double>>> ConvolutionalLayer::forwardPass(const std::vector<std::vector<std::vector<double>>>& input) const {
    int input_height = input.size();
    int input_width = input[0].size();
    int output_height = (input_height - kernel_size_) / stride_ + 1;
    int output_width = (input_width - kernel_size_) / stride_ + 1;

    // Initialize output tensor
    std::vector<std::vector<std::vector<double>>> output(output_channels_, 
    std::vector<std::vector<double>>(output_height, 
    std::vector<double>(output_width, 0.0)));

    std::vector<std::vector<std::vector<double>>> input(input_channels_, std::vector<std::vector<double>>(input_height, std::vector<double>(input_width, 0.0)));

    for (int oc = 0; oc < output_channels_; ++oc) { // For each output channel
        for (int i = 0; i < output_height; ++i) {
            for (int j = 0; j < output_width; ++j) {
                double sum = 0.0;
                for (int ic = 0; ic < input_channels_; ++ic) { // For each input channel
                    for (int ki = 0; ki < kernel_size_; ++ki) {
                        for (int kj = 0; kj < kernel_size_; ++kj) {
                            int input_row = i * stride_ + ki; // Use int for row index
                            int input_col = j * stride_ + kj; // Use int for column index
                            if (input_row >= 0 && input_row < input_height && input_col >= 0 && input_col < input_width)  {
                                double input_value = input[ic][input_row][input_col];
                                int kernel_index = ki * kernel_size_ + kj;
                                double kernel_weight = kernels_[oc][ic][kernel_index];
                                sum += input_value * kernel_weight;
                            }
                        }
                    }
                }
                output[oc][i][j] = sum + biases_[oc]; // Add bias to the sum
            }
        }
    }
    return output; // Return the computed output
}
