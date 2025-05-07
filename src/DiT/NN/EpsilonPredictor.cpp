#include "EpsilonPredictor.hpp"
#include <vector>
#include <omp.h>

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
    std::vector<std::vector<std::vector<double>>> raw_result = nn_.forward(input); // Assuming nn_.forward returns a 3D vector of doubles
    int dim1 = raw_result.size(); // Get the first dimension size
    int dim2 = raw_result[0].size(); // Get the second dimension size
    int dim3 = raw_result[0][0].size(); // Get the third dimension siz

    

    // Flatten the 3D result and convert to integers
    std::vector<int> result(dim1 * dim2 * dim3); // Preallocate the result vector
    #pragma omp parallel for (const auto& matrix : raw_result) { // Iterate over each 2D matrix
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                for (int k = 0; k < dim3; ++k) {
                    int flat_idx = i * dim2 * dim3 + j * dim3 + k; // Calculate the flat index
                    result[flat_idx] = static_cast<int>(raw_result[j][k]); // Convert to int and store in the result vector
                }
            }
        }
    }
    return result;
}