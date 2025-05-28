#include "FullyConnected.hpp"
#include <vector>

// input connection- takes all the prevous outputs and calculates the weights
std::vector<std::vector<std::vector<double>>> Inputweights(const std::vector<std::vector<std::vector<double>>>& input) {
    std::vector<std::vector<std::vector<double>>> weights(input.size(), std::vector<std::vector<double>>(input[0].size(), std::vector<double>(input[0][0].size())));
    
    // Initialize weights with random values or zeros
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < input[i].size(); ++j) {
            for (size_t k = 0; k < input[i][j].size(); ++k) {
                weights[i][j][k] = 0.0; // or use a random initialization
            }
        }
    }
    
    return weights;
}
// z_j = sum(w_ij * x_i) + b_j

//Forward pass
std::vector<std::vector<std::vector<double>>> Forward(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& weights) {
    std::vector<std::vector<std::vector<double>>> output(input.size(), std::vector<std::vector<double>>(weights[0].size(), std::vector<double>(weights[0][0].size())));
    
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            for (size_t k = 0; k < weights[0][0].size(); ++k) {
                double sum = 0.0;
                for (size_t l = 0; l < input[i].size(); ++l) {
                    sum += input[i][l][k] * weights[l][j][k];
                }
                output[i][j][k] = sum; // Add bias if needed
            }
        }
    }
    
    return output;
}

// Activation function
// a_j = f(z_i)
std::vector<std::vector<std::vector<double>>> Activation(const std::vector<std::vector<std::vector<double>>>& input, std::vector<std::vector<std::vector<double>>> weights) const {
    std::vector<std::vector<std::vector<double>>> output(input.size(), std::vector<std::vector<double>>(weights[0].size(), std::vector<double>(weights[0][0].size())));
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            for (size_t k = 0; k < weights[0][0].size(); ++k) {
                // Example activation function: ReLU
                output[i][j][k] = std::max(0.0, input[i][j][k]); // Replace with your activation function
            }
        }
    }
    return output;
}

// backward pass
std::vector<std::vector<std::vector<double>>> Backward(const std::vector<std::vector<std::vector<double>>>& input, const std::vector<std::vector<std::vector<double>>>& weights, const std::vector<std::vector<std::vector<double>>>& grad_output) {
    std::vector<std::vector<std::vector<double>>> grad_input(input.size(), std::vector<std::vector<double>>(weights[0].size(), std::vector<double>(weights[0][0].size())));
    
    for (size_t i = 0; i < input.size(); ++i) {
        for (size_t j = 0; j < weights[0].size(); ++j) {
            for (size_t k = 0; k < weights[0][0].size(); ++k) {
                double sum = 0.0;
                for (size_t l = 0; l < input[i].size(); ++l) {
                    sum += grad_output[i][j][k] * weights[l][j][k];
                }
                grad_input[i][j][k] = sum; // Add bias gradient if needed
            }
        }
    }
    
    return grad_input;
}



