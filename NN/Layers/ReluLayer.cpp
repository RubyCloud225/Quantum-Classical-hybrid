#include "ReluLayer.hpp"

ReLu::ReLu() {}

// Method to compute the output of the ReLU layer
std::vector<double> ReLu::forward(const std::vector<double>& input) const {
    std::vector<double> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0, input[i]);
    }
    return output;
}

// Helper function to compute the ReLu activation
double ReLu::relu(double x) {
    return std::max(0.0, x);
}

//Method for backward Propagation
std::vector<double> ReLu::backward(const std::vector<double>& input, const std::vector<double>& upstreamGradient) {
    std::vector<double> downstreamGradient(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        downstreamGradient[i] = reluGradient(input[i]) * upstreamGradient[i];
    }
    return downstreamGradient;
}

double ReLu::reluGradient(double x) {
    return x > 0 ? 1 : 0.0;
}

