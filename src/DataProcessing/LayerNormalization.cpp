
#include "LayerNormalization.hpp"
#ifndef __APPLE__
#include "LayerNormalizationKernels.cuh"
#endif
#include <numeric>
#include <cmath>
#include <stdexcept>

LayerNormalization::LayerNormalization(
    int features, 
    double epsilon) : 
    normal_shape(features), 
    epsilon(epsilon), 
    gamma(features, 1.0), 
    beta(features, 0.0) {}

void LayerNormalization::resetParameters() {
    std::fill(gamma.begin(), gamma.end(), 1.0);
    std::fill(beta.begin(), beta.end(), 0.0);
}

std::vector<double> LayerNormalization::forward(const std::vector<double>& input) {
    if (input.size() != normal_shape) {
        throw std::invalid_argument("input size must be Normal_shape");
    }

    // calculate mean and variance using Welford's online algorithm for numerical stability
    double mean = 0.0;
    double M2 = 0.0;
    int n = 0;
    for (const auto& value : input) {
        n++;
        double delta = value - mean;
        mean += delta / n;
        double delta2 = value - mean;
        M2 += delta * delta2;
    }
    double variance = M2 / normal_shape;

    // Calculate the standard deviation
    double std_dev = std::sqrt(variance + epsilon);
    if (std_dev == 0) {
        throw std::runtime_error("Standard deviation is zero, cannot normalize.");
    }
    std::vector<double> normalized(normal_shape);
    for (int i = 0; i < normal_shape; ++i) {
        normalized[i] = gamma[i] * ((input[i] - mean) / std_dev) + beta[i];
    }
    return normalized;
}
