#include "LayerNormalization.hpp"
#include <numeric>
#include <cmath>
#include <stdexcept>

LayerNormalization::LayerNormalization(int features, double epsilon) : normal_shape(features), epsilon(epsilon), gamma(features, 1.0), beta(features, 0.0) {}

void LayerNormalization::resetParameters() {
    std::fill(gamma.begin(), gamma.end(), 1.0);
    std::fill(beta.begin(), beta.end(), 0.0);
}

std::vector<double> LayerNormalization::forward(const std::vector<double>& input) {
    if (input.size() != normal_shape) {
        throw std::invalid_argument("input size must be Normal_shape");
    }

    // Calculate mean
    double mean = std::accumulate(input.begin(), input.end(), 0.0) / normal_shape;

    // calculate the variance 
    double variance = 0.0;
    for (const auto& value : input) {
        variance += (value - mean) * (value - mean);
    }
    variance /= normal_shape;

    // Calculate the standard deviation
    double std_dev = std::sqrt(variance + epsilon);

    // Normalize
    std::vector<double> normalized(normal_shape);
    for (int i = 0; i < normal_shape; ++i) {
        normalized[i] = (input[i] - mean) / std_dev;
    }

    // Scale and shift
    for (int i = 0; i < normal_shape; ++ i) {
        normalized[i] = normalized[i] * gamma[i] + beta[i];
    }

    return normalized;
}