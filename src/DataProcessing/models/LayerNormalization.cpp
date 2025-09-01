#include "LayerNormalization.hpp"
#include "utils/logger.hpp"
#include <numeric>
#include <cmath>
#include <stdexcept>

LayerNormalization::LayerNormalization(
    int features, 
    double epsilon) : 
    normal_shape(features), 
    epsilon(epsilon), 
    gamma(features, 1.0), 
    beta(features, 0.0) {
    Logger::log("Initializing LayerNormalization with " + std::to_string(features) + " features", LogLevel::INFO, "LayerNormalization");
}

void LayerNormalization::resetParameters() {
    std::fill(gamma.begin(), gamma.end(), 1.0);
    Logger::log("Reset gamma parameters to 1.0", LogLevel::INFO, "LayerNormalization");
    std::fill(beta.begin(), beta.end(), 0.0);
    Logger::log("Reset beta parameters to 0.0", LogLevel::INFO, "LayerNormalization");
}

std::vector<double> LayerNormalization::forward(const std::vector<double>& input) {
    Logger::log("Running LayerNormalization forward pass", LogLevel::INFO, "LayerNormalization");
    if (input.size() != normal_shape) {
        Logger::log("Input size mismatch in LayerNormalization", LogLevel::ERROR, "LayerNormalization");
        throw std::invalid_argument("input size must be Normal_shape");
    }

    // Parallelized mean calculation using OpenMP reduction
    double mean = 0.0;
    #pragma omp parallel for reduction(+:mean)
    for (int i = 0; i < normal_shape; ++i) {
        mean += input[i];
    }
    mean /= normal_shape;

    // Parallelized variance calculation using OpenMP reduction
    double M2 = 0.0;
    #pragma omp parallel for reduction(+:M2)
    for (int i = 0; i < normal_shape; ++i) {
        double delta = input[i] - mean;
        M2 += delta * delta;
    }
    double variance = M2 / normal_shape;

    // Calculate the standard deviation
    double std_dev = std::sqrt(variance + epsilon);
    if (std_dev == 0) {
        Logger::log("Standard deviation is zero during normalization", LogLevel::ERROR, "LayerNormalization");
        throw std::runtime_error("Standard deviation is zero, cannot normalize.");
    }
    std::vector<double> normalized(normal_shape);
    // Parallelize normalization
    #pragma omp parallel for
    for (int i = 0; i < normal_shape; ++i) {
        normalized[i] = gamma[i] * ((input[i] - mean) / std_dev) + beta[i];
    }
    Logger::log("LayerNormalization forward pass completed", LogLevel::INFO, "LayerNormalization");
    return normalized;
}
