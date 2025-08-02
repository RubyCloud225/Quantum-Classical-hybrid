#include "GaussianNoise.hpp"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <ctime>
#include "utils/logger.hpp"


GaussianNoise::GaussianNoise(const std::vector<double>& mean, const std::vector<std::vector<double>>& covariance, const std::vector<double>& weights)
    : mean_(mean), covariance_(covariance), weights_(weights), distribution_(0.0, 1.0) {

    Logger::log("Initializing GaussianNoise with mean size: " + std::to_string(mean.size()), LogLevel::INFO, __FILE__, __LINE__);
    if (covariance.size() != covariance[0].size()) {
        Logger::log("Covariance matrix is not square", LogLevel::ERROR, __FILE__, __LINE__);
        throw std::invalid_argument("Covariance matrix must be square.");
    }
    if (mean.size() != weights.size()) {
        Logger::log("Mean and weights size mismatch", LogLevel::ERROR, __FILE__, __LINE__);
        throw std::invalid_argument("Mean and weights must have the same size.");
    }
    Logger::log("Cholesky decomposition starting", LogLevel::INFO, __FILE__, __LINE__);

    choleskyDecomposition();

    Logger::log("Cholesky decomposition completed", LogLevel::INFO, __FILE__, __LINE__);
}

void GaussianNoise::choleskyDecomposition() {
    size_t n = covariance_.size();
    L_.resize(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += L_[i][k] * L_[j][k];
            }
            if (i == j) {
                L_[i][j] = std::sqrt(covariance_[i][i] - sum);
            } else {
                L_[i][j] = (1.0 / L_[j][j]) * (covariance_[i][j] - sum);
            }
        }
    }
}

std::vector<double> GaussianNoise::generateNoise() {
    size_t n = mean_.size();
    std::vector<double> noise(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        noise[i] = distribution_(generator_);
    }

    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j <= i; ++j) {
            sum += L_[i][j] * noise[j];
        }
        noise[i] = sum;
    }

    for (size_t i = 0; i < n; ++i) {
        noise[i] += mean_[i];
        noise[i] *= weights_[i];
    }
    Logger::log("Generated Gaussian noise vector", LogLevel::INFO, __FILE__, __LINE__);
    return noise;
}

double GaussianNoise::calculateDensity(const std::vector<double>& sample) {
    // Implementation omitted for brevity
    return 0.0;
}

double GaussianNoise::negativeLogLikelihood(const std::vector<double>& sample) {
    double density = calculateDensity(sample);
    if (density <= 0.0) {
        Logger::log("Density calculation returned non-positive value", LogLevel::ERROR, __FILE__, __LINE__);
        throw std::runtime_error("Non-positive density.");
    }
    return -std::log(density);
}

#ifndef __APPLE__
void GaussianNoise::uploadToDevice() {
    Logger::log("Uploading GaussianNoise data to device", LogLevel::INFO, __FILE__, __LINE__);
    int n = static_cast<int>(mean_.size());
    size_t vec_size = n * sizeof(double);
    size_t mat_size = n * n * sizeof(double);

    // CUDA memory allocation and copying omitted for brevity

    Logger::log("Freed device memory for GaussianNoise", LogLevel::INFO, __FILE__, __LINE__);
}

extern void launchGaussianNoiseKernel(double* d_L, double* d_mean, double* d_weights, double* d_noise, int dim, unsigned long long seed);

void GaussianNoise::runKernel() {
    uploadToDevice();
    launchGaussianNoiseKernel(d_L_, d_mean_, d_weights_, d_noise_, dim, static_cast<unsigned long long>(time(nullptr)));
    output.resize(dim);
    cudaMemcpy(output.data(), d_noise_, dim * sizeof(double), cudaMemcpyDeviceToHost);
    Logger::log("CUDA kernel execution complete", LogLevel::INFO, __FILE__, __LINE__);
    freeDeviceMemory();
}
#endif
