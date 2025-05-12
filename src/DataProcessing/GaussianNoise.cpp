#include "GaussianNoise.hpp"
#include <stdexcept>
#include <cmath>
#include <cuda_runtime.h>

// Constructor
GaussianNoise::GaussianNoise(const std::vector<double>& mean,
                             const std::vector<std::vector<double>>& covariance,
                             const std::vector<double>& weights)
    : mean_(mean), covariance_(covariance), weights_(weights), distribution_(0.0, 1.0) {

    if (covariance.size() != covariance[0].size()) {
        throw std::invalid_argument("Covariance matrix must be square.");
    }
    if (mean.size() != weights.size()) {
        throw std::invalid_argument("Mean and weights must have the same size.");
    }

    choleskyDecomposition();
}

void GaussianNoise::choleskyDecomposition() {
    size_t n = covariance_.size();
    L_.resize(n, std::vector<double>(n, 0.0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = covariance_[i][j];
            for (size_t k = 0; k < j; ++k) {
                sum -= L_[i][k] * L_[j][k];
            }
            if (i == j) {
                L_[i][j] = std::sqrt(sum);
            } else {
                L_[i][j] = sum / L_[j][j];
            }
        }
    }
}

std::vector<double> GaussianNoise::generateNoise() {
    size_t n = mean_.size();
    std::vector<double> z(n);
    for (size_t i = 0; i < n; ++i) {
        z[i] = distribution_(generator_);
    }

    std::vector<double> noise(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            noise[i] += L_[i][j] * z[j];
        }
        noise[i] += mean_[i];
        noise[i] *= weights_[i];
    }
    return noise;
}

double GaussianNoise::calculateDensity(const std::vector<double>& sample) {
    size_t n = mean_.size();
    double determinant = 1.0;
    for (size_t i = 0; i < n; ++i) {
        determinant *= L_[i][i] * L_[i][i];
    }
    double exponent = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double diff = sample[i] - mean_[i];
        exponent -= 0.5 * diff * diff / covariance_[i][i];
    }
    return (1.0 / std::sqrt(std::pow(2 * M_PI, n) * determinant)) * std::exp(exponent);
}

double GaussianNoise::negativeLogLikelihood(const std::vector<double>& sample) {
    double density = calculateDensity(sample);
    if (density <= 0.0) throw std::runtime_error("Non-positive density.");
    return -std::log(density);
}

double GaussianNoise::calculateEntropy() const {
    size_t n = mean_.size();
    double determinant = 1.0;
    for (size_t i = 0; i < n; ++i) {
        determinant *= L_[i][i] * L_[i][i];
    }
    return 0.5 * (n * std::log(2 * M_PI) + std::log(determinant));
}

void GaussianNoise::uploadToDevice() {
    int n = static_cast<int>(mean_.size());
    size_t vec_size = n * sizeof(double);
    size_t mat_size = n * n * sizeof(double);

    std::vector<double> L_flat(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            L_flat[i * n + j] = L_[i][j];

    cudaMalloc(&d_L_, mat_size);
    cudaMalloc(&d_mean_, vec_size);
    cudaMalloc(&d_weights_, vec_size);
    cudaMalloc(&d_noise_, vec_size);

    cudaMemcpy(d_L_, L_flat.data(), mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean_, mean_.data(), vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_, weights_.data(), vec_size, cudaMemcpyHostToDevice);
}

void GaussianNoise::freeDeviceMemory() {
    cudaFree(d_L_);
    cudaFree(d_mean_);
    cudaFree(d_weights_);
    cudaFree(d_noise_);
    d_L_ = d_mean_ = d_weights_ = d_noise_ = nullptr;
}

extern void launchGaussianNoiseKernel(double* d_L, double* d_mean, double* d_weights, double* d_noise, int dim, unsigned long long seed);

void GaussianNoise::runCUDAKernel(std::vector<double>& output) {
    int dim = static_cast<int>(mean_.size());
    uploadToDevice();
    launchGaussianNoiseKernel(d_L_, d_mean_, d_weights_, d_noise_, dim, static_cast<unsigned long long>(time(nullptr)));
    output.resize(dim);
    cudaMemcpy(output.data(), d_noise_, dim * sizeof(double), cudaMemcpyDeviceToHost);
    freeDeviceMemory();
}