#include "GaussianNoise.hpp"
#include <stdexcept>
#include <cmath>

GaussianNoise::GaussianNoise(const std::vector<double>& mean, const std::vector<std::vector<double>>& coveriance) : mean_(mean), covariance_(coveriance), distribution_(0.0, 1.0) {
        // Check if covariance matrix is square and positive definite
    if (coveriance.size() != coveriance[0].size()) {
        throw std::invalid_argument("Coverance matrix must be square.");
    }

    choleskyDecomposition();
};

void GaussianNoise::choleskyDecomposition() {
    // Perform Cholesky decomposition on the covariance matrix
    size_t n = covariance_.size();
    L_.resize(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            double sum = covariance_[i][j];
            L_.resize(n, std::vector<double>(n, 0.0));
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
    size_t n = mean_.size(); /// t = time? ?
    std::vector<double> z(n);
    // Generate Standard normal random variables
    for (size_t i = 0; i < n; ++i) {
        z[i] = distribution_(generator_);
    }

    // Transform to multivariate Gaussian
    std::vector<double> noise(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j <= i; ++j) {
            noise[i] += L_[i][j] * z[j];
        }
        noise[i] += mean_[i];
    }
    return noise;
}

double GaussianNoise::calculateDensity(const std::vector<double>& sample) {
    // Calculate the density of the multivariate Gaussian distribution
    size_t n = mean_.size();
    double determinant = 1.0;
    for (size_t i = 0; i < n; ++i) {
        determinant *= L_[i][i] * L_[i][i]; // Calculate the determinant from L
    }
    double exponent = 00;
    for (size_t i = 0; i < n; ++i) {
        double diff = sample[i] - mean_[i];
        exponent -= 0.5 * diff * diff / covariance_[i][i]; // Simplify the diagonal covariance
    }
    return (1.0 / std::sqrt(std::pow(2 * M_PI, n) * determinant)) * std::exp(exponent);
}