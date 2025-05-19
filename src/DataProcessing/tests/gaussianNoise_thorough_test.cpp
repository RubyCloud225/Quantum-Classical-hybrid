#include "../GaussianNoise.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <chrono>

void testInvalidCovariance() {
    try {
        std::vector<double> mean = {0.0, 0.0};
        std::vector<std::vector<double>> covariance = {{1.0, 0.5}};
        std::vector<double> weights = {1.0, 1.0};
        GaussianNoise gn(mean, covariance, weights);
        std::cout << "testInvalidCovariance failed: no exception thrown\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "testInvalidCovariance passed\n";
    }
}

void testMeanWeightsSizeMismatch() {
    try {
        std::vector<double> mean = {0.0, 0.0};
        std::vector<std::vector<double>> covariance = {{1.0, 0.5}, {0.5, 1.0}};
        std::vector<double> weights = {1.0};
        GaussianNoise gn(mean, covariance, weights);
        std::cout << "testMeanWeightsSizeMismatch failed: no exception thrown\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "testMeanWeightsSizeMismatch passed\n";
    }
}

void testGenerateNoisePerformance() {
    std::vector<double> mean = {0.0, 0.0};
    std::vector<std::vector<double>> covariance = {{1.0, 0.5}, {0.5, 1.0}};
    std::vector<double> weights = {1.0, 1.0};
    GaussianNoise gn(mean, covariance, weights);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; ++i) {
        std::vector<double> noise = gn.generateNoise();
        (void)noise;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "testGenerateNoisePerformance: generated 100000 samples in " << diff.count() << " seconds\n";
}

void testCalculateDensityAndNLL() {
    std::vector<double> mean = {0.0, 0.0};
    std::vector<std::vector<double>> covariance = {{1.0, 0.5}, {0.5, 1.0}};
    std::vector<double> weights = {1.0, 1.0};
    GaussianNoise gn(mean, covariance, weights);

    std::vector<double> sample = gn.generateNoise();
    double density = gn.calculateDensity(sample);
    double nll = gn.negativeLogLikelihood(sample);

    if (density > 0) {
        std::cout << "testCalculateDensityAndNLL passed\n";
    } else {
        std::cout << "testCalculateDensityAndNLL failed: density <= 0\n";
    }

    if (nll >= 0) {
        std::cout << "testCalculateDensityAndNLL NLL passed\n";
    } else {
        std::cout << "testCalculateDensityAndNLL NLL failed: negative NLL\n";
    }
}

int main() {
    testInvalidCovariance();
    testMeanWeightsSizeMismatch();
    testGenerateNoisePerformance();
    testCalculateDensityAndNLL();
    return 0;
}
