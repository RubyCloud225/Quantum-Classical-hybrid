#include "../GaussianNoise.hpp"
#include "../LayerNormalization.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void testGaussianNoiseLayerNormalizationIntegration() {
    int size = 5;
    std::vector<double> mean(size, 0.0);
    std::vector<std::vector<double>> covariance(size, std::vector<double>(size, 0.0));
    std::vector<double> weights(size, 1.0);

    for (int i = 0; i < size; ++i) {
        covariance[i][i] = 1.0;
    }

    GaussianNoise gn(mean, covariance, weights);
    LayerNormalization ln(size, 1e-5);
    ln.resetParameters();

    std::vector<double> noise = gn.generateNoise();
    std::vector<double> normalized = ln.forward(noise);

    double out_mean = 0.0;
    for (double v : normalized) out_mean += v;
    out_mean /= size;

    double variance = 0.0;
    for (double v : normalized) variance += (v - out_mean) * (v - out_mean);
    variance /= size;

    std::cout << "Integration test output mean: " << out_mean << std::endl;
    std::cout << "Integration test output variance: " << variance << std::endl;

    assert(std::abs(out_mean) < 1e-5 && "Mean is not close to 0");
    assert(std::abs(variance - 1.0) < 1e-5 && "Variance is not close to 1");
    std::cout << "Integration test passed!" << std::endl;
}

int main() {
    testGaussianNoiseLayerNormalizationIntegration();
    return 0;
}
