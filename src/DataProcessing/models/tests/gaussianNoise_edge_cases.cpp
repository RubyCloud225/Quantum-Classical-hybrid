#include "../GaussianNoise.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

void testZeroVariance() {
    std::vector<double> mean = {0.0, 0.0};
    std::vector<std::vector<double>> covariance = {{0.0, 0.0}, {0.0, 0.0}};
    std::vector<double> weights = {1.0, 1.0};

    try {
        GaussianNoise gn(mean, covariance, weights);
        std::vector<double> noise = gn.generateNoise();
        std::cout << "Generated noise with zero variance: ";
        for (double n : noise) {
            std::cout << n << " ";
        }
        std::cout << std::endl;
        std::cout << "Zero variance test passed." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Zero variance test failed: " << e.what() << std::endl;
        assert(false);
    }
}

int main() {
    testZeroVariance();
    return 0;
}
