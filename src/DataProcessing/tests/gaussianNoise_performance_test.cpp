#include "../GaussianNoise.hpp"
#include <iostream>
#include <vector>
#include <chrono>

void testLargeInputPerformance() {
    int size = 1000;
    std::vector<double> mean(size, 0.0);
    std::vector<std::vector<double>> covariance(size, std::vector<double>(size, 0.0));
    std::vector<double> weights(size, 1.0);

    // Initialize covariance as identity matrix
    for (int i = 0; i < size; ++i) {
        covariance[i][i] = 1.0;
    }

    GaussianNoise gn(mean, covariance, weights);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> noise = gn.generateNoise();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "testLargeInputPerformance: generated noise for size " << size << " in " << diff.count() << " seconds\n";
}

void testMultipleRunsPerformance() {
    int size = 100;
    std::vector<double> mean(size, 0.0);
    std::vector<std::vector<double>> covariance(size, std::vector<double>(size, 0.0));
    std::vector<double> weights(size, 1.0);

    // Initialize covariance as identity matrix
    for (int i = 0; i < size; ++i) {
        covariance[i][i] = 1.0;
    }

    GaussianNoise gn(mean, covariance, weights);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        std::vector<double> noise = gn.generateNoise();
        (void)noise;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "testMultipleRunsPerformance: generated noise 10000 times for size " << size << " in " << diff.count() << " seconds\n";
}

int main() {
    testLargeInputPerformance();
    testMultipleRunsPerformance();
    return 0;
}
