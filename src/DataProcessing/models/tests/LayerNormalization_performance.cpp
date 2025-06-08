#include "../LayerNormalization.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>

std::vector<double> generateDataTest(int size) {
    std::vector<double> data(size);
    std::mt19937 gen(42);
    std::normal_distribution<> dist(0.0, 1.0);
    for (double& d : data) d = dist(gen);
    return data;
}

void benchmarkLayerNormalization(size_t size) {
    std::vector<double> query = generateDataTest(size);
    LayerNormalization layerNorm(size, 1e-5);
    layerNorm.resetParameters();
    auto start = std::chrono::high_resolution_clock::now();
    auto output = layerNorm.forward(query);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "LayerNormalization with size " << size << " took " << duration << " microseconds." << std::endl;
}
int main() {
    std::cout << "Running LayerNormalization performance tests..." << std::endl;
    benchmarkLayerNormalization(10);
    benchmarkLayerNormalization(100);
    benchmarkLayerNormalization(1000);
    benchmarkLayerNormalization(10000);
    benchmarkLayerNormalization(100000);
    benchmarkLayerNormalization(1000000);
    return 0;
}