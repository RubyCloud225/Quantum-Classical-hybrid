#include "../LayerNormalization.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void testEmptyInput() {
    try {
        LayerNormalization layerNorm(0, 1e-5);
        std::vector<double> input;
        auto output = layerNorm.forward(input);
        std::cerr << "testEmptyInput failed: exception not thrown\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "testEmptyInput passed\n";
    }
}

void testMismatchedInputSize() {
    try {
        LayerNormalization layerNorm(5, 1e-5);
        std::vector<double> input = {1.0, 2.0};
        auto output = layerNorm.forward(input);
        std::cerr << "testMismatchedInputSize failed: exception not thrown\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "testMismatchedInputSize passed\n";
    }
}

void testZeroVariance() {
    try {
        LayerNormalization layerNorm(3, 1e-5);
        std::vector<double> input = {1.0, 1.0, 1.0};
        auto output = layerNorm.forward(input);
        std::cerr << "testZeroVariance failed: exception not thrown\n";
    } catch (const std::runtime_error& e) {
        std::cout << "testZeroVariance passed\n";
    }
}

int main() {
    testEmptyInput();
    testMismatchedInputSize();
    testZeroVariance();
    std::cout << "LayerNormalization edge case tests completed.\n";
    return 0;
}
