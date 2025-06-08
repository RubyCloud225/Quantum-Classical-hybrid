#include <iostream>
#include <vector>
#include <numeric>
#include "../LayerNormalization.hpp"
#include <cmath>
#include <cassert>

double mean(const std::vector<double>& input) {
    double sum = std::accumulate(input.begin(), input.end(), 0.0);
    return sum / input.size();
}

double variance(const std::vector<double>& input, double mean_val) {
    double sum = 0.0;
    for (double v : input) {
        sum += (v - mean_val) * (v - mean_val);
    }
    return sum / input.size();
}

int main() {
    // Create a LayerNormalization object
    LayerNormalization layerNorm(5, 1e-5);

    // Reset parameters
    layerNorm.resetParameters();

    // Create an input vector
    std::vector<double> input = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Perform forward pass
    std::vector<double> output = layerNorm.forward(input);
    double out_mean = mean(output);
    double out_variance = variance(output, out_mean);

    std::cout << "Output Mean: " << out_mean << std::endl;
    std::cout << "Output Variance: " << out_variance << std::endl;

    // Check with a tolerance for floating point errors
    assert(std::abs(out_mean - 0.0) < 1e-6 && "Mean is not close to 0");
    assert(std::abs(out_variance - 1.0) < 1e-5 && "Variance is not close to 1");

    std::cout << "LayerNormalization test passed!" << std::endl;

    return 0;
}