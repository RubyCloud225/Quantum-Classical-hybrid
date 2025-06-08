#include "../LinearRegression.hpp"
#include "../GaussianNoise.hpp"
#include "../LayerNormalization.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

void testIntegration() {
    int size = 3;
    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {6, 15, 24};

    LinearRegression lr;
    std::vector<std::pair<double, double>> data;
    lr.reshapeData(x, y, data);
    lr.fit(data);

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

    for (double val : normalized) {
        double pred = lr.predict(val);
        std::cout << "Integration test prediction: " << pred << std::endl;
        assert(std::abs(pred) > 0); // Basic check that prediction returns a value
    }

    std::cout << "Integration test passed!" << std::endl;
}

int main() {
    testIntegration();
    return 0;
}
