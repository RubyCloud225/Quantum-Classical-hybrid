#include "../LinearRegression.hpp"
#include <iostream>
#include <vector>
#include <cassert>

void testBasicLinearRegression() {
    LinearRegression lr;
    std::vector<double> x = {1, 2, 3};
    std::vector<double> y = {3, 5, 7};
    std::vector<std::pair<double, double>> data;
    lr.reshapeData(x, y, data);
    lr.fit(data);
    for (size_t i = 0; i < x.size(); ++i) {
        double pred = lr.predict(x[i]);
        std::cout << "Predicted: " << pred << ", Actual: " << y[i] << std::endl;
        assert(std::abs(pred - y[i]) < 1e-5);
    }
    std::cout << "Basic LinearRegression test passed." << std::endl;
}

int main() {
    testBasicLinearRegression();
    return 0;
}
