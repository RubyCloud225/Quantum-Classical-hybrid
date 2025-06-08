#include "../LinearRegression.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>

void testEmptyData() {
    try {
        LinearRegression lr;
        std::vector<double> x;
        std::vector<double> y;
        std::vector<std::pair<double, double>> data;
        lr.reshapeData(x, y, data);
        lr.fit(data);
        std::cerr << "testEmptyData failed: exception not thrown\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "testEmptyData passed\n";
    }
}

void testMismatchedData() {
    try {
        LinearRegression lr;
        std::vector<double> x = {1, 2};
        std::vector<double> y = {1};
        std::vector<std::pair<double, double>> data;
        lr.reshapeData(x, y, data);
        lr.fit(data);
        std::cerr << "testMismatchedData failed: exception not thrown\n";
    } catch (const std::invalid_argument& e) {
        std::cout << "testMismatchedData passed\n";
    }
}

int main() {
    testEmptyData();
    testMismatchedData();
    std::cout << "LinearRegression edge case tests completed.\n";
    return 0;
}
