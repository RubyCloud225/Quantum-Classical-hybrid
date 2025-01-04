#include "LinearRegression.hpp"
#include <numeric>
#include <stdexcept>

LinearRegression::LinearRegression() : slope(0), intercept(0) {}

void LinearRegression::fit(const std::vector<std::pair<double, double>>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }

    double n = data.size();
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    for (const auto& point : data) {
        sum_x += point.first;
        sum_y += point.second;
        sum_xy += point.first * point.second;
        sum_x2 += point.first * point.first;
    }
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    intercept = (sum_y - slope * sum_x) / n;
}

double LinearRegression::predict(double x) const {
    return slope * x + intercept;
}

void LinearRegression::reshapeData(const std::vector<double>& x, const std::vector<double>& y, std::vector<std::pair<double, double>>& reshapedData) const {
    if (x.size() != y.size()) {
        throw std::invalid_argument("X and Y vectors must be of the same length");
    }
    reshapedData.clear();
    for (size_t i = 0; i < x.size(); ++i) {
        reshapedData.emplace_back(x[i], y[i]);
    }
}