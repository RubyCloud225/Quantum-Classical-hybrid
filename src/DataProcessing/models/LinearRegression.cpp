#include "LinearRegression.hpp"
#include <stdexcept>
#include <cmath>
#include "utils/logger.hpp"
#include <math.h>

LinearRegression::LinearRegression() : slope(0.0), intercept(0.0) {
    Logger::log("Initialized LinearRegression model", LogLevel::INFO, __FILE__, __LINE__);
}

void LinearRegression::fit(const std::vector<std::pair<double, double>>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    size_t n = data.size();

    for (const auto& pair : data) {
        sum_x += pair.first;
        sum_y += pair.second;
        sum_xy += pair.first * pair.second;
        sum_xx += pair.first * pair.first;
    }

    double denominator = n * sum_xx - sum_x * sum_x;
    if (denominator == 0) {
        throw std::runtime_error("Denominator in slope calculation is zero");
    }

    slope = (n * sum_xy - sum_x * sum_y) / denominator;
    intercept = (sum_y - slope * sum_x) / n;

    Logger::log("Fitted LinearRegression model with slope: " + std::to_string(slope) + " intercept: " + std::to_string(intercept), LogLevel::INFO, __FILE__, __LINE__);
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
        if (isnan(x[i]) || isnan(y[i])) {
            Logger::log("NaN value found in input data at index " + std::to_string(i), LogLevel::WARNING, __FILE__, __LINE__);
            continue; // Skip NaN values
        }
        reshapedData.emplace_back(x[i], y[i]);
    }
}
