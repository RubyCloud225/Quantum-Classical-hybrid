#include "LinearRegression.hpp"
#include <stdexcept>
#include <cmath>
#include "utils/logger.hpp"
#include <math.h>
#include <omp.h>

LinearRegression::LinearRegression() : slope(0.0), intercept(0.0) {
    Logger::log("Initialized LinearRegression model", LogLevel::INFO, __FILE__, __LINE__);
}

void LinearRegression::fit(const std::vector<std::pair<double, double>>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
    size_t n = data.size();

    #pragma omp parallel for reduction(+:sum_x, sum_y, sum_xy, sum_xx)
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
    // OpenMP parallel reshape
    size_t n = x.size();
    std::vector<std::pair<double, double>> tempData;
    tempData.reserve(n);
    #pragma omp parallel
    {
        std::vector<std::pair<double, double>> localData;
        #pragma omp for nowait
        for (size_t i = 0; i < n; ++i) {
            if (isnan(x[i]) || isnan(y[i])) {
                #pragma omp critical
                Logger::log("NaN value found in input data at index " + std::to_string(i), LogLevel::WARNING, __FILE__, __LINE__);
                continue;
            }
            localData.emplace_back(x[i], y[i]);
        }
        #pragma omp critical
        tempData.insert(tempData.end(), localData.begin(), localData.end());
    }
    reshapedData = std::move(tempData);
}
