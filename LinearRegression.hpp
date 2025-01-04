#ifndef LINEARREGRESSION_HPP
#define LINEARREGRESSION_HPP

#include <vector>
#include <utility>

class LinearRegression {
    public:
    LinearRegression();
    void fit(const std::vector<std::pair<double, double>>& data);
    double predict(double x) const;
    void reshapeData(const std::vector<double> & x, const std::vector<double>& y, std::vector<std::pair<double, double>>& reshapedData) const;
    private:
    double slope;
    double intercept;
};

#endif // LINEARREGRESSION_HPP