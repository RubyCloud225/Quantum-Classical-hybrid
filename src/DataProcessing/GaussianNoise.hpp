#ifndef GAUSSIAN_NOISE_HPP
#define GAUSSIAN_NOISE_HPP

#include <vector>
#include <random>

class GaussianNoise {
public:
    GaussianNoise(const std::vector<double>& mean,
                  const std::vector<std::vector<double>>& covariance,
                  const std::vector<double>& weights);

    std::vector<double> generateNoise();     // CPU version
    

    double calculateDensity(const std::vector<double>& sample);
    double negativeLogLikelihood(const std::vector<double>& sample);
    double calculateEntropy() const;

private:
    void choleskyDecomposition();
    

    std::vector<double> mean_;
    std::vector<std::vector<double>> covariance_;
    std::vector<double> weights_;
    std::vector<std::vector<double>> L_;

    

    std::mt19937 generator_{std::random_device{}()};
    std::normal_distribution<double> distribution_;
};

#endif