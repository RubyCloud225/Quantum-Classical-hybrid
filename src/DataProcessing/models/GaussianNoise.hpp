#ifndef GAUSSIAN_NOISE_HPP
#define GAUSSIAN_NOISE_HPP

#include <vector>
#include <random>

#ifndef __APPLE__
#include <cuda_runtime.h>
#endif

class GaussianNoise {
public:
    GaussianNoise(const std::vector<double>& mean,
                  const std::vector<std::vector<double>>& covariance,
                  const std::vector<double>& weights);

    std::vector<double> generateNoise();     // CPU version
    

    double calculateDensity(const std::vector<double>& sample);
    double negativeLogLikelihood(const std::vector<double>& sample);
    double calculateEntropy() const;

#ifndef __APPLE__
    void uploadToDevice();
    void freeDeviceMemory();
    void runCUDAKernel(std::vector<double>& output);
#endif

private:
    void choleskyDecomposition();
    

    std::vector<double> mean_;
    std::vector<std::vector<double>> covariance_;
    std::vector<double> weights_;
    std::vector<std::vector<double>> L_;

#ifndef __APPLE__
    double* d_L_ = nullptr;
    double* d_mean_ = nullptr;
    double* d_weights_ = nullptr;
    double* d_noise_ = nullptr;
#endif

    std::mt19937 generator_{std::random_device{}()};
    std::normal_distribution<double> distribution_;
};

#endif
