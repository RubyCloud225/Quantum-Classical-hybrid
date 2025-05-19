#include "Diffusion_model.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <cassert>

DiffusionModel::DiffusionModel(int input_size, int output_size) : input_size(input_size), output_size(output_size), normal_dist(0.0, 1.0) {
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("Input and output sizes must be positive integers.");
    }
}

void DiffusionModel::compute_mean_variance(const std::vector<double>& x_t, int t, std::vector<double>& mean, std::vector<double>& variance) {
    if (x_t.size() != input_size) {
        throw std::invalid_argument("Input size does not match the model's input size.");
    }
    if (t < 0 || t >= 1000) { // Assuming a fixed number of timesteps
        throw std::out_of_range("Time step t is out of range.");
    }
    mean.resize(x_t.size());
    variance.resize(x_t.size());
    // Example mean and variance computation
    #pragma omp parallel for (int i = 0; i < output_size; ++i) {
        mean[i] = x_t[i % input_size] * (1.0 - t / 1000.0); // Placeholder computation
        variance[i] = 1.0 - t / 1000.0; // Placeholder computation
    }
}

std::vector<double> DiffusionModel::sample(
    const std::vector<double>& x, 
    int t,
    bool clip_denoised,
    const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn,
    const std::function<std::vector<double>(const std::vector<double>&)>& cond_fn,
    const std::unordered_map<std::string, double>& model_kwags
) {
    std::vector<double> mean, variance;
    compute_mean_variance(x, t, mean, variance);
    // Apply Conditioning Function
    if (cond_fn) {
        mean = cond_fn(mean);
    }
    // predict x_start using denoised function
    std::vector<double> x_start = (denoised_fn) ? denoised_fn(mean) : mean;

    // Clip denoised values if required
    #pragma omp parallel for (size_t i = 0; i < x_start.size(); ++i) {
        x_start[i] = std::clamp(x_start[i], -1.0, 1.0);
    }
    // Sample from the normal distribution
    std::vector<double> sample(x.size());
    #pragma omp parallel for (size_t i = 0; i < x.size(); ++i) {
        sample[i] = x_start[i] + std::sqrt(variance[i]) * normal_dist(generator);
    }

    return sample;
}
