#include "../Diffusion_model.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <numeric>

double compute_mean(const std::vector<double>& data) {
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double compute_variance(const std::vector<double>& data , double mean) {
    double sum = 0.0;
    for (double val : data) {
        sum += (val - mean) * (val - mean);
    }
    return sum / data.size();
}

int main() {
    const int input_size = 10;
    const int output_size = 10;
    DiffusionModel model(input_size, output_size);
    std::vector<double> input(input_size, 0.5);
    int timestep = 500;
    auto dummy_denoised = [] (const std::vector<double>& x) {
        return x;
    };
    auto dummy_cond = [] (const std::vector<double>& x) {
        return x;
    };
    std::unordered_map<std::string, double> model_kwags;
    std::vector<double> output = model.sample(input, timestep, true, dummy_denoised, dummy_cond, model_kwags);
    //Stats test
    double mean = compute_mean(output);
    double variance = compute_variance(output, mean);
    std::cout << "Mean: " << mean << ", Variance: " << variance << std::endl;
    //Edge case test- Wrong input size
    try {
        std::vector<double> invalid_input(input_size + 1, 0.5);
        model.sample(invalid_input, timestep, true, dummy_denoised, dummy_cond, model_kwags);
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }
    //Invalid timestep test
    try {
        model.sample(input, 2000, true, dummy_denoised, dummy_cond, model_kwags);
        assert(false && "Expected out_of_range for timestep"); // Should not reach here
    } catch (const std::out_of_range& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }
}

