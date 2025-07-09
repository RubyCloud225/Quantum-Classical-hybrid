#include "../Diffusion_model.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <numeric>
#include <chrono>
#include <functional>

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

void run_edge_case_tests(DiffusionModel& model, 
    const std::vector<double>& input, 
    int timestep, 
    const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn, 
    const std::function<std::vector<double>(const std::vector<double>&)>& cond_fn, 
    const std::unordered_map<std::string, double>& model_kwags) {
    // Test with invalid input size
    try {
        std::vector<double> invalid_input(input.size() + 1, 0.5);
        model.sample(invalid_input, timestep, true, denoised_fn, cond_fn, model_kwags);
        assert(false && "Expected Invalid Argument for input size"); // Should not reach here
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }

    // Test with invalid timestep
    try {
        model.sample(input, 2000, true, denoised_fn, cond_fn, model_kwags);
        assert(false && "Expected out_of_range for timestep"); // Should not reach here
    } catch (const std::out_of_range& e) {
        std::cout << "Caught expected exception: " << e.what() << std::endl;
    }
}

//Performance test
void run_performance_test(DiffusionModel& model, 
    const std::vector<double>& input,
    int timestep,
    const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn,
    const std::function<std::vector<double>(const std::vector<double>&)>& cond_fn,
    const std::unordered_map<std::string, double>& model_kwags) {
    const int trials = 1000;
    double total_mean = 0.0;
    double total_variance = 0.0;
    double total_time = 0.0;

    for (int i = 0; i < trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> output = model.sample(input, timestep, true, denoised_fn, cond_fn, model_kwags);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        double m = compute_mean(output);
        double v = compute_variance(output, m);
        total_mean += m;
        total_variance += v;
        total_time += elapsed.count();
    }
    std::cout << "Performance Test Results:" << trials << " trials" << std::endl;
    std::cout << "Average Mean: " << total_mean / trials << std::endl;
    std::cout << "Average Variance: " << total_variance / trials << std::endl;
    std::cout << "Average Time: " << total_time / trials << " ms" << std::endl;
}

void run_varied_input_tests();
void run_timestep_kwags_tests();
void run_output_validation_tests();
void run_stress_test();

void run_thorough_test();

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
    // Edge case tests
    run_edge_case_tests(model, input, timestep, dummy_denoised, dummy_cond, model_kwags);
    // Performance test
    run_performance_test(model, input, timestep, dummy_denoised, dummy_cond, model_kwags);

    // Additional thorough tests
    run_varied_input_tests();
    run_timestep_kwags_tests();
    run_output_validation_tests();
    run_stress_test();
    run_thorough_test();
}

// Additional test functions

void run_varied_input_tests() {
    std::cout << "Running varied input size and value tests..." << std::endl;
    std::vector<int> input_sizes = {5, 10, 20};
    for (int size : input_sizes) {
        DiffusionModel model(size, size);
        std::vector<double> input(size, 0.3);
        int timestep = 100;
        auto dummy_denoised = [] (const std::vector<double>& x) { return x; };
        auto dummy_cond = [] (const std::vector<double>& x) { return x; };
        std::unordered_map<std::string, double> model_kwags;
        std::vector<double> output = model.sample(input, timestep, true, dummy_denoised, dummy_cond, model_kwags);
        double mean = compute_mean(output);
        double variance = compute_variance(output, mean);
        std::cout << "Input size: " << size << ", Mean: " << mean << ", Variance: " << variance << std::endl;
    }
}

void run_timestep_kwags_tests() {
    std::cout << "Running timestep and model_kwags variation tests..." << std::endl;
    int input_size = 10;
    DiffusionModel model(input_size, input_size);
    std::vector<double> input(input_size, 0.5);
    auto dummy_denoised = [] (const std::vector<double>& x) { return x; };
    auto dummy_cond = [] (const std::vector<double>& x) { return x; };
    std::vector<int> timesteps = {0, 250, 999};
    for (int timestep : timesteps) {
        std::unordered_map<std::string, double> model_kwags = {{"param", timestep * 0.01}};
        std::vector<double> output = model.sample(input, timestep, true, dummy_denoised, dummy_cond, model_kwags);
        double mean = compute_mean(output);
        double variance = compute_variance(output, mean);
        std::cout << "Timestep: " << timestep << ", Mean: " << mean << ", Variance: " << variance << std::endl;
    }
}

void run_output_validation_tests() {
    std::cout << "Running output validation tests..." << std::endl;
    int input_size = 10;
    DiffusionModel model(input_size, input_size);
    std::vector<double> input(input_size, 0.5);
    int timestep = 500;
    auto dummy_denoised = [] (const std::vector<double>& x) { return x; };
    auto dummy_cond = [] (const std::vector<double>& x) { return x; };
    std::unordered_map<std::string, double> model_kwags;
    std::vector<double> output = model.sample(input, timestep, true, dummy_denoised, dummy_cond, model_kwags);
    for (double val : output) {
        if (val < 0.0 || val > 1.0) {
            std::cerr << "Output value out of expected range: " << val << std::endl;
        }
    }
    std::cout << "Output validation completed." << std::endl;
}

void run_stress_test() {
    std::cout << "Running stress test with large input size..." << std::endl;
    int input_size = 1000;
    DiffusionModel model(input_size, input_size);
    std::vector<double> input(input_size, 0.5);
    int timestep = 500;
    auto dummy_denoised = [] (const std::vector<double>& x) { return x; };
    auto dummy_cond = [] (const std::vector<double>& x) { return x; };
    std::unordered_map<std::string, double> model_kwags;
    const int trials = 100;
    double total_time = 0.0;
    for (int i = 0; i < trials; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> output = model.sample(input, timestep, true, dummy_denoised, dummy_cond, model_kwags);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time += elapsed.count();
    }
    std::cout << "Stress Test: Average Time for " << trials << " trials: " << total_time / trials << " ms" << std::endl;
}

void run_thorough_test() {
    std::cout << "Running thorough test with varied inputs and validations..." << std::endl;
    std::vector<int> input_sizes = {1, 5, 10, 50, 100};
    std::vector<int> timesteps = {0, 100, 250, 500, 750, 999};
    auto dummy_denoised = [] (const std::vector<double>& x) { return x; };
    auto dummy_cond = [] (const std::vector<double>& x) { return x; };

    for (int size : input_sizes) {
        DiffusionModel model(size, size);
        std::vector<double> input(size, 0.5);
        for (int timestep : timesteps) {
            std::unordered_map<std::string, double> model_kwags = {{"param", timestep * 0.01}};
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<double> output = model.sample(input, timestep, true, dummy_denoised, dummy_cond, model_kwags);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;

            // Validate output range
            bool valid = true;
            for (double val : output) {
                if (val < -1.0 || val > 1.0) {
                    std::cerr << "Output value out of expected range: " << val << std::endl;
                    valid = false;
                }
            }

            std::cout << "Input size: " << size << ", Timestep: " << timestep
                      << ", Time: " << elapsed.count() << " ms"
                      << ", Output valid: " << (valid ? "Yes" : "No") << std::endl;
        }
    }
}

