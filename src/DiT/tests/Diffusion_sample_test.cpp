#include "../Diffusion_Sample.hpp"
#include "../Diffusion_model.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include <unordered_map>
#include <numeric>
#include <chrono>
#include <functional>
#include <map>
#include <algorithm>

void test_performance();
void test_more();

int main() {
    // Setup
    int batch = 1, channels = 3, height = 32, width = 32;
    std::vector<int> shape = {batch, channels, height, width};
    bool clip_denoised = true;
    std::unordered_map<std::string, double> model_kwags;
    std::string device = "cpu";

    int total_size = batch * channels * height * width;

    // Dummy denoised function (identity)
    auto denoised_fn = [](const std::vector<double>& x) {
        return x;
    };

    // Dummy model and sample class
    DiffusionModel dummy_model(total_size, total_size);
    std::vector<double> dummy_schedule(1000, 0.1);
    DiffusionModel model_ref = dummy_model;
    DiffusionSample sampler(model_ref, dummy_schedule);

    // Call progressive sampling
    auto progressive_samples = sampler.p_sample_loop_progressive(
        shape, clip_denoised, denoised_fn, model_kwags, device
    );

    // Check output
    int step = 0;
    for (const auto& sample : progressive_samples) {
        std::cout << "Step: " << step << ", Sample size: " << sample.at("sample").size() << std::endl;
        assert(sample.at("sample").size() == channels * height * width);
        ++step;
    }

    std::cout << "Basic test for p_sample_loop_progressive passed.\n";
    // Edge case: empty shape
    try {
        std::vector<int> empty_shape;
        sampler.p_sample_loop_progressive(
            empty_shape, clip_denoised, denoised_fn, model_kwags, device
        );
        assert(false && "Expected an exception for empty shape");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for empty shape: " << e.what() << std::endl;
    }
    // edge case: shape with zero
    try {
        std::vector<int> zero_shape = {0, 0, 0, 0};
        sampler.p_sample_loop_progressive(
            zero_shape, clip_denoised, denoised_fn, model_kwags, device
        );
        assert(false && "Expected an exception for zero shape");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for zero shape: " << e.what() << std::endl;
    }
    // edge case: shape with negative
    try {
        std::vector<int> negative_shape = {-1, -1, -1, -1};
        sampler.p_sample_loop_progressive(
            negative_shape, clip_denoised, denoised_fn, model_kwags, device
        );
        assert(false && "Expected an exception for negative shape");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for negative shape: " << e.what() << std::endl;
    }
    // edge case: shape with non-integer
    try {
        std::vector<int> non_integer_shape = {1, 3, 32, 32};
        sampler.p_sample_loop_progressive(
            non_integer_shape, clip_denoised, denoised_fn, model_kwags, device
        );
        assert(false && "Expected an exception for non-integer shape");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for non-integer shape: " << e.what() << std::endl;
    }
    // edge case: shape with non-numeric
    try {
        std::vector<int> non_numeric_shape = {1, 3, 32, 32};
        sampler.p_sample_loop_progressive(
            non_numeric_shape, clip_denoised, denoised_fn, model_kwags, device
        );
        assert(false && "Expected an exception for non-numeric shape");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for non-numeric shape: " << e.what() << std::endl;
    }
    // edge case: excessive dimensions
    try {
        std::vector<int> excessive_shape = {1, 3, 32, 32, 32};
        sampler.p_sample_loop_progressive(
            excessive_shape, clip_denoised, denoised_fn, model_kwags, device
        );
        assert(false && "Expected an exception for excessive dimensions");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for excessive dimensions: " << e.what() << std::endl;
    }
    // edge case: Nullptr denoised function (default or fail gracefully)
    try {
        std::function<std::vector<double>(const std::vector<double>&)> null_denoised_fn = nullptr;
        sampler.p_sample_loop_progressive(
            shape, clip_denoised, nullptr, model_kwags, device
        );
        assert(false && "Expected an exception for null denoised function");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for null denoised function: " << e.what() << std::endl;
    }
    // edge case: empty denoised function
    try {
        std::function<std::vector<double>(const std::vector<double>&)> empty_denoised_fn = [](const std::vector<double>& x) { return std::vector<double>(); };
        sampler.p_sample_loop_progressive(
            shape, clip_denoised, empty_denoised_fn, model_kwags, device
        );
        assert(false && "Expected an exception for empty denoised function");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for empty denoised function: " << e.what() << std::endl;
    }
    // edge case: empty model_kwags
    try {
        std::unordered_map<std::string, double> empty_model_kwags;
        sampler.p_sample_loop_progressive(
            shape, clip_denoised, denoised_fn, empty_model_kwags, device
        );
        assert(false && "Expected an exception for empty model_kwags");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for empty model_kwags: " << e.what() << std::endl;
    }
    // edge case: invalid device
    try {
        std::string invalid_device = "invalid_device";
        sampler.p_sample_loop_progressive(
            shape, clip_denoised, denoised_fn, model_kwags, invalid_device
        );
        assert(false && "Expected an exception for invalid device");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for invalid device: " << e.what() << std::endl;
    }
    // performance testing
    test_performance();
    // more tests
    test_more();
    return 0;
}

void test_performance() {
    // Create a sampler with a large shape
    int large_batch = 1000, large_channels = 3, large_height = 64, large_width = 64;
    std::vector<int> large_shape = {large_batch, large_channels, large_height, large_width};
    // Create a denoised function that returns a random vector
    std::function<std::vector<double>(const std::vector<double>&)> random_denoised_fn = [](const std::vector<double>& x) {
        std::vector<double> result(x.size());
        std::generate(result.begin(), result.end(), []() { return static_cast<double>(rand()) / RAND_MAX; });
        return result;
    };
    // Measure performance
    auto start_time = std::chrono::high_resolution_clock::now();
    DiffusionModel large_model(large_batch * large_channels * large_height * large_width, large_batch * large_channels * large_height * large_width);
    DiffusionSample sampler(large_model, std::vector<double>(1000, 0.1));
    sampler.p_sample_loop_progressive(
        large_shape, true, random_denoised_fn, std::unordered_map<std::string, double>(), "cpu"
    );
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::cout << "Performance test completed in " << duration << " ms.\n";
}

void test_more() {
    // Create a sampler with a specific shape
    int batch = 1, channels = 3, height = 256, width = 256;
    std::vector<int> shape = {batch, channels, height, width};
    // Create a denoised function that returns a random vector
    std::function<std::vector<double>(const std::vector<double>&)> random_denoised_fn = [](const std::vector<double>& x) {
        std::vector<double> result(x.size());
        std::generate(result.begin(), result.end(), []() { return static_cast<double>(rand()) / RAND_MAX; });
        return result;
    };
}
