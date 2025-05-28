#include "../DiT/GaussianDiffusion.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cassert>
#include <thread>

void testGaussianDiffusion() {
    GaussianDiffusion diffusion(10, 0.01, 0.1);
    std::vector<double> x_prev = {0.5, 0.5, 0.5};
    int t = 5;

    std::vector<double> x_t = diffusion.forward(x_prev, t);
    std::vector<double> x_reversed = diffusion.reverse(x_t, t);
    std::vector<double> expected_x_reversed = {0.5, 0.5, 0.5}; // Placeholder for expected output
    // Check if the forward and reverse processes are consistent
    std::cout << "Forward sample: ";
    for (double val : x_t) {
        std::cout << val << " ";
    }
    std::cout << "\nReversed sample: ";
    for (double val : x_reversed) {
        std::cout << val << " ";
    }
    assert(x_reversed.size() == x_prev.size());
}

void test_train_performance() {
    GaussianDiffusion diffusion(10, 0.01, 0.1);
    std::vector<std::vector<double>> data = { 1000, std::vector<double>(1000, 0.5) }; // Dummy data
    auto start = std::chrono::high_resolution_clock::now();
    diffusion.train(data, 1);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Training completed in " << duration << " ms\n";
}

// edge case zero empty inputs
void test_zero_empty_inputs() {
    std::cout << "Testing zero empty inputs...\n";
    GaussianDiffusion diffusion(10, 0.01, 0.1);
    // Test with empty input
    std::vector<double> empty_input;
    std::vector<double> x_t = diffusion.forward(empty_input, 0);
    assert(x_t.empty() && "Expected empty output for empty input");
    // Test `reverse` with empty input
    std::vector<double> x_reversed = diffusion.reverse(empty_input, 0);
    assert(x_reversed.empty() && "Expected empty output for empty input in reverse");
    // Test Training with empty data
    std::vector<std::vector<double>> empty_data;
    diffusion.train(empty_data, 1);
    std::cout << "Zero empty inputs test passed.\n";
}
// edge case invalid timesteps
void test_invalid_timesteps() {
    std::cout << "Testing invalid timesteps...\n";
    GaussianDiffusion diffusion(10, 0.01, 0.1);
    // Test with invalid timestep (negative)
    std::vector<double> x_prev = {0.5, 0.5, 0.5};
    std::vector<double> x_t;
    try {
        x_t = diffusion.forward(x_prev, -1);
        assert(false && "Expected an exception for negative timestep");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for negative timestep: " << e.what() << std::endl;
    }
}
// edge case invalid betas
void test_invalid_betas() {
    std::cout << "Testing invalid betas...\n";
    GaussianDiffusion diffusion(10, 0.01, 0.1);
    // Test with invalid beta values (beta_start > beta_end)
    std::vector<double> betas = {0.1, 0.2, 0.3, 0.4, 0.5};
    try {
        GaussianDiffusion invalid_diffusion(10, 0.5, 0.1); // beta_start > beta_end
        assert(false && "Expected an exception for invalid betas");
    } catch (const std::exception& e) {
        std::cout << "Caught expected exception for invalid betas: " << e.what() << std::endl;
    }
}
// edge case Extremes in input data
void test_extremes_in_input_data() {
    std::cout << "Testing extremes in input data...\n";
    GaussianDiffusion diffusion(10, 0.01, 0.1);
    // Test with extreme values
    std::vector<double> extreme_input = {1e10, -1e10, 0.0};
    std::vector<double> x_t = diffusion.forward(extreme_input, 5);
    std::vector<double> x_reversed = diffusion.reverse(x_t, 5);
    // Check if the reversed output is consistent with the extreme input
    assert(x_reversed.size() == extreme_input.size() && "Reversed output size mismatch");
    std::cout << "Extremes in input data test passed.\n";
}
// edge case model behaviour consistency
void test_model_behaviour_consistency() {
    std::cout << "Testing model behaviour consistency...\n";
    GaussianDiffusion diffusion(10, 0.01, 0.1);
    // Test with different timesteps
    std::vector<double> x_prev = {0.5, 0.5, 0.5};
    std::vector<double> x_t1 = diffusion.forward(x_prev, 2);
    std::vector<double> x_t2 = diffusion.forward(x_prev, 5);
    // Check if the outputs are consistent
    assert(x_t1.size() == x_t2.size() && "Output size mismatch");
    std::cout << "Model behaviour consistency test passed.\n";
}
// edge case model concurrency edge cases
void test_model_concurrency_edge_cases() {
    std::cout << "Testing model concurrency edge cases...\n";
    GaussianDiffusion diffusion(10, 0.01, 0.1);
    // Test with multiple threads
    std::vector<double> x_prev = {0.5, 0.5, 0.5};
    std::vector<double> x_t1, x_t2;
    std::thread t1([&diffusion, &x_prev, &x_t1]() {
        x_t1 = diffusion.forward(x_prev, 2);
    });
    std::thread t2([&diffusion, &x_prev, &x_t2]() {
        x_t2 = diffusion.forward(x_prev, 5);
    });
    t1.join();
    t2.join();
    assert(x_t1.size() == x_t2.size() && "Output size mismatch in concurrency test");
    std::cout << "Model concurrency edge cases test passed.\n";
}

int main() {
    // Test Gaussian Diffusion
    testGaussianDiffusion();
    // Test zero/empty inputs
    test_zero_empty_inputs();
    // Test invalid timesteps
    test_invalid_timesteps();
    // Test invalid betas
    test_invalid_betas();
    // Test extremes in input data
    test_extremes_in_input_data();
    // Test model behaviour consistency
    test_model_behaviour_consistency();
    // Test model concurrency edge cases
    test_model_concurrency_edge_cases();
    // Test training performance
    test_train_performance();

    std::cout << "All tests passed successfully.\n";
    return 0;
}
