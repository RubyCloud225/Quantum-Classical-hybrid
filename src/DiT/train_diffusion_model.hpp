#pragma once
#include <string>

void train_diffusion_model(
    double initial_beta,
    int total_epochs,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    int input_size,
    int output_size,
    int epochs,
    const std::string& sample_data_path
);