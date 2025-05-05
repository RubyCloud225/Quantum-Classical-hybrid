#include "Diffusion_Sample.hpp"
#include "Diffusion_model.hpp"

DiffusionSample::DiffusionSample(DiffusionModel& model, const std::vector<double>& noise_schedule)
    : model(model), noise_schedule_(noise_schedule), generator_(std::random_device{}()), normal_dist_(0.0, 1.0) {}

std::vector<std::vector<double>> DiffusionSample::p_sample(
    const std::vector<int>& shape,
    bool clip_denoised,
    const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn,
    const std::unordered_map<std::string, double>& model_kwags,
    const std::string& device
) {
    std::vector<std::vector<double>> samples(shape[0], std::vector<double>(shape[1] * shape[2] * shape[3], 0.0));
    for (int n = 0; n < shape[0]; ++n) {
        std::vector<double> x_t(shape[1] * shape[2] * shape[3], 0.0);
        for (int t = shape[0] - 1; t >= 0; --t) {
            double noise = distribution_(generator_) * std::sqrt(variance[i]);
            x_t[i] = mean[i] + noise;
        }

        if (denoised_fn) {
            x_t = denoised_fn(x_t);
        }

        if (clip_denoised) {
            for (auto& value : x_t) {
                value = std::clamp(value, -1.0, 1.0);
            }
        }
    samples[n] = x_t; // Store the sample
    }
    return samples;
}

std::vector<std::unordered_map<std::string, std::vector<double>>> DiffusionSample::p_sample_loop_progressive(
    const std::vector<int>& shape,
    bool clip_denoised,
    const std::function<std::vector<double>(const std::vector<double>&)>& denoised_fn,
    const std::unordered_map<std::string, double>& model_kwags,
    const std::string& device
) {
    std::vector<std::unordered_map<std::string, std::vector<double>>> samples;
    for (int n = 0; n < shape[0]; ++n) {
        std::unordered_map<std::string, std::vector<double>> sample;
        std::vector<double> x_t(shape[1] * shape[2] * shape[3], 0.0);
        for (int t = shape[0] - 1; t >= 0; --t) {
            double noise = distribution_(generator_) * std::sqrt(variance[i]);
            x_t[i] = mean[i] + noise;
        }

        if (denoised_fn) {
            x_t = denoised_fn(x_t);
        }

        if (clip_denoised) {
            for (auto& value : x_t) {
                value = std::clamp(value, -1.0, 1.0);
            }
        }
        sample["sample"] = x_t; // Store the sample
        samples.push_back(sample);
    }
    return progressive_samples;
}