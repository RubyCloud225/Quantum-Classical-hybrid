#include <vector>
#include "BetaSchedule.hpp"
#include "GaussianDiffusion.hpp"
#include "EpsilonPredictor.hpp"
#include "Diffusion_model.hpp"
#include "Diffusion_Sample.hpp"
#include "DataProcessing/sampleData.hpp"
#include "train_diffusion_model.hpp"

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
    const std::string& sample_data_path) {
    // Create an instance of BetaSchedule
    BetaSchedule beta_schedule(initial_beta, total_epochs);

    // Create an instance of GaussianDiffusion
    int num_timesteps = 1000;
    std::vector<double> betas(num_timesteps, initial_beta);
    GaussianDiffusion gaussian_diffusion(num_timesteps, betas);

    // Create an instance of DiffusionModel
    DiffusionModel diffusion_model(input_size, output_size); // Example input and output sizes

    // Load sample data
    std::vector<Sample> loadedSamples = loadSamples(sample_data_path);
    std::vector<std::vector<double>> training_noise, targets, nll_losses;

    for (const auto& s : loadedSamples) {
        std::vector<double> combined;
        combined.insert(combined.end(), s.token_embedding.begin(), s.token_embedding.end());
        combined.insert(combined.end(), s.normalized.begin(), s.normalized.end());
        combined.insert(combined.end(), s.noise.begin(), s.noise.end());
        training_noise.push_back(combined);
        targets.push_back({s.target_value});
        nll_losses.push_back({s.nll});
    }

    // Simulate updates for each epoch
    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        double current_beta = beta_schedule.update(nll_losses, {}, epoch);
        gaussian_diffusion.train(training_noise, targets, nll_losses, epoch);
        for (int size_t i = 0; i < training_noise.size(); ++i) {
            const auto& sample = training_noise[i];
            const double target = targets[i];
            const double nll = nll_losses[i];
            std::vector<double> x_t = gaussian_diffusion.forward(sample, epoch % num_timesteps);
            EpsilonPredictor epsilon_predictor(sample.size(), sample.size());
            std::vector<double> eps_pred = epsilon_predictor.predict(x_t, epoch % num_timesteps);
            std::vector<double> sampled_x = diffusion_model.forward(x_t, epoch % num_timesteps, false, nullptr, {});

            try {
                double log_prob = NormalDist::log_prob_from_predictions(sampled_x[0], x_t[0], eps_pred[0]);
                double dfd_y, dfd_mu, dfd_sigma;
                NormalDist::gradients(sampled_x[0], x_t[0], eps_pred[0], dfd_y, dfd_mu, dfd_sigma);
                // Print gradients
                std::cout << "Gradients: dfd_y=" << dfd_y << ", dfd_mu=" << dfd_mu << ", dfd_sigma=" << dfd_sigma << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }
        //Progressive sampling
        std::vector<int> shape = {1, 3, 64, 64}; // Example shape
        bool clip_denoised = true;
        DiffusionSample diffusion_sample;
        auto progressive_samples = diffusion_sample.p_sample_loop_progressive(
            training_noise, targets, nll_losses, epoch, shape, clip_denoised);
        // log the progressive samples
        int step = 0;
        for (const auto& sample : progressive_samples) {
            std::cout << "Step " << step++ << ": ";
            for (const auto& value : sample) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
            ++step;
        }
    }
}