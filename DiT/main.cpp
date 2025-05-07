#include "BetaSchedule.hpp"
#include "GaussianDiffusion.hpp"
#include "EpsilonPredictor.hpp"
#include "Diffusion_model.hpp"
#include "Diffusion_Sample.hpp"
#include "DataProcessing/sampleData.hpp"

int main() {
    //Initialize parameters
    double initial_beta = 0.1;
    int total_epochs = 1000;
    // Create an instance of BetaSchedule
    BetaSchedule beta_schedule(initial_beta, total_epochs);

    // Initialize parameters for adamOptimizer
    double learning_rate = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int input_size = 3; // Example input size
    int outsize_size = 3; // Example output size
    int epochs = 5;
    AdamOptimizer adam_optimizer(learning_rate, beta1, beta2, epsilon);
    // Create an instance of GaussianDiffusion
    int num_timesteps = 1000;
    std::vector<double> betas(num_timesteps, initial_beta);
    GaussianDiffusion gaussian_diffusion(num_timesteps, betas, adam_optimizer);
    DiffusionModel diffusion_model(input_size, outsize_size); // Example input and output sizes
    // Example data for the forward process
    std::vector<Sample> loadedSamples = loadSamples("path/to/sample_data.bin");
    // log to verify
    for (int i = 0; i < 3 && i < loadedSamples.size(); ++i) {
        const auto& s = loadedSamples[i];
        std::cout << "Sample " << i << ": NLL=" << s.nll << ", Density =" << s.density << ", Entropy=" << s.entropy << std::endl;
    }

    // this is a basic implementation
    std::vector<std::vector<double>> training_noise;
    std::vector<std::vector<double>> targets;
    std::vector<std::vector<double>> nll_losses;
    for (const auto& s : loadedSamples) {
        std::vector<double> combined;
        combined.insert(combined.end(), s.token_embedding.begin(), s.token_embedding.end());
        combined.insert(combined.end(), s.normalized.begin(), s.normalized.end());
        combined.insert(combined.end(), s.noise.begin(), s.noise.end());)
        training_data.push_back(combined);
        targets.push_back({s.target_value});
        nll_losses.push_back({s.nll});
    }
    // Simulate updates for each epoch
    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        // losses (not used in current implementation)
        std::vector<double> nll_losses;
        std::vector<double> entropy_losses;
        //update beta for the current epoch
        double current_beta = beta_schedule.update(nll_losses, entropy_losses, epoch);
        std::cout << "Epoch: " << epoch << ", Current Beta: " << current_beta << std::endl;
        // Update parameters using Adam optimizer
        std::vector<double> params = {0.1, 0.2, 0.3}; // Example parameters
        std::vector<double> gradients = {0.01, 0.02, 0.03}; // Example gradients
        adam_optimizer.update(params, gradients);
        // Print updated parameters
        std::cout << "Updated Parameters: ";
        for (const auto& param : params) {
            std::cout << param << " ";
        }
        std::cout << std::endl;

        // Train Gaussian Diffusion model
        gaussian_diffusion.train(training_noise, targets, nll_losses, epoch);
        // After training, compute predictions for NormalDist
        for (size_t i = 0; i < training_noise.size(); ++i) {
            const auto& sample = training_noise[i];
            const double target = targets[i];
            const double nll = nll_losses[i];
            std::vector<double> x_t = gaussian_diffusion.forward(sample, epoch % num_timesteps);
            EpsilonPredictor predictor(sample.size(), sample.size());
            std::vector<double> epsilon_pred = predictor.predict(x_t, epoch % num_timesteps);
            std::vector<double> sampled_x = diffusion_model.sample(x_t, t, true, nullptr, nullptr, {});
            double x_start_pred = sampled_x[0]; // Example prediction
            double eps_pred = epsilon_pred[0]; // Example prediction
            double y = sample[0]; // Example observation
            

            //log probability
            try {
                double log_prob = NormalDist::log_prob_from_predictions(y, x_start_pred, eps_pred);
                double dfd_y, dfd_mu, dfd_sigma;
                NormalDist::gradients(y, x_start_pred, eps_pred, dfd_y, dfd_mu, dfd_sigma);
                std::cout << "Log Probability: " << log_prob << std::endl;
                std::cout << "Gradients: dfd_y: " << dfd_y << ", dfd_mu: " << dfd_mu << ", dfd_sigma: " << dfd_sigma << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }

        // usage of p_sample_loop_progressive
        std::vector<int> shape = {1, 3, 32, 32}; // Example shape
        bool clip_denoised = true;
        auto progressive_sample = Diffusion_Sample.p_sample_loop_progressive(training_noise, targets, shape, clip_denoised, nll_losses, epoch);
        int step = 0;
        for (const auto& sample : progressive_sample) {
            std::cout << "Step: " << step << ", Sample: ";
            for (const auto& val : sample) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
            ++step;
        }
    }
    return 0;
}