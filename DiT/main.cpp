#include "BetaSchedule.hpp"
#include "GaussianDiffusion.hpp"
#include "EpsilonPredictor.hpp"

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
    AdamOptimizer adam_optimizer(learning_rate, beta1, beta2, epsilon);
    // Create an instance of GaussianDiffusion
    int num_timesteps = 1000;
    std::vector<double> betas(num_timesteps, initial_beta);
    GaussianDiffusion gaussian_diffusion(num_timesteps, betas, adam_optimizer);
    // Example data for the forward process
    std::vector<std::vector<double>> training_data = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9}
    };
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
        gaussian_diffusion.train(training_data, 10, 0.1);
        // After training, compute predictions for NormalDist
        for (const auto& sample : training_data) {
            std::vector<double> x_t = gaussian_diffusion.forward(sample, epoch % num_timesteps);
            EpsilonPredictor predictor(sample.size(), sample.size());
            std::vector<double> epsilon_pred = predictor.predict(x_t, epoch % num_timesteps);
            double x_start_pred = x_t[0]; // Example prediction
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
    }
    return 0;
}