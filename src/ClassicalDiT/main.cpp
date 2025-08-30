#include "BetaSchedule.hpp"
#include "GaussianDiffusion.hpp"
#include "NN/EpsilonPredictor.hpp"
#include "Diffusion_model.hpp"
#include "Diffusion_Sample.hpp"
#include "sampleData.hpp"

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
    GaussianDiffusion gaussian_diffusion(num_timesteps, initial_beta, initial_beta);
    DiffusionModel diffusion_model(input_size, outsize_size); // Example input and output sizes
    // Example data for the forward process
    std::vector<SampleData> loadedSamples = loadSamples("path/to/sample_data.bin");
    if (loadedSamples.empty()) {
        std::cout << "No samples loaded, creating dummy sample data." << std::endl;
        SampleData dummy;
        dummy.token_embedding = {0.1, 0.2, 0.3};
        dummy.normalized_noise = {0.01, 0.02, 0.03};
        dummy.noise = {0.001, 0.002, 0.003};
        dummy.target_value = 1.0;
        dummy.density = 0.5;
        dummy.nll = 0.1;
        dummy.entopy = 0.05;
        loadedSamples.push_back(dummy);
    }
    // log to verify
    for (int i = 0; i < 3 && i < loadedSamples.size(); ++i) {
        const auto& s = loadedSamples[i];
        std::cout << "Sample " << i << ": NLL=" << s.nll << ", Density =" << s.density << ", Entropy=" << s.entopy << std::endl;
    }

    // this is a basic implementation
    std::vector<std::vector<double>> training_noise;
    std::vector<std::vector<double>> targets;
    std::vector<std::vector<double>> nll_losses;
    for (const auto& s : loadedSamples) {
        std::vector<double> combined;
        combined.insert(combined.end(), s.token_embedding.begin(), s.token_embedding.end());
        combined.insert(combined.end(), s.normalized_noise.begin(), s.normalized_noise.end());
        combined.insert(combined.end(), s.noise.begin(), s.noise.end());
        training_noise.push_back(combined);
        targets.push_back({s.target_value});
        nll_losses.push_back({s.nll});
    }
    // Simulate updates for each epoch
    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        // losses (not used in current implementation)
        std::vector<double> nll_losses_epoch;
        std::vector<double> entropy_losses;
        //update beta for the current epoch
        if (epoch >= 0 && epoch < total_epochs) {
            double current_beta = beta_schedule.update(nll_losses_epoch, entropy_losses, epoch);
            std::cout << "Epoch: " << epoch << ", Current Beta: " << current_beta << std::endl;
        } else {
            std::cerr << "Epoch out of range: " << epoch << std::endl;
            break;
        }
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
        gaussian_diffusion.train(training_noise, epoch);
        // After training, compute predictions for NormalDist
        for (size_t i = 0; i < training_noise.size(); ++i) {
            const auto& sample = training_noise[i];
            const double target = targets[i][0];
            const double nll = nll_losses[i][0];
            std::vector<double> x_t = gaussian_diffusion.forward(sample, epoch % num_timesteps);
            EpsilonPredictor predictor(sample.size(), sample.size());
            std::vector<int> epsilon_pred_int = predictor.predictEpsilon(x_t, epoch % num_timesteps);
            std::vector<double> epsilon_pred(epsilon_pred_int.begin(), epsilon_pred_int.end());
            std::vector<double> sampled_x = diffusion_model.sample(x_t, epoch % num_timesteps, true, nullptr, nullptr, {});
            double x_start_pred = sampled_x[0]; // Example prediction
            double eps_pred = epsilon_pred[0]; // Example prediction
            double y = sample[0]; // Example observation
            

            //log probability
            try {
                // Remove or comment out NormalDist usage as it is undefined
                // double log_prob = NormalDist::log_prob_from_predictions(y, x_start_pred, eps_pred);
                // double dfd_y, dfd_mu, dfd_sigma;
                // NormalDist::gradients(y, x_start_pred, eps_pred, dfd_y, dfd_mu, dfd_sigma);
                // std::cout << "Log Probability: " << log_prob << std::endl;
                // std::cout << "Gradients: dfd_y: " << dfd_y << ", dfd_mu: " << dfd_mu << ", dfd_sigma: " << dfd_sigma << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }

        // usage of p_sample_loop_progressive
        std::vector<int> shape = {1, 3, 32, 32}; // Example shape
        bool clip_denoised = true;
        DiffusionSample sampler(diffusion_model, std::vector<double>(1000, 0.1));
        std::unordered_map<std::string, double> model_kwags;
        model_kwags["dummy_param1"] = 1.0;
        model_kwags["dummy_param2"] = 2.0;
        model_kwags["dummy_param3"] = 3.0;
        
        // Create a simple denoised function
        auto denoised_fn = [](const std::vector<double>& x) -> std::vector<double> {
            std::vector<double> result = x;
            for (auto& val : result) {
                val = val * 0.9; // Simple denoising operation
            }
            return result;
        };
        
        auto progressive_sample = sampler.p_sample_loop_progressive(shape, clip_denoised, denoised_fn, model_kwags, "cpu");
        int step = 0;
        for (const auto& sample : progressive_sample) {
            std::cout << "Step: " << step << ", Sample: ";
            for (const auto& val : sample.at("sample")) {
                std::cout << val << " ";
            }
            std::cout << std::endl;
            ++step;
        }
    }
    return 0;
}