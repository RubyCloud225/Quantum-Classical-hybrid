#include "GaussianDiffusion.hpp"
#include "EpsilonPredictor.hpp"
#include <omp.h>

// Adam Optimizer Constructor
AdamOptimizer::AdamOptimizer(double learning_rate, double beta1, double beta2, double epsilon) : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {
    // Initialize first and second moment vectors
}

void AdamOptimizer::update(std::vector<double>& params, std::vector<double>& gradients) {
    t_++;
    if (m_.empty()) {
        m_.resize(params.size(), 0);
        v_.resize(params.size(), 0);
    }
    #pragma omp parallel for
    for (size_t i = 0; i < params.size(); ++i) {
        m_[i] = beta1_ * m_[i] + (1 - beta1_) * gradients[i];
        v_[i] = beta2_ * v_[i] + (1 - beta2_) * gradients[i] * gradients[i];
        double m_hat = m_[i] / (1 - std::pow(beta1_, t_));
        double v_hat = v_[i] / (1 - std::pow(beta2_, t_));
        params[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
    }
}

// Gaussian Diffusion Constructor
GaussianDiffusion::GaussianDiffusion(int num_timesteps, double beta_start, double beta_end) : num_timesteps_(num_timesteps), beta_start_(beta_start), beta_end_(beta_end), optimizer_(0.001, 0.9, 0.999, 1e-8) {
    betas_.resize(num_timesteps_);
    for (int t = 0; t < num_timesteps_; ++t) {
        betas_[t] = beta_start + (beta_end - beta_start) * (static_cast<double>(t) / num_timesteps_);
    }
}

// Forward process
std::vector<double> GaussianDiffusion::forward(const std::vector<double>& x_prev, int t) {
    std::vector<double> x_t(x_prev.size());
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, std::sqrt(betas_[t]));
    for (size_t i = 0; i < x_prev.size(); ++i) {
        x_t[i] = x_prev[i] + noise(generator);
    }
    return x_t;
}

std::vector<double> GaussianDiffusion::reverse(const std::vector<double>& x_t, int t) {
    std::vector<double> x_prev(x_t.size());
    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, 1.0); //placeholder
    double beta_t = betas_[t];
    double sqrt_one_minus_beta_t = std::sqrt(1.0 - beta_t);
    double sqrt_beta_t = std::sqrt(beta_t);
    for (size_t i = 0; i < x_t.size(); ++i) {
        // Estimate the noise term ( replace this with a better method)
        double epsilon = noise(generator); // sample
        // mean from the reverse process
        double mu_t_minus_1 = (x_t[i] - sqrt_beta_t * epsilon) / sqrt_one_minus_beta_t;
        // sample x_{t_1}
        x_prev[i] = mu_t_minus_1 + sqrt_beta_t * noise(generator);
    }
    return x_prev; // placeholder
}

//Sigmoid activation function
double GaussianDiffusion::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of the sigmoid function
double GaussianDiffusion::sigmoid_derivative(double x) const {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

void GaussianDiffusion::train(const std::vector<std::vector<double>>& data, int epochs) {
    int input_size = data.empty() ? 0 : data[0].size();
    int output_size = input_size; // Assuming output size same as input size

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            // Current timestamp
            int t = epoch % num_timesteps_;
            // Forward process
            std::vector<double> x_t = forward(sample, t);
            //Predict Epsilon
            EpsilonPredictor epsilon_predictor(input_size, output_size);
            std::vector<int> epsilon_t = epsilon_predictor.predictEpsilon(x_t, t);
            // Calculate mean, variance and log variance
            double beta_t = betas_[t];
            std::vector<double> mu(x_t.size());
            std::vector<double> variance(betas_[t]);
            std::vector<double> log_var(x_t.size());

            for (size_t i = 0; i < x_t.size(); ++i) {
                mu[i] = (x_t[i] - beta_t * epsilon_t[i]) / std::sqrt(1 - beta_t);
            }
            //predict X_start
            std::vector<double> x_start(x_t.size());
            for (size_t i = 0; i < x_t.size(); ++i) {
                x_start[i] = mu[i] + epsilon_t[i] * std::exp(0.5 *log_var[i]);
            }
            // Gradient process
            std::vector<double> gradients(x_t.size());
            for (size_t i = 0; i < x_t.size(); ++i) {
                gradients[i] = (x_t[i] - x_start[i]) * sigmoid_derivative(x_start[i]);
            }
            std::vector<double> params(x_t.size());
            optimizer_.update(params, gradients);
        }
    }
}
