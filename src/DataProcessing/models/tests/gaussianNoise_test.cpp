#include "../GaussianNoise.hpp"
#include <iostream>
#include <vector>

int main() {
    std::vector<double> mean = {0.0, 0.0};
    std::vector<std::vector<double>> covariance = {{1.0, 0.5}, {0.5, 1.0}};
    std::vector<double> weights = {1.0, 1.0};

    //Create Gaussian Noise
    GaussianNoise gaussianNoise(mean, covariance, weights);
    //Generate Noise
    std::vector<double> noise = gaussianNoise.generateNoise();
    //Generate Noise Sample
    std::vector<double> sample = gaussianNoise.generateNoise();
    std::cout << "Generated Noise: ";
    for (const auto& val : noise) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    //Calculate Density
    double density = gaussianNoise.calculateDensity(noise);
    std::cout << "Density: " << density << std::endl;
    //Calculate NLL of the sample
    double nll = gaussianNoise.negativeLogLikelihood(sample);
    std::cout << "Negative Log Likelihood: " << nll << std::endl;
    // Calculate entropy of the distribution
    double entropy = gaussianNoise.calculateEntropy();
    std::cout << "Entropy: " << entropy << std::endl;
    return 0;

}
