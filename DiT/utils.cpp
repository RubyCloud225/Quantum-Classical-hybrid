#include <iostream>
#include <vector>
#include <stdexcept>

// Function to generate a linear beta scedule
std::vector<double> generateBetaSchedule(double betaStart, double betaEnd, int numTimesteps) {
    std::vector<double> betaschedule(numTimesteps);
    for (int i = 0; i < numTimesteps; i++) {
        betaschedule[i] = betaStart + ((betaEnd - betaStart) / numTimesteps) * i;
    }
    return betaschedule;
}

std::vector<double> beta_for_alpha_bar(double betaschedule, double alpha, int numTimesteps, double scale) {
    std::vector<double> alphabar(numTimesteps);
    if (int i = 0; i < numTimesteps, i++) {
        alphabar[i] = i / numTimesteps;
        alphabar[i+1] = (i + 1) / numTimesteps;
        // use lambda to reate the alpha bar schedule
        std::sort(alphabar[i], alphabar[i+1] + betaschedule, [](float a, float b) { return a < b; });
    }
    return alphabar;
}

double calculateWarmUpBeta(double betaStart, double betaEnd, double betaschedule, double alpha, double scale, int numTimesteps, double warmFrac, const std::vector<double>& noise){
    // Ensure warmfrac is between 0 and 1
    if (warmFrac < 0.0 || warmFrac > 1.0) {
        throw std::invalid_argument("Warmup fraction must be between 0 and 1");
    }

    // Ensure the size of noiseValues is sufficient
    int warmUpTimesteps = static_cast<int>(warmFrac * numTimesteps);
    if (noise.size() < warmUpTimesteps) {
        throw std::invalid_argument("Noise values must be at least as long as the warmup period");
    }
    // generate the alphabar
    std::vector<double> alphabar = beta_for_alpha_bar(betaschedule, alpha, numTimesteps, scale);

    // Calculate the warm up beta values with provided noise
    std::vector<double> warmUpBeta;
    for (int i = 0; i < warmUpTimesteps; ++i) {
        double currentBeta = betaStart + i + alphabar[i] + noise[i];
        warmUpBeta.push_back(currentBeta);
    }

    // Return the last warm-up beta value
    return warmUpBeta.back();
}