#include <iostream>
#include <vector>
#include <stdexcept>

double calculateWarmUpBeta(double betaStart, double betaEnd, int numTimesteps, double warmFrac, const std::vector<double>& noiseValues) {
    // Ensure warmfrac is between 0 and 1
    if (warmFrac < 0.0 || warmFrac > 1.0) {
        throw std::invalid_argument("Warmup fraction must be between 0 and 1");
    }

    // Ensure the size of noiseValues is sufficient
    int warmUpTime
}