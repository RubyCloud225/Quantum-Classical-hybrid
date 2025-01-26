#include "Utils.hpp"
#include <vector>
#include <string>
#include <iostream>
#include "meanmodel.hpp"

int DiffusionMain() {
    double betaStart = 0.1;
    double betaEnd = 1.0;
    int numTimesteps = 10;
    double warmFrac = 0.5;
    double scale = 1000;
    double betaschedule = 0.5;
    double alpha = 0.8;

    // precalculated noise values (example values)
    std::vector<double> noiseValues = {0.02, -0.01, 0.03, 0.01, -0.02}; // this needs to adjust for calculated noise values

    try {
        Utils calculateWarmUpBeta (double betaStart, double betaEnd, double betaschedule, double alpha, double scale, int numTimesteps, double warmFrac, std::vector<double>& noiseValues);
        std::cout << calculateWarmUpBeta << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    MeanVariance mv(noiseValues);
    double mean = mv.calculateMean();
    double variance = mv.calculateVariance();
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Variance: " << variance << std::endl;

    std::vector<double> previousXPredictions = mv.makeXPredictions(numTimesteps);
    std::vector<double> startXPredictions = mv.makeStartXPredictions(numTimesteps);

    return 0;
}