#ifndef UTILS_HPP
#define UTILS_HPP
#include <iostream>
#include <vector>

class Utils {
    public:
    double calculateWarmUpBeta(double betaStart, double betaEnd, int numTimesteps, double warmFrac, const std::vector<double>& noiseValues);
    private:
    double betaStart;
    double betaEnd;
    int numTimesteps;
    double warmFrac;
    double scale;
    double alpha;
    double betaschedule;
    std::vector<double> noiseValues;
    std::vector<double> geneateBetaSchedule(double betaStart, double betaEnd, int numTimesteps);
    std::vector<double> beta_for_alpha_bar(double betaschedule, double alpha, int numTimesteps, double scale);
};

#endif // UTILS_HPP