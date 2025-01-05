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
    std::vector<double> noiseValues;
};

#endif // UTILS_HPP