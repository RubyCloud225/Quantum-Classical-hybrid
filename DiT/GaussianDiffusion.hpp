#ifndef GAUSSIANDIFFUSION_HPP
#define GAUSSIANDIFFUSION_HPP

#include <iostream>
#include <cmath>
#include <vector>

class GaussianDiffusion {
    public:
    void Diffusion(const std::vector<double>& mean, const std::vector<double>& variance, const std::vector<double>& loss);
    std::vector<double> calculatemeanvariance();
    std::vector<double> sampleQ();
    std::vector<double> posteriorMeanVariance();
    std::vector<double> Pmean();
    std::vector<double> processX();
    std::vector<double> predictX();
};

#endif // GAUSSIANDIFFUSION_HPP