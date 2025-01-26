#include "GaussianDiffusion.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <random>

std::vector<double> GaussianDiffusion::diffusion(double beta, double meantype, double vartype, double losstype, double alpha) {
    if (beta > 0 || beta < 1) {
        throw std::invalid_argument("Beta must be between 0 and 1");
    }
    
}