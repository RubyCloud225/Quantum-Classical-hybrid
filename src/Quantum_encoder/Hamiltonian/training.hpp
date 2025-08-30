#ifndef TRAINING_HPP
#define TRAINING_HPP

#include <vector>
#include <complex>
#include <random>
#include <iostream>
#include "time_H.hpp"

namespace Training {
    using cplx = std::complex<double>;

    class Training {
        public:
        Time::Mat toMat(const std::vector<cplx>& vec, int d);
        Time::Vec toVec(const std::vector<cplx>& vec);
        double run_episode(int d, const Time::Mat& H0, const Time::Mat& H_X, const Time::Mat& H_Z, const Time::Vec& psi_target, int steps, double dt, double lambda_energy,double lambda_band);
        private:
        double fidelity(const Time::Vec& psi_target, const Time::Vec& psi_final);
        double leakagePenalty(const Time::Vec& psi, int logical_dim);
        double energyPenalty(const Time::Mat& H, const Time::Vec& psi);
        double bandwidthPenalty(const std::vector<double>& uX, const std::vector<double>& uZ);
    };
};
#endif // TRAINING_HPP