#include "training.hpp"
#include "time_H.hpp"
#include <random>
#include <iostream>
#include <vector>
#include <complex>
#include <omp.h>

namespace Training {

    using cplx = Time::cplx;

    Time::Mat toMat(const std::vector<cplx>& vec, int d) {
        if (vec.size() != d * d) throw std::invalid_argument("Vector size does not match matrix dimensions");
        if (d <= 0) throw std::invalid_argument("Matrix dimension must be positive");
        Time::Mat mat(d, d);
        return mat;
    }

    double fidelity (const Time::Vec& psi_target, const Time::Vec& psi_final) {
        cplx overlap(0.0, 0.0);
        #pragma omp parallel for reduction(+:overlap)
        for (int i=0; i < psi_target.size(); ++i) {
            overlap += std::conj(psi_target[i]) * psi_final[i];
        }
        return std::norm(overlap);
    }

    double leakagePenalty(const Time::Vec& psi, int logical_dim) {
        double leakage = 0.0;
        #pragma omp parallel for reduction(+:leakage)
        for (int i = logical_dim; i < psi.size(); ++i) {
            leakage += std::norm(psi[i]);
        }
        return leakage;
    }

    double energyPenalty(const Time::Mat& H, const Time::Vec& psi) {
        Time::Vec Hpsi = H * psi;
        cplx exp_val(0.0,0.0);
        #pragma omp parallel for reduction(+:exp_val)
        for (int i = 0; i < psi.size(); ++i) {
            exp_val += std::conj(psi[i]) * Hpsi[i];
        }
        return std::real(exp_val);
    }

    double bandwidthPenalty(const std::vector<double>& uX, const std::vector<double>& uZ) {
        double sum=0.0;
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < uX.size(); ++i) sum += uX[i]*uX[i];
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < uZ.size(); ++i) sum += uZ[i]*uZ[i];
        return sum;
    }

    double run_episode(int d, const Time::Mat& H0, const Mat& H_X, const Time::Mat& H_Z, const Time::Vec& psi_target, int steps, double dt, double lambda_leak, double lambda_energy, double lambda_band) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-0.2, 0.2);
        Time::Vec psi(d);
        std::vector<double> uX_schedule, uZ_schedule;
        for (int k=0; k<steps; ++k) {
            double uX = dist(rng);
            double uZ = dist(rng);
            uX_schedule.push_back(uX);
            uZ_schedule.push_back(uZ);
            // compose the Hamiltonian
            Time::Mat Hk = composeHamiltonian(H0, {H_X}, {H_Z}, {uX}, {uZ});
            Time::Mat Uk = spectralPropagator(Hk, dt);
            psi = Uk * psi;
        }
        // calculate penalties
        double F = fidelity(psi_target, psi);
        double P_leak = leakagePenalty(psi, 2); // assuming logical dimension is 2
        double P_energy = energyPenalty(H0, psi);
        double P_band = bandwidthPenalty(uX_schedule, uZ_schedule);
        // calculate total cost
        double reward = F - lambda_leak * P_leak - lambda_energy * P_energy - lambda_band * P_band;
        return reward;
    }
}
