#include "wavelet_compression.hpp"
#include <cmath>
#include <iostream>
#include <complex>


using cdouble = std::complex<double>;

// map to 2D wavelet transform psi(x, y)
// input state to size N = 2^n
// output vector of wavelet coefficients at each scale

void qubits_to_wave(const std::vector<cdouble>& qubits, std::vector<cdouble>& psi, int nx, int ny) {
    size_t N = qubits.size();
    if (nx * ny !=N) {
        throw std::runtime_error("Size mismatch in qubits_to_wave");
    }
    // Normalise qubit amplitudes if needed
    double norm = 0.0;
    for (const auto& q : qubits) norm += std::norm(q);
    norm = std::sqrt(norm);
    for (size_t k = 0; k < N; k++) {
        int j = k / ny;
        int j = k % ny;
        psi[i * ny + j] = qubits[k] / norm;
}

py::dict propagate_and_compress(int nx, int ny, int nz, double dx, double dz, double k, double re, int J, double F_star) {
    // Allocate the 3d field
    std::vector<std::<cdouble>> psi(nx * ny * nz, 0.0);

    // Propagate the field using Hamiltonian propagation, this calls in our hamiltonian library
    py::array_t<std::cdouble> psi_array({nz, ny, nx}, reinterpret_cast<std::cdouble*>(psi.data()));
    hamiltonian_propagation(psi.data(), nx, ny, nz, dx, dz, k, re);
    // Perform wavelet compression
    std::vector<double> E_j(J, 0.0);
    std::vector<double> eps_j(J, 0.0);

    for (int iz=0; iz<nz; iz++) {
        std::vector<std::cdouble> slice(nx * ny);
        for (int iy=0; iy<ny; iy++) {
            for (int ix=0; ix<nx; ix++) {
                slice[iy * nx + ix] = psi[iz * nx * ny + iy * nx + ix];
            }
        }
        auto coeffs = wavelet_transform_2d(slice, nx, ny, J);
        for (int j=0; j<J; j++) {
            double energy = 0.0;
            for (const auto& c : coeffs[j]) {
                energy += std::norm(c);
            }
            E_j[j] += energy;
        }
    }
    for (int j=0; j<J; j++) {
        double sigma2 = E_j[j] / (nx * ny);
        double target_drop = 2*(1.0-F_star)/J;
        eps_j[j] = std::sqrt(-sigma2 * std::log(1.0 - target_drop / (nx * ny * sigma2)));
    }
    py::dict result;
    result["E_j"] = py::array_t<double>(E_j.size(), E_j.data());
    result["eps_j"] = py::array_t<double>(eps_j.size(), eps_j.data());
    result["J"] = J;
    result["F_star"] = F_star;
    result["psi"] = psi_array;
    return result;
}