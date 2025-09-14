#ifndef PROPAGATE_HAMILTONIAN_HPP
#define PROPAGATE_HAMILTONIAN_HPP

#include <complex>
#include <vector>
#include <cmath>

namespace Hamiltonian_propagation {

    using cdouble = std::complex<double>;

    // indexing function for 2D arrays stored in 1D
    inline int idx2d(int i, int j, int nx, int ny);
    double N(double x, double y, double z);
    void laplacian_2d(const std::vector<cdouble>& R, const std::vector<cdouble>& I, std::vector<double>& lapR, std::vector<double>& lapI, int nx, int ny, double dx);
    void propagate_hamiltonian_step(std::vector<double>& R, std::vector<double>& I, int nx, int ny, double dx, double dz, double k, double re, double z_val);
    double intensity(const std::vector<double>& R, const std::vector<double>& I);
    void propagate_hamiltonian_3d(std::<cdouble> psi_in, int nx, int ny, int nz, double dx, double dz, double k, double re);


} // namespace hamiltonian_propagation