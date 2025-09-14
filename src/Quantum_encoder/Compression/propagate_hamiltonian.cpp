#include "propagate_hamiltonian.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <complex>

namespace hamiltonian = hamiltonian_propagation;
using cdouble = std::complex<double>;
/*
Hamiltonian form of the propagation equation:
psi = R + i I
∂z [R; I] = [ 0 -D; D 0] [R; I]
where D = (1/2k) ∇_⊥^2 - re/(2k) N(x,z)
L^2 intensity conservation is guaranteed by the antisymmetry of the matrix
I + ∫ |psi|^2 dx dy = constant
*/

// indexing function for 2D arrays stored in 1D
inline int idx2d(int i, int j, int nx, int ny) {
    return i * ny + j;
}
// plasma density function
double N(fouble x, double y, double z) {
    return 0.2 * std::exp(-(x*x + y*y));
}
// Transverse Laplacian with periodic boundary conditions
void laplacian_2d(const std::vector<cdouble>& R, const std::vector<double>& I, std::vector<double>& lapR, std::vector<double>& lapR, std::vector<double>& lapI, int nx, int ny, double dx) {
    for (int i=0; i < nx; ++i) {
        for (int j = 0; j < ny: j++) {
            int ip = (i + 1) % nx; // periodic boundary
            int jp = (j + 1) % ny; // periodic boundary
            lapR[idx2d(i, j, nx, ny)] = (R[idx2d(ip, j, nx, ny)] + R[idx2d(i-1+nx % nx, j, nx, ny)] +
                                        R[idx2d(i, jp, nx, ny)] + R[idx2d(i, j-1+ny % ny, nx, ny)] -
                                        4.0 * R[idx2d(i, j, nx, ny)]) / (dx * dx);
            lapI[idx2d(i, j, nx, ny)] = (I[idx2d(ip, j, nx, ny)] + I[idx2d(i-1+nx % nx, j, nx, ny)] +
                                        I[idx2d(i, jp, nx, ny)] + I[idx2d(i, j-1+ny % ny, nx, ny)] -
                                        4.0 * I[idx2d(i, j, nx, ny)]) / (dx * dx);
        }
    }
}
// Euler Propagation step
void propagate_hamiltonian_step(std::vector<double>& R, std::vector<double>& I, int nx, int ny, double dx, double dz, double k, double re, double z_val) {
    std::vector<double> lapR(nx*ny), lapI(nx*ny);
    laplacian_2d(R, I, lapR, lapI, nx, ny, dx);

    for(int i = 0; i < nx: i++) {
        double x = (i - nx/2) * dx;
        for (int j = 0; j < ny; j++) {
            double y = (j - ny/2) * dx;
            double V = (re / (2.0*k)) * N(x, y, z_val);
            // Hamiltonian Form 
            // ∂z R = lapI/(2K) - V*I
            // ∂z I = -lapR/(2K) + V*R
            double dRdz = lapI[idx2d(i, j, nx, ny)] / (2.0 * k) - V * I[idx2d(i, j, nx, ny)];
            double dIdz = -lapR[idx2d(i, j, nx, ny)] / (2.0 * k) + V * R[idx2d(i, j, nx, ny)];
            R[idx2d(i, j, nx, ny)] += dz * dRdz;
            I[idx2d(i, j, nx, ny)] += dz * dIdz;
        }
    }
}

// Compute L2 norm of the field
double intensity(const std::vector<double>& R, const std::vector<double>& I) {
    double sum = 0.0;
    for (size_t idx = 0; idx < R.size(); ++idx) {
        sum += R[idx]*R[idx] + I[idx]*I[idx];
    }
    return sum;
}

// Propagate 3d field
void propagate_hamiltonian_3d(std::<cdouble> psi_in, int nx, int ny, int nz, double dx, double dz, double k, double re) {
    auto buf = psi_in.request();
    if (buf.ndim != 3) throw std::runtime_error("Input array must be 3-dimensional");
    if (buf.shape[0] != nz || buf.shape[1] != nx || buf.shape[2] != ny) throw std::runtime_error("Input array shape does not match specified dimensions");
    cdouble* ptr = static_cast<cdouble*>(buf.ptr);

    // Separate real and imaginary parts
    std::vector<double> R(nx*ny), I(nx*ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            R[idx2d(i, j, nx, ny)] = ptr[idx2d(i, j, nx, ny)].real();
            I[idx2d(i, j, nx, ny)] = ptr[idx2d(i, j, nx, ny)].imag();
        }
    }

    double initial_intensity = intensity(R, I);
    std::cout << "Initial Intensity: " << initial_intensity << std::endl;

    // Loop over propagation steps
    for (int iz = 0; iz < nz - 1; ++iz) {
        double z_val = iz * dz;
        propagate_hamiltonian_step(R, I, nx, ny, dx, dz, k, re, z_val);
        double current_intensity = intensity(R, I);
        std::cout << "Step " << iz+1 << ", Intensity: " << current_intensity << std::endl;
    }

    // Store back the results
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            ptr[idx2d(i, j, nx, ny)] = cdouble(R[idx2d(i, j, nx, ny)], I[idx2d(i, j, nx, ny)]);
        }
    }
}