#include "core_propagation.hpp"
#include <vector>
#include <cmath>
#include <complex>

namespace wave = wavepropagation;
using cdouble = std::complex<double>;

// indexing function for 2D arrays stored in 1D
// using psi (x_i, y_j) = psi[i * ny + j] for (i in [0, nx), j in [0, ny))
// i = 0..nx-1, j = 0..ny-1
// converting 2D (i, j) to 1D index
inline int idx2D(int i, int j, int nx, int ny) {
    return i * ny + j;
}

// Calculating the plasma density to simulate the wave propagation
// Transverse Laplacian with periodic boundary conditions
// ∇_⊥^2 ψ = (ψ(x+dx, y) + ψ(x-dx, y) + ψ(x, y+dy) + ψ(x, y-dy) - 4ψ(x, y)) / dx^2
void laplacian_2d(const std::vector<cdouble>& psi, std::vector<cdouble>& lap, int nx, int ny, double dx) {
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            int ip = (i + 1) % nx; // periodic boundary
            int im = (i - 1 + nx) % nx; // periodic boundary
            int jp = (j + 1) % ny; // periodic boundary
            int jm = (j - 1 + ny) % ny; // periodic boundary
            lap[idx2D(i, j, nx, ny)] = (psi[idx2D(ip, j, nx, ny)] + psi[idx2D(im, j, nx, ny)] +
                                        psi[idx2D(i, jp, nx, ny)] + psi[idx2D(i, jm, nx, ny)] -
                                        4.0 * psi[idx2D(i, j, nx, ny)]) / (dx * dx);
        }
    }
}

// Euler Propagation step
// ψ(z+dz) = ψ(z) - i dz [ - (1/2k) ∇_⊥^2 ψ + V ψ ]
void propagate_step_2d(std::vector<cdouble> & psi, int nx, int ny, double dx, double dz, double k, double re, double z_val) {
    std::vector<cdouble> lap(nx*ny);
    laplacian_2d(psi, lap, nx, ny, dx);
    for (int i = 0; i < nx; ++i) {
        double x = (i - nx/2) * dx;
        for (int j = 0; i < ny; ++ j) {
            double y = (j - ny/2) * dx;
            cdouble potential = (re / (2.0*k)) * N(x, y, z_val);
            // Euler step: ψ(z+dz) = ψ(z) - i dz [ - (1/2k) ∇_⊥^2 ψ + V ψ ]
            psi[idx2D(i, j, nx, ny)] += dz * (1i / (2.0 * k) * lap[idx2D(i, j, nx, ny)] - 1i * potential * psi[idx2D(i, j, nx, ny)]);
        }
    }
}

// propagate full 3D field
// psi_in: input 3D numpy array (nz, nx, ny)
// nx, ny: transverse dimensions
// nz: number of propagation steps
void propagate_3d(py::array_t<cdouble> psi_in, int nx, int ny, int nz, double dx, double dz, double k, double re) {
    auto buf = psi_in.request();
    if (buf.ndim != 3) throw std::runtime_error("Input array must be 3-dimensional");
    if (buf.shape[0] != nz || buf.shape[1] != nx || buf.shape[2] != ny) throw std::runtime_error("Input array shape does not match specified dimensions");
    cdouble* ptr = static_cast<cdouble*>(buf.ptr);
    // Loop over propagation steps
    for (int iz = 0; iz < nz - 1; ++iz) {
        double z_val = iz * dz;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                // Extract 2D slice at current z
                slice[i * ny + j] = ptr[iz * nx * ny + i * ny + j];
                // perform Eular step propagation on the 2D slice
                propagate_step_2d(slice, nx, ny, dx, dz, k, re, z_val);
                // Write back the updated slice to the 3D array
                for (int i = 0; i < nx; ++i) {
                    for (int j = 0; j < ny; ++j) {
                        ptr[(iz + 1) * nx * ny + i * ny + j] = slice[i * ny + j];
                    }
                }
            }
        }
    }
}
       