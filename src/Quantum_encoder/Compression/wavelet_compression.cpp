#include "wavelet_compression.hpp"
#include <cmath>
#include <iostream>
#include <complex>


using cdouble = std::complex<double>;

py::dict propagate_and_compress(int nx, int ny, int nz, double dx, double dz, double k, double re, int J, double F_star) {
    // Allocate the 3d field
    std::vector<std::<cdouble>> psi(nx * ny * nz, 0.0);
}