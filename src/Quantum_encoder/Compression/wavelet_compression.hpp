#ifndef WAVELET_COMPRESSION_HPP
#define WAVELET_COMPRESSION_HPP

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>

#include "core_propagation.hpp"
#include "propagate_hamiltonian.hpp"

namespace py = pybind11;
using cdouble = std::complex<double>;

void qubits_to_wave(const std::vector<cdouble>& qubits, std::vector<cdouble>& psi, int nx, int ny);
// Function to perform wavelet compression on a numpy array
py::dict propagate_and_compress(int nx, int ny, int nz, double dx, double dz, double k, double re, int J double F_star);

#endif // WAVELET_COMPRESSION_HPP