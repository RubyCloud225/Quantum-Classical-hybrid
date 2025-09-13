#ifndef CORE_PROPAGATION_HPP
#define CORE_PROPAGATION_HPP


#include <complex>

namespace wavepropagation {

    using cdouble = std::complex<double>;

    // indexing function for 2D arrays stored in 1D
    inline in idx2d(int i, int j, int nx, int ny);
    void laplacian_2d(const std::vector<cdouble>& psi, std::vector<cdouble>& lap, int nx, int ny, double dx);
    void propagate_step_2d(std::vector<cdouble> & psi, int nx, int ny, double dx, double dz, double k, double re, double z_val);
    void propagate_3d(py::array_t<cdouble> psi_in, int nx, int ny, int nz, double dx, double dz, double k, double re);

} // namespace wavepropagation

