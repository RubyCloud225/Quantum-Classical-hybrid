#include "H_map.hpp"

#include <vector>
#include <cmath>
#include <iostream>
#include <complex>
#include <utility> // for std::pair

using cplx = std::complex<double>;

inline size_t idx(size_t row, size_t col, size_t dim) {
    return row * dim + col;
}

// anaihilation operator a (d x d) as dense matrix
std::vector<cplx> H_map_Single_Transmon::annihilation_op(size_t d) {
    std::vector<cplx> a(d * d, cplx(0.0, 0.0));
    for (size_t n = 1; < d; ++n) {
        a[idx(n - 1, n, d)] = std::sqrt(static_cast<double>(n));
    }
    return a;
}

// C = A * B
std::vector<cplx> H_map_Single_Transmon::matmul(const std::vector<cplx> &A, const std::vector<cplx> &B, size_t d) {
    std::vector<cplx> C(d * d, cplx(0.0, 0.0));
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            cplx sum(0.0, 0.0);
            for (size_t k = 0; k < d; ++k) {
                sum += A[idx(i, k, d)] * B[idx(k, j, d)];
            }
            C[idx(i, j, d)] = sum;
        }
    }
    return C;
}

// Matrix Addition: C = A + B
std::vector<cplx> H_map_Single_Transmon::matadd(const std::vector<cplx> & A, const std::vector<cplx> & B, size_t d) {
    std::vector<cplx> C(d * d);
    for (size_t i = 0; i < d * d; ++i) {
        C[i] = A[i] + B[i];
    }
    return C;
}

// Scalar-matrix C = s * A
std::vector<cplx> H_map_Single_Transmon::matscale(const std::vector<cplx> &A, cplx s, size_t d) {
    std::vector<cplx> C(d * d);
    for (size_t i = 0; i < d * d; ++i) {
        C[i] = s * A[i];
    }
    return C;
}

// Identity matrix I (d x d)
std::vector<cplx> H_map_Single_Transmon::identity(size_t d) {
    std::vector<cplx> I(d * d, cplx(0.0, 0.0));
    for (size_t i = 0; i < d; ++i) {
        I[idx(i, i, d)] = cplx(1.0, 0.0);
    }
    return I;
}

// (A + A†) / 2
std::vector<cplx> H_map_Single_Transmon::hermitian_sym(const std::vector<cplx> &A, size_t d) {
    std::vector<cplx> C(d * d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            C[idx(i, j, d)] = (A[idx(i, j, d)] + std::conj(A[idx(j, i, d)])) / 2.0;
        }
    }
    for (size_t i = 0; i < d; ++i) {
        C[idx(i, i, d)] = std::real(C[idx(i, i, d)]); // Ensure diagonal elements are real
    }
    return C;
}
// H_{\text{transmon},i} = \omega_i\, a_i^\dagger a_i •	\frac{\alpha_i}{2} \, a_i^\dagger a_i^\dagger a_i a_i
// Transmon Hamiltonian H = omega * n + (alpha/2) * n(n-1)
std::vector<cplx> H_map_Single_Transmon::build_transmons_H(double omega, double alpha, size_t d) {
    auto a = annihilation_op(d);
    auto a_dag = std::vector<cplx>(d * d);
    // a_dag = a†
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            a_dag[idx(i, j, d)] = std::conj(a[idx(j, i, d)]);
        }
    }
    auto n_op = matmul(a_dag, a, d);
    auto n_minus_I = matadd(n_op, matscale(identity(d), cplx(-1.0, 0.0), d), d);
    auto n_n_minus_1 = matmul(n_minus_I, n_op, d);

    auto H = matadd(matscale(n_op, cplx(omega, 0.0), d),
                    matscale(n_n_minus_1, cplx(0.5 * alpha, 0.0), d), d);
    return hermitain_sym(H, d);
}

// Control Hamiltonian H_c,j = (a^dagger + a) / 2
std::pair<std::vector<cplx>, std::vector<cplx>> H_map_Single_Transmon::build_control_H(size_t d) {
    auto a = annihilation_op(d);
    std::vector<cplx> a_dag(d * d);
    for (size_t i = 0; i < d; ++i) {
        for (size_t j = 0; j < d; ++j) {
            a_dag[idx(i, j, d)] = std::conj(a[idx(j, i, d)]);
        }
    }
    auto n_op = matmul(a_dag, a, d);
    auto H_X = matadd(a_dag, a, d);
    H_X = hermitian_sym(H_X, d);
    auto H_Z = hermitian_sym(matadd(a_dag, matscale(identity(d), cplx(1.0, 0.0), d), d), d);
    return std::make_pair(H_X, H_Z);
}

