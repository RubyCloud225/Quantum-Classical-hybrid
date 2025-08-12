#ifndef H_MAP_HPP
#define H_MAP_HPP
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

using cplx = std::complex<double>;

class H_map_Single_Transmon{
    // establish H_0 as a drift hamiltonian 

    // establish H_c,j as a control hamiltonian
    // use the untiary as a control parameters

    // build transmon (use Duffing Model)
    // h{transmon, i} = w_i a^dagger_i a_i (a_i/2) a^dagger_i a^dagger_i a_i a_i
    // w_i fundamental transition frequency
    // a_i < 0 = anharmonicity (negative for transmons)
    // a_i, a^dagger_i creation operators for mode i
    // ni = a^dagger_i a_i
    // building the matrix first
    public:
        // Helper to index into row-major 1D vector
        inline size_t idx(sixe_t row, size_t col, size_t dim);
        // annihilation operator a (d x d) as a dense matrix - create matrix to calculate H_0
        std::vector<cplx> annihilation_op(size_t d);
        // matrix Multplication: C = A * B
        std::vector<cplx> matmul(const std::vector<cplx> &A, const std::vector<cplx> &B, size_t d);
        // matrix addition : C = A + B
        std::vector<cplx> matadd(const std::vector<cplx> &A, const std::vector<cplx> &B, size_t d);
        // Scalar Matrix Multiply C = s * A
        std::vector<cplx> matscale(const std::vector<cplx> &A, cplx s, size_t d);
        // Identity Matrix
        std::vector<cplx> identity(size_t d);
        // Hermintian Symmetrize (A + A†)/2
        std::vector<cplx> hermitian_sym(const std::vector<cplx> &A, size_t d);
        // build single transmon Hamiltonian Hamiltonian H = omega * n + (alpha/2) * n(n-1)
        Std::vector<cplx> build_transmons_H(double omega, double alpha, size_t d)

}

// int main() {
//    size_t d = 4;          // truncation level
//    double omega = 8.0;    // GHz or sim units
//    double alpha = -0.4;   // GHz or sim units

//    auto H = build_transmon_H(omega, alpha, d);

//    std::cout << "Single-transmon Hamiltonian (" << d << "x" << d << "):\n";
//    print_matrix(H, d);

//    return 0;
//}

// / Tensor product 