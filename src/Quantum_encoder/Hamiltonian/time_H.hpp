#ifndef TIME_H_HPP
#define TIME_H_HPP

#include <vector>
#include <limits>
#include <complex>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using cplx = std::complex<double>;

namespace Time {
    // Minimal complex matrix exponentiation for Hermitian matrices
    struct Mat {
        int nrows{0}, ncols{0};
        std::vector<cplx> a; // row major

        Mat() = default;
        Mat(int rows, int cols, cplx fill=cplx(0.0,0.0));
        cplx& operator()(int rows, int cols);
        const cplx& operator()(int rows, int cols) const;
        static Mat identity(int n);
    };

    struct Vec {
        std::vector<cplx> data;
        Vec() = default;
        explicit Vec(int n, cplx fill=cplx(0.0,0.0));
        int size() const;
        cplx& operator[](int i);
        const cplx& operator[](int i) const;
    };

    struct EigResult {
        std::vector<double> lambda; // real
        Mat V; // unitary matrix
    };

    Vec toVec(const std::vector<cplx>& vec);
    Mat adjoint(const Mat& M);
    Mat hermitianize(const Mat& A);
    double frobenius_norm(const Mat& A);
    EigResult jacobiHermitian(const Mat& H_in, int maxSweeps=100, double tol = 1e-12);
    Mat spectralPropagator(const Mat& H_in, double dt);
    Mat composeHamiltonian(const Mat& H0, const std::vector<Mat>& HX, const std::vector<Mat>& HZ, const std::vector<double>& uX, const std::vector<double>& uZ);
    void applyTimeStep(Vec& psi, const Mat& U);
    void stepControlled(Vec& psi, const Mat& H0, const std::vector<Mat>& HX, const std::vector<Mat>& HZ, const std::vector<double>& uX, const std::vector<double>& uZ, double dt);
    double unitaryError(const Mat& U);
    Mat identity(int d);
    Mat zero(int d);
}

#endif // TIME_H_HPP
