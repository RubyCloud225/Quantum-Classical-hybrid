#ifndef TIME_H_HPP
#define TIME_H_HPP

#include <vector>
#include <limits>
#include <complex>
#include <stdexcept>
#include <cmath>
#include <algorithm>

using cplx = std::complex<double>;

class Time {
    public:
    struct Mat {
        // Minimal complex matrix exponentiation for Hermitian matrices
        struct Mat;
        struct Vec;
        inline Mat adjoint(const Mat& M);
        inline Mat Hermianize(const Mat& A);
        inline Mat operator*(const Mat& A, const Mat& B);
        inline Vec operator*(const Mat& A, const Vec& x);
        inline Mat operator*(const cplx& s, const Mat& A);
        inline Mat operator*(const Mat& A, const cplx& s);
        inline double frobenius_norm(const Mat& A);
        struct EigResult;
        inline EigResult jacobiHermitian(const Mat& H_in);
        inline Mat spectralPropagator(const Mat& H_in, double dt);
        inline Mat composeHamiltonian(const Mat& H0, const std::vector<Mat>& HX, const std::vector<Mat>& HZ, const std::vector<double>& uX, const std::vector<double>& uZ);
        inline void applyTimeStep(Vec& psi, const Mat& U);
        inline void stepControlled(Vec& psi, const Mat& H0, const std::vector<Mat>& HX, const std::vector<Mat>& HZ, const std::vector<double>& uX, const std::vector<double>& uZ, double dt);
        inline double unitaryError(const Mat& U);

    }
}

#endif // TIME_H_HPP
