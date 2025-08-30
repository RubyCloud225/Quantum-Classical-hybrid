// Compute U_k = exp(-i H dt) for Hermitian H using Jacobi diagonalization
//
// Equations:
// H^(k) = H0 + sum_i [ u_X,i^(k) H_X,i + u_Z,i^(k) H_Z,i ]
// Jacobi rotations(spectral decomposition): H = V Λ V†
// propagator: U = V diag( e^{-i λ_m dt} ) V†
// complex matrix exponentiation: exp(-i λ_m dt) = cos(λ_m dt) - i sin(λ_m dt)
// hermitian matrix: H = H† that implies real eigenvalues λ_m
// convenience functions to compute the propagator U_k
// Unitarity diagnostic || U U† - I ||_F = 0
// 
// No external dependencies except for the C++ standard library
#include "time_H.hpp"
#include <cmath>
#include <complex>
#include <vector>
#include <stdexcept>
#include <limits>
#include <algorithm>

namespace Time {
    using cplx = std::complex<double>;
    Time::Vec Time::toVec(const std::vector<cplx>& vec) {
        Time::Vec result;
        result.data = vec; // Assuming Vec has a `data` member
        return result;
    };

    // Minimal complex matrix exponentiation for Hermitian matrices
    struct Mat {
        int nrows{0}, ncols{0};
        std::vector<cplx> a; // row major

        Mat() = default;
        Mat(int rows, int cols, cplx fill=cplx(0.0,0.0)) : nrows(rows), ncols(cols), a(static_cast<size_t>(rows*cols), fill) {
            if (rows <= 0 || cols <= 0) {
                throw std::invalid_argument("Matrix dimensions must be positive");
            }
        }
        inline cplx& operator()(int rows, int cols) {
            return a[static_cast<size_t>(rows)*ncols + static_cast<size_t>(cols)];
        }
        inline const cplx& operator()(int rows, int cols) const {
            return a[static_cast<size_t>(rows)*ncols + static_cast<size_t>(cols)];
        }
        static Mat identity(int n) {
            Mat I(n,n);
            for (int i = 0; i < n; ++i) I(i, i) = cplx(1.0, 0.0);
            return I;
        }
    };

    struct Vec {
        std::vector<cplx> v;
        Vec() = default;
        explicit Vec(int n, cplx fill=cplx(0.0,0.0)) : v(static_cast<size_t>(n), fill) {}
        inline int size() const {
            return static_cast<int>(v.size());
        }
        inline cplx& operator[](int i) {
            return v[static_cast<size_t>(i)];
        }
        inline const cplx& operator[](int i) const {
            return v[static_cast<size_t>(i)];
        }
    };

    // basic ops
    inline Mat adjoint(const Mat& M) {
        Mat R(M.ncols, M.nrows);
        for (int r=0; r < M.nrows; ++r) {
            for (int c=0; c < M.ncols; ++c) {
                R(c, r) = std::conj(M(r, c));
            }
        }
        return R;
    }

    inline Mat hermitianize(const Mat& A) {
        if (A.nrows != A.ncols) throw std::invalid_argument("Matrix must be square for hermitianization");
        Mat R = A;
        for (int i = 0; i < A.nrows; ++i) {
            // diagonal elements to real average
            R(i, i) = cplx((A(i, i).real() + A(i, i).imag()) / 2.0, 0.0);
            for (int j=i+1; j < A.ncols; ++j) {
                // off-diagonal elements to average
                cplx avg = (A(i, j) + std::conj(A(j, i))) / 2.0;
                R(i, j) = avg;
                R(j, i) = std::conj(avg);
            }
        }
        return R;
    }

    inline Mat operator*(const Mat& A, const Mat& B) {
        if (A.ncols!= B.nrows) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication");
        }
        Mat R(A.nrows, B.ncols);
        for (int i=0; i < A.nrows; ++i) {
            for (int k=0; k < A.ncols; ++k) {
                cplx aik = A(i, k);
                if (aik == cplx(0.0, 0.0)) continue; // skip zero elements
                for (int j=0; j < B.ncols; ++j) {
                    R(i, j) += aik * B(k, j);
                }
            }
        }
        return R;
    }

    inline Vec operator*(const Mat& A, const Vec& x) {
        if (A.ncols!= x.size()) {
            throw std::invalid_argument("Matrix and vector dimensions do not match for multiplication");
        }
        Vec y(A.nrows);
        for (int i = 0; i < A.nrows; ++i) {
            cplx s(0.0, 0.0);
            for (int j = 0; j < A.ncols; ++ j) s+= A(i,j)*x.v[static_cast<size_t>(j)];
            y.v[static_cast<size_t>(i)] = s;
        }
        return y;
    }

    inline Mat operator*(const cplx& s, const Mat& A) {
        Mat R(A.nrows, A.ncols);
        for (size_t k = 0; k < A.a.size(); ++k) R.a[k] = s * A.a[k];
        return R;
    }

    inline Mat operator*(const Mat& A, const cplx& s) {
        return s * A; // scalar multiplication is commutative
    }

    inline double frobenius_norm(const Mat& A) {
        double ss = 0.0;
        for (const auto& z : A.a) {
            ss += std::norm(z);
        }
        return std::sqrt(ss);
    }

    // -------- Jacobi diagonalization for Hermitian matrices --------
    // Return eigenvalues (real) and eigenvectors (unitary matrix) of a Hermitian matrix
    struct EigResult {
        std::vector<double> lambda; // real
        Mat V; // unitary matrix
    };

    inline EigResult jacobiHermitian(Mat H, int maxSweeps=100, double tol = 1e-12) {
        if (H.ncols != H.nrows) {
            throw std::invalid_argument("Matrix must be square for Jacobi diagonalization");
        }
        const int n = H.nrows;
        // exact Hermiticity
        H = hermitianize(H);
        // initialize eigenvalues and eigenvectors
        Mat V = Mat::identity(n);
        auto absOffdiagMax = [&](int& p, int& q){
            double m=0.0; p = 0; q = 1;
            for (int i =0; i < n; ++i){
                for (int j = i + 1; j < n; ++j) {
                    double absVal = std::abs(H(i, j));
                    if (absVal > m) {
                        m = absVal;
                        p = i;
                        q = j;
                    }
                }
            }
            return m;
        };

        for (int sweep=0; sweep <maxSweeps; ++sweep) {
            int p=0, q=1; double maxOff = absOffdiagMax(p,q);
            if (maxOff < tol) {
                // off-diagonal elements are small enough, stop
                break;
            }
            cplx hpq = H(p,q);
            if (std::abs(hpq) < tol) continue; // skip zero elements
            double app = H(p, p).real();
            double aqq = H(q, q).real();
            double abs_hpq = std::abs(hpq);
            // make H(p,q) real and >=0
            cplx phase = (abs_hpq > 0.0) ? (hpq / abs_hpq) : cplx(1.0, 0.0);
            cplx e_minus_i_alpha = std::conj(phase); // e^{-ia}
            cplx e_plus_i_alpha = phase; // e^{+ia}
            // compute the Jacobi rotation in 2x2 block
            double tau = (aqq - app);
            double t;
            if (std::abs(tau) < 1e-300) t = 1.0; else {
                double tau_over_2 = tau / (2.0 * abs_hpq);
                t = (tau_over_2 >= 0.0) ? 1.0/(tau_over_2 + std::sqrt(1.0 + tau_over_2 * tau_over_2)) : 1.0/(tau_over_2 - std::sqrt(1.0 + tau_over_2 * tau_over_2));
            }
            double c = 1.0 / std::sqrt(1.0 + t * t);
            double s = t * c; // s = sin(theta), c = cos(theta)

            // Apply complex jacobi rotation to H: U* H U†
            // Where U acts in (p,q) plane with phase a and angle theta
            // Update rows/cols p,q
            for (int k = 0; k < n; ++k) {
                if (k == p || k == q) continue; // skip p,q rows/cols
                cplx Hkp = H(k,p);
                cplx Hkq = H(k,q);
                // rotate columns with phase
                cplx col_p = c * Hkp - s * (Hkq * e_minus_i_alpha);
                cplx col_q = s * (Hkp * e_plus_i_alpha) + c * Hkq;
                H(k,p) = col_p;
                H(p,k) = std::conj(col_p); // hermitian
                H(k,q) = col_q;
                H(q,k) = std::conj(col_q); // hermitian
            }
            //update the diagonal block
            double app_new = c * c * app - 2.0 * s * c * hpq.real() + s * s * aqq;
            double aqq_new = s * s * app + 2.0 * s * c * hpq.real() + c * c * aqq;
            H(p,p) = cplx(app_new, 0.0);
            H(q,q) = cplx(aqq_new, 0.0);
            H(p,q) = cplx(0.0, 0.0); // off-diagonal becomes zero
            H(q,p) = cplx(0.0, 0.0); // off-diagonal becomes zero
            // update the eigenvector matrix V
            for (int k=0; k<n; ++k) {
                cplx vkp = V(k, p);
                cplx vkq = V(k, q);
                cplx new_p = c * vkp - s * (vkq * e_minus_i_alpha);
                cplx new_q = s * (vkp * e_plus_i_alpha) + c * vkq;
                V(k, p) = new_p;
                V(k, q) = new_q;
            }
        }
        // extract eigenvalues from diagonal
        std::vector<double> lambda(static_cast<size_t>(n));
        for (int i = 0; i<n; ++i) lambda[static_cast<size_t>(i)] = H(i, i).real(); // real eigenvalues
        // return eigenvalues and eigenvectors
        return EigResult{std::move(lambda), std::move(V)};
    }

    // --------- Spectral Propagator -------------

    Mat spectralPropagator(const Mat& H_in, double dt) {
        if (H_in.nrows != H_in.ncols) {
            throw std::invalid_argument("Matrix must be square for spectral propagator");
        }
        // Diagonalize the Hermitian matrix H
        Mat H = hermitianize(H_in);
        EigResult er = jacobiHermitian(H);
        const int n = H.nrows;
        // Build D = diag(exp(-i λ_m dt))
        Mat D(n, n);
        const cplx im(0.0,1.0);
        for (int i = 0; i< n; ++i) {
            double ang = -er.lambda[static_cast<size_t>(i)] * dt;
            D(i,i) = std::exp(im * ang); // e^{-i λ_m dt} 
        }
        Mat U = er.V * D * adjoint(er.V); // U = V D V†
        return U;
    }

    //----------- VQE control composition and step -------------

    inline Mat composeHamiltonian(const Mat& H0, const std::vector<Mat>& HX, const std::vector<Mat>& HZ, const std::vector<double>& uX, const std::vector<double>& uZ) {
        if (HX.size() != uX.size()) {
            throw std::invalid_argument("HX and uX must have the same size");
        }
        if (HZ.size() != uZ.size()) {
            throw std::invalid_argument("HZ and uZ must have the same size");
        }
        Mat H = H0; // start with H0
        for (size_t i=0; i<HX.size(); ++i) {
            H = H + (cplx(uX[i], 0.0) * HX[i]);
        }
        for (size_t j=0; j < HZ.size(); ++j) {
            H = H + (cplx(0.0, uZ[j]) * HZ[j]); // imaginary part for Z terms
        }
        return hermitianize(H); // ensure Hermitian
    }

    inline void applyTimeStep(Vec& psi, const Mat& U) {
        psi = U * psi;
    }

    inline void stepControlled(Vec& psi, const Mat& H0, const std::vector<Mat>& HX, const std::vector<Mat>& HZ, const std::vector<double>& uX, const std::vector<double>& uZ, double dt) {
        // compose the hamiltonian
        Mat Hk = composeHamiltonian(H0,HX,HZ,uX,uZ);
        // compose the spectral propagator
        Mat Uk = spectralPropagator(Hk, dt);
        // apply timestep 
        applyTimeStep(psi, Uk);
    }

    inline double unitaryError(const Mat& U) {
        // Compute || U U† - I ||_F
        if (U.nrows != U.ncols) {
            throw std::invalid_argument("Matrix must be square for unitarity error");
        }
        Mat I = Mat::identity(U.nrows);
        Mat UUdag = U * adjoint(U);
        for (int i = 0; i < I.nrows; ++i) {
            UUdag(i,i) -= cplx(1.0, 0.0); // subtract identity
        }
        return frobenius_norm(UUdag); // return Frobenius norm of the error
    }
}

// int N = 4; // example size
// Time::Mat H0(N, N, Time::cplx(0.0, 0.0)); // example Hamiltonian
// std::vector<Time::Mat> HX, HZ; // example interaction terms
// std::vector<double> uX, uZ; // example coefficients
// ... fill H0, HX[k], HZ[k], uX[k], uZ[k] ... for the time slice
// Time::Vec psi(N);
// psi[0] = Time::cplx(1.0, 0.0); // initial state
// double dt = 0.01; // time step
// Time::Mat HK = Time::composeHamiltonian(H0, HX, HZ, uX, uZ);
// Time::Mat UK = Time::spectralPropagator(HK, dt);
// double err = Time::unitaryError(UK); // check unitarity
// psi = UK * psi; // advance state