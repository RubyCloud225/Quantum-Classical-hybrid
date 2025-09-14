#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cmath>
#include <complex>
#include <iostream>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}
#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        std::cerr << "CUFFT error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << err << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel: Build Gaussian Packet F_j(k)

__global__ void gaussian_wavepacket(cufftDoubleComplex* f, const double* k, double k0, double x0, double sigma, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double arg = -((k[i] - k0) * (k[i] - k0)) / (4.0 * sigma * sigma);
        double phase = k[i] * x0;
        double pref = pow(1.0 / (2.0 * M_PI * sigma * sigma), 0.25);
        double gauss = pref * exp(arg);
        f[i].x = gauss * cos(phase);
        f[i].y = gauss * sin(phase);
    }
}

// Compute Overlap g(k,t)
__global__ void compute_overlap(cufftDoubleComplex* g, const cufftDoubleComplex* f, const cufftDoubleComplex* psi, const double* omega, double t, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double real_part = f[i].x * psi[i].x + f[i].y * psi[i].y;
        double imag_part = f[i].y * psi[i].x - f[i].x * psi[i].y; // conjugate multiplication
        double phase = -omega[i]*t;
        double norm = rsqrt(4.0 * M_PI * omega[i] + 1e-12);
        double cos_phase = cos(phase);
        double sin_phase = sin(phase);
        g[i].x = norm * (real_part * cos_phase - imag_part * sin_phase);
        g[i].y = norm * (real_part * sin_phase + imag_part * cos_phase);
    }
}

// Reduction compute E_tot(t) = ∫ dx |E(x,t) |^2

double compute_Etot(const cufftDoubleComplex* E, int N) {
    thrust::device_ptr<const cufftDoubleComplex> dptr(E);
    auto sq_norm = [=] __device__(cufftDoubleComplex z) {
        return z.x*z.x + z.y*z.y;
    };
    return thrust::transform_reduce(dptr, dptr+N, sq_norm, 0.0, thrust::plus<double>());
}

// Entry point (called by the bindings)
extern "C"
void run_echo_pipeline(double* k_host, double* omega_host, int N, double k1, double x1, double k2, double x2, double sigma, double t, cufftDoubleComplex* E_out) {
    double *d_k, *d_omega;
    cufftDoubleComplex *f1, *f2, *g, *E;
    CHECK_CUDA(cudaMalloc(&d_k, N*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_omega, N*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&f1, N*sizeof(cufftDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&f2, N*sizeof(cufftDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&g, N*sizeof(cufftDoubleComplex)));
    CHECK_CUDA(cudaMalloc(&E, N*sizeof(cufftDoubleComplex)));

    // copy K, omega
    CHECK_CUDA(cudaMemcpy(d_k, k_host, N*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_omega, omega_host, N*sizeof(double), cudaMemcpyHostToDevice));
    // Launch Gaussian Builders
    int threads = 256;
    int blocks = (N + threads -1) / threads;
    gaussian_wavepacket<<<blocks, threads>>>(f1, d_k, k1, x1, sigma, N);
    gaussian_wavepacket<<<blocks, threads>>>(f2, d_k, k2, x2, sigma, N);
    // compute g(k, t)
    compute_overlap<<<blocks, threads>>>(g, f1, f2, d_omega, t, N);
    // FFT g(k,t) -> E(x, t)
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));
    CHECK_CUFFT(cufftExecZ2Z(plan, g, E, CUFFT_INVERSE));
    CHECK_CUFFT(cufftDestroy(plan));
    // copy E(x, t) to host
    CHECK_CUDA(cudaMemcpy(E_out, E, N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
    // Compute E_tot
    double E_tot = compute_Etot(E, N);
    // Free
    cudaFree(d_k); cudaFree(d_omega);
    cudaFree(f1); cudaFree(f2); cudaFree(g); cudaFree(E);
}