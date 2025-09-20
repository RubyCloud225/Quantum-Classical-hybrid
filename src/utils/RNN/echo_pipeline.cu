#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
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

struct EchoResult {
    double E_tot;
    std::vector<cufftDoubleComplex> E_out;
};

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
    auto sq_norm = [] __device__ (cufftDoubleComplex z) {
        return z.x*z.x + z.y*z.y;
    };
    return thrust::transform_reduce(dptr, dptr+N, sq_norm, 0.0, thrust::plus<double>());
}

// 1D convolutional Lattice (A h) [i] = \sum_{j=-r}^{r} k[j] \, h[i+j] with k
// being antisymmetric k[-j] = - \overline{k[j]}
// cayley transform becomes U = (I - A)^{-1} (I + A) D

__global__ void cayley_conv_3d(
    const cuFloatComplex* h_prev, //N_x * N_y * N_z
    const cuFloatComplex* k, // (2*r_x+1)*(2*r_y+1)*(2*r_z+1) kernel
    const cuFloatComplex* D, // Diagonal phases
    cuFloatComplex* h_next,
    int N_x, int N_y, int N_z, int r_x, int r_y, int r_z
) {
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx_x >= N_x || idx_y >= N_y || idx_z >= N_z) return;

    int voxel_idx = idx_z * (N_x * N_y) + idx_y * N_x + idx_x;
    cuFloatComplex sum = make_cuFloatComplex(0.0f, 0.0f);

    // apply convolution lattice
    for (int dz = -r_z; dz <= r_z; dz++) {
        int nz = idx_z + dz;
        if (nz < 0 || nz >= N_z) continue;
        for (int dy = -r_y; dy <= r_y; dy++) {
            int ny = idx_y + dy;
            if (ny < 0 || ny >= N_y) continue;
            for (int dx = -r_x; dx <= r_x; dx++) {
                int nx = idx_x + dx;
                if (nx < 0 || nx >= N_x) continue;
                int k_idx = (dz + r_z) * ((2 * r_y + 1) * (2 * r_x + 1)) +
                           (dy + r_y) * (2 * r_x + 1) + (dx + r_x);
                int h_idx = nz * (N_x * N_y) + ny * N_x + nx;
                sum = cuCaddf(sum, cuCmulf(k[k_idx], h_prev[h_idx]));
            }
        }
    }

    // Cayley Numerator
    cuFloatComplex num = cuCaddf(h_prev[voxel_idx], sum);
    // Denominator
    cuFloatComplex denom = cuCsubf(make_cuFloatComplex(1.0f, 0.0f), sum);
    cuFloatComplex h_temp = cuCdivf(num, denom);
    h_next[voxel_idx] = cuCmulf(h_temp, D[voxel_idx]);
}

// Apply a anti wave mirror on selected voxels
__global__ void anti_wave_mirror_3d(
    cuFloatComplex* h,
    int N_x, int N_y, int N_z,
    int mirror_start_x, int mirror_end_x,
    int mirror_start_y, int mirror_end_y,
    int mirror_start_z, int mirror_end_z) {

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_z = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx_x >= N_x || idx_y >= N_y || idx_z >= N_z) return;

    if (idx_x >= mirror_start_x && idx_x < mirror_end_x &&
        idx_y >= mirror_start_y && idx_y < mirror_end_y &&
        idx_z >= mirror_start_z && idx_z < mirror_end_z) {
            int voxel_idx = idx_z*(N_x * N_y) + idx_y * N_x + idx_x;
            h[voxel_idx].x = -h[voxel_idx].x;
            h[voxel_idx].y = -h[voxel_idx].y;
    }
}

// Entry point (called by the bindings)
extern "C"
EchoResult run_echo_pipeline(double* k_host, double* omega_host, int N, double k1, double x1, double k2, double x2, double sigma, double t) {
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
    // Allocate host vector for E_out
    std::vector<cufftDoubleComplex> E_host(N);
    // copy E(x, t) to host
    CHECK_CUDA(cudaMemcpy(E_host.data(), E, N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
    // Compute E_tot
    double E_tot = compute_Etot(E, N);
    // Free device memory
    cudaFree(d_k); cudaFree(d_omega);
    cudaFree(f1); cudaFree(f2); cudaFree(g); cudaFree(E);

    EchoResult result;
    result.E_tot = E_tot;
    result.E_out = std::move(E_host);
    return result;
}
