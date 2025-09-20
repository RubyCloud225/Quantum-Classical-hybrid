#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <iostream>

using complxed = thrust::complex<double>;
// ----------------------------------------------------------------------
// CUDA Kernels
// ----------------------------------------------------------------------

// Apply momentum operator elementwise: psi_p -> p * psi_p
__global__ void apply_momentum_operator(complexd* psi_p, double* p_vals, complexd* out, int N) {
    int idx = blockIdx.x blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = p_vals[idx] * psi_p[idx];
    }
} 

// apply projector mask elementwise: psi_proj = mask * psi
__global__ void apply_projector(complexd* psi, double* mask, complexd* psi_proj, int N) {
    int idx = blockId.x * blockDim.x + threadIdx.x;;
    if (idx < N) {
        psi_proj[idx] = mask[idx] * psi[idx];
    }
}

// compute dot product <a|b> (partial sum reduction)
__global__ void dot_product_kernel(complexd* a, complexd* b, double* result, int N) {
    __shared__ double temp[1024];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (idx < N) {
        val = thrust::abs(a[idx] * thrust::conj(b[idx]));
    }
    temp[threadIdx.x] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x <stride) temp[threadIdx.x] += temp[threadIdx.x + stride];
    }
    if (threadIdx.x == 0) atomicAdd(result, temp[0]);
}
// compute cross term 2*RE(<A|B>)
__global__ void cross_term_kernel(complexd* a, complexd* b, double* result, int N) {
    __shared__ double temp[1024];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (idx < N) {
        val = 2.0 * thrust::real(thrust::conj(a[idx]) * b[idx]);
    }
    temp[threadIdx.x] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
        if (threadIdx.x < stride) temp[threadIdx.x] += temp[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(result, temp[0]);
}

// Main momentum Space Computation

int main() {
    const int N = 1024;
    complexd* h_psi = new complexd[N];
    double* h_p = new double[N];
    double* h_Pmask = new double[N]; // kept projector
    double* h_Qmask = new double[N]; // Vacuum projector

    // Initialize state momentum projectors
    for (int i = 0; i < N; i++) {
        h_psi[i] = complxd(1.0, 0.0); // TODO implement this with a state vector
        h_p[i] = i; // Example momentum values
        h_Pmask[i] = (i < N/2) ? 1.0 : 0.0; // First half kept
        h_Qmask[i] = 1.0 - h_Pmask[i]; // second half vacuum
    }

    // Device allocations
    complexd *d_psi, *d_psi_p, *d_temp, *d_psi_P, *d_psi_Q;
    double *d_p, *d_Pmask, *d_Qmask, *d_result;
    cudaMalloc(&d_psi, N * sizeof(complexd));
    cudaMalloc(&d_psi_p, N * sizeof(complexd));
    cudaMalloc(&d_temp, N * sizeof(complexd));
    cudaMalloc(&d_psi_P, N * sizeof(complexd));
    cudaMalloc(&d_psi_Q, N * sizeof(complexd));
    cudaMalloc(&d_p, N * sizeof(double));
    cudaMalloc(&d_Pmask, N * sizeof(double));
    cudaMalloc(&d_Qmask, N * sizeof(double));
    cudaMalloc(&d_result, sizeof(double));

    cudaMemcpy(d_psi, h_psi, N * sizeof(complexd), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Pmask, h_Pmask, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Qmask, h_Qmask, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(double));

    // FFT Plan
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex*>(d_psi),
                 reinterpret_cast<cufftDoubleComplex*>(d_psi_p), CUFFT_FORWARD);

    int blockSize = 1024;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Apply momentum operator
    apply_momentum_operator<<<numBlocks, blockSize>>>(d_psi_p, d_p, d_temp, N);

    // Apply projectors
    apply_projector<<<numBlocks, blockSize>>>(d_psi_p, d_Pmask, d_psi_P, N);
    apply_projector<<<numBlocks, blockSize>>>(d_psi_p, d_Qmask, d_psi_Q, N);

    // Compute W_p components
    cudaMemset(d_result, 0, sizeof(double));
    dot_product_kernel<<<numBlocks, blockSize>>>(d_psi_P, d_psi_P, d_result, N);
    double W_p_kept; cudaMemcpy(&W_p_kept, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemset(d_result, 0, sizeof(double));
    dot_product_kernel<<<numBlocks, blockSize>>>(d_psi_Q, d_psi_Q, d_result, N);
    double W_p_vac; cudaMemcpy(&W_p_vac, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemset(d_result, 0, sizeof(double));
    cross_term_kernel<<<numBlocks, blockSize>>>(d_psi_P, d_psi_Q, d_result, N);
    double W_p_cross; cudaMemcpy(&W_p_cross, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    double W_p_total = W_p_kept + W_p_vac + W_p_cross;

    std::cout << "W_p_kept = " << W_p_kept << std::endl;
    std::cout << "W_p_vac = " << W_p_vac << std::endl;
    std::cout << "W_p_cross = " << W_p_cross << std::endl;
    std::cout << "W_p_total = " << W_p_total << std::endl;

    // -----------------------------------------------
    // Placeholder for Projection Correction Σ(E)
    // TODO: Solve (E - QHQ)x = QHP u_j and construct Σ(E)
    //       then use H_eff(E) = PHP + Σ(E) for compressed evolution
    // -----------------------------------------------

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_psi); cudaFree(d_psi_p); cudaFree(d_temp);
    cudaFree(d_psi_P); cudaFree(d_psi_Q);
    cudaFree(d_p); cudaFree(d_Pmask); cudaFree(d_Qmask);
    cudaFree(d_result);
    delete[] h_psi; delete[] h_p; delete[] h_Pmask; delete[] h_Qmask;
}