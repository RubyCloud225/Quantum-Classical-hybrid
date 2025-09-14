#include <iostream>
#include <vector>
#include <cmath>
#include <cufft.h>

extern "C" void run_echo_pipeline(double* k_host, double* omega_host, int N, double k1, double x1, double k2, double x2, double sigma, double t, cufftDoubleComplex* E_out);

int main() {
    int N = 1024;
    double L = 100.0;
    std::vector<double> k(N), omega(N);
    // build momentum grid
    for (int i = 0; i < N; i++) {
        double n = i -N/2;
        k[i] = 2.0 * M_PI * n / L;
        omega[i] = std::sqrt(k[i] * k[i] + 1e-6);
    }
    std::vector<cufftDoubleComplex> E_out(N);
    // run pipeline
    run_echo_pipeline(k.data(), omega.data(), N, 10.0, -10.0, 12.0, 10.0, 2.0, 8.0, E_out.data()); // k1, x1, k2, x2, sigma, t
    // print output
    std::cout << "First 5 values of E(x,t):\n";
    for (int i=0; i<5; i++) {
        std::cout << E_out[i].x << " + " << E_out[i].y << "i\n";
    }
    return 0;
}