#include "quantum_compressor.hpp"
#include <cmath>
#include <iostream>
#include <eigen/Dense>

using Complex = std::complex<double>;
using Matrix2cd = Eigen::Matrix2cd;

namespace Clifford {
    Matrix2cd H() {
        Matrix2cd m;
        m << 1, 1,
                1, -1;
        return m / std::sqrt(2);
    }
    Matrix2cd Z() {
        Matrix2cd m;
        m << 1, 0,
                0, -1;
        return m;
    }
}

// Simplify the state vector using Clifford Algorithm
std::vector<Complex> QuantumCompressor::simplify_clifford(const std::vector<Complex>& state_vector) {
    int dim = state_vector.size();
    int num_qubits = std::log2(dim);
    std::vector<Complex> new_state = state_vector;
    for (int q = 0; q < num_qubits; ++q) {
        Matrix2cd H = Clifford::H();
        Matrix2cd Z = Clifford::Z();
        Matrix2cd HZH = H * Z * H;

        for (int i = 0; i < dim; ++i) {
            if (((i >> q) & 1) == 0) {
                int j = i | (1 << q);
                Complex a = new_state[i];
                Complex b = new_state[j];
                new_state[i] = HZH(0, 0) * a + HZH(0, 1) * b;
                new_state[j] = HZH(1, 0) * a + HZH(1, 1) * b;
            }
        }
    }
    return new_state;
}

// Contract low-importance qubits from tensor product
std::vector<Complex> QuantumCompressor::contract_low_importance_qubits(const std::vector<Complex>& state_vector, const std::vector<int>& low_importance_qubits) {
    // Placeholder for contraction logic
    // This function would typically contract the specified low-importance qubits
    std::vector<Complex> contracted_vector = state_vector; // For now, just return the input
    return contracted_vector;
}

// Simulate measurement on a specific qubit and collapse the state vector
std::vector<Complex> QuantumCompressor::simulate_measurement(const std::vector<Complex>& state_vector, int qubit_index, int measurement_result) {
    int dim = state_vector.size();
    std::vector<Complex> collapsed_state(dim, Complex(0, 0));
    double norm = 0.0;
    for (int i = 0; i < dim; ++i) {
        bool bit = (i >> qubit_index) & 1;
        if (bit == measurement_result) {
            collapsed_state[i] = state_vector[i];
            norm += std::norm(state_vector[i]);
        }
    }
    double scale = 1.0 / std::sqrt(norm);
    for (int i = 0; i < dim; ++i) {
        collapsed_state[i] *= scale;
    }
    return collapsed_state;
}

// Utility: Calculate the probability of measurement for a specific qubit
double QuantumCompressor::calculate_measurement_probability(const std::vector<Complex>& state_vector, int qubit_index, int measurement_result) {
    double probability = 0.0;
    int dim = state_vector.size();
    for (int i = 0; i < dim; ++i) {
        bool bit = (i >> qubit_index) & 1;
        if (bit == measurement_result) {
            probability += std::norm(state_vector[i]);
        }
    }
    return probability;
}