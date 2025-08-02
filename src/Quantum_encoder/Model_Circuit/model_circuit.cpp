#include "model_circuit.hpp"
#include <vector>
#include "utils/logger.hpp"
#include <complex>
#include <iostream>
#include <cmath>
#include <numeric>
#include <stdexcept>

const double ModelCircuit::pi = 3.14159265358979323846;

using Complex = std::complex<double>;

std::vector<std::vector<ModelCircuit::Complex>> ModelCircuit::U_theta_lambda(double theta, double lambda) {
    using Complex = ModelCircuit::Complex;
    std::vector<std::vector<Complex>> unitary_matrix = {
        {std::cos(theta / 2), -std::exp(Complex(0, lambda)) * std::sin(theta / 2)},
        {std::exp(Complex(0, 0)) * std::sin(theta / 2), std::exp(Complex(0, lambda + 0)) * std::cos(theta / 2)}
    };
    Logger::log("Created unitary matrix U(theta, lambda) with theta: " + std::to_string(theta) + ", lambda: " + std::to_string(lambda), LogLevel::INFO);
    return unitary_matrix;
}

// apply various gates to the circuit matrix
void ModelCircuit::apply_gate(std::vector<Complex>& state_vector, const std::vector<std::vector<Complex>>& gate_matrix, int target_qubit) {
    int dim = 1 << target_qubit; // Dimension of the state vector
    for (int i = 0; i < dim; ++i) {
        if (((i >> target_qubit) & 1) == 0) {
            int j = i | (1 << target_qubit);
            Complex a = state_vector[i];
            Complex b = state_vector[j];
            state_vector[i] = gate_matrix[0][0] * a + gate_matrix[0][1] * b;
            state_vector[j] = gate_matrix[1][0] * a + gate_matrix[1][1] * b;
        }
    }
    Logger::log("Applied gate to target qubit: " + std::to_string(target_qubit), LogLevel::INFO, __FILE__, __LINE__);
}

void ModelCircuit::apply_hadamard(std::vector<Complex>& state_vector, int target_qubit) {
    // Apply Hadamard gate to the target qubit
    std::vector<std::vector<Complex>> hadamard_matrix = {
        {1 / std::sqrt(2), 1 / std::sqrt(2)},
        {1 / std::sqrt(2), -1 / std::sqrt(2)}
    };
    apply_gate(state_vector, hadamard_matrix, target_qubit);
}

void ModelCircuit::apply_cnot(std::vector<Complex>& state_vector, int control_qubit, int target_qubit) {
    // Apply CNOT gate with control and target qubits
    std::vector<std::vector<Complex>> cnot_matrix = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 0, 1},
        {0, 0, 1, 0}
    };
    // CNOT gate matrix for two qubits
    // CNOT gate flips the target qubit if the control qubit is in state |1>
    Logger::log("Applying CNOT gate with control qubit: " + std::to_string(control_qubit) + ", target qubit: " + std::to_string(target_qubit), LogLevel::INFO, __FILE__, __LINE__);
    apply_gate(state_vector, cnot_matrix, target_qubit);
}

// Function to initialize the circuit matrix with a single qubit in the |0> state
std::vector<ModelCircuit::Complex> ModelCircuit::apply_unitary_to_encoded_state(const std::vector<Complex>& phi_x, const std::vector<double>& theta, const std::vector<double>& lambda) {
    // Apply a unitary transformation to the encoded state
    int num_qubits = std::log2(phi_x.size());
    std::vector<Complex> state_vector = phi_x; // Start with the encoded state
    // parameters for the rotation gates
    for (int i = 0; i < num_qubits; ++i) {
        auto U = U_theta_lambda(theta[i], lambda[i]);
        apply_gate(state_vector, U, i); // Apply the unitary gate to the i-th qubit
    }
    // Apply Hadamard gates to the first qubit
    for (int i = 0; i < num_qubits; ++i) {
        apply_hadamard(state_vector, i);
    }
    // Apply CNOT gates between the first qubit and all other qubits
    for (int i = 1; i < num_qubits; ++i) {
        apply_cnot(state_vector, i, (i + 1) % num_qubits);
    }
    Logger::log("Applied unitary transformation to the encoded state with " + std::to_string(num_qubits) + " qubits", LogLevel::INFO, __FILE__, __LINE__);
    return state_vector; // Return the transformed state vector

}

// measurement
double measure_overlap_with_zero(const std::vector<Complex>& state_vector) {
    std::vector<Complex> zero_state = {Complex(1, 0), Complex(0, 0)};
    Complex init = Complex(0, 0);
    auto result = std::inner_product(
        state_vector.begin(),
        state_vector.end(),
        zero_state.begin(),
        init,
        std::plus<Complex>(),
        std::multiplies<Complex>()
    );
    Logger::log("Measured overlap with zero state: " + std::to_string(std::norm(result)), LogLevel::INFO, __FILE__, __LINE__);
    return std::norm(result);
}

double measure_overlap_with_one(const std::vector<Complex>& state_vector) {
    std::vector<Complex> one_state = {Complex(0, 0), Complex(1, 0)};
    Complex init = Complex(0, 0);
    auto result = std::inner_product(
        state_vector.begin(),
        state_vector.end(),
        one_state.begin(),
        init,
        std::plus<Complex>(),
        std::multiplies<Complex>()
    );
    Logger::log("Measured overlap with one state: " + std::to_string(std::norm(result)), LogLevel::INFO, __FILE__, __LINE__);
    return std::norm(result);
}

double measure_projection_onto_zero(const std::vector<Complex>& state_vector) {
    // Measure the projection of the state vector onto the zero state
    double overlap = measure_overlap_with_zero(state_vector);
    Logger::log("Measured projection onto zero state: " + std::to_string(overlap),LogLevel::INFO, __FILE__, __LINE__);
    return overlap; // Return the probability of measuring zero
}

double measure_projection_onto_one(const std::vector<Complex>& state_vector) {
    // Measure the projection of the state vector onto the one state
    double overlap = measure_overlap_with_one(state_vector);
    Logger::log("Measured projection onto one state: " + std::to_string(overlap), LogLevel::INFO, __FILE__, __LINE__);
    return overlap; // Return the probability of measuring one
}

// Quantum circuit that encodes a parameterized graph state
//  takes classical data applies a single qubit unitary gate - theta and lambda
// applies a hadamard gate to each qubit
// applies a cnot gate between each pair of qubits
// and returns the final state vector
//
// Hybrid encoding: apply a sequence of parameterized rotation gates to each qubit
std::vector<ModelCircuit::Complex> ModelCircuit::apply_hybrid_encoding(const std::vector<HybridGate>& hybridGates) {
    int num_qubits = hybridGates.size();
    std::vector<Complex> state_vector(1 << num_qubits, Complex(0.0, 0.0));
    state_vector[0] = Complex(1.0, 0.0);  // Start in |0...0>

    for (int i = 0; i < num_qubits; ++i) {
        const auto& gate = hybridGates[i].angleGate;
        double amplitude = hybridGates[i].amplitude;
        double theta = gate.angle * amplitude;

        std::vector<std::vector<Complex>> rotation_matrix;

        switch (gate.type) {
            case RotationAxis::X:
                rotation_matrix = {
                    {std::cos(theta / 2), Complex(0, -std::sin(theta / 2))},
                    {Complex(0, -std::sin(theta / 2)), std::cos(theta / 2)}
                };
                break;
            case RotationAxis::Y:
                rotation_matrix = {
                    {std::cos(theta / 2), -std::sin(theta / 2)},
                    {std::sin(theta / 2), std::cos(theta / 2)}
                };
                break;
            case RotationAxis::Z:
                rotation_matrix = {
                    {std::exp(Complex(0, -theta / 2)), Complex(0, 0)},
                    {Complex(0, 0), std::exp(Complex(0, theta / 2))}
                };
                break;
        }

        apply_gate(state_vector, rotation_matrix, i);
    }

    Logger::log("Applied hybrid encoding gates to state.", LogLevel::INFO, __FILE__, __LINE__);
    return state_vector;
}