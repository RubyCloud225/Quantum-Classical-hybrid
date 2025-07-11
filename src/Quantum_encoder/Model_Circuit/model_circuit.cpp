#include "model_circuit.hpp"
#include <vector>
#include <complex>
#include <iostream>
#include <cmath>

ModelCircuit::ModelCircuit() {
    // Initialize the circuit matrix with a single qubit in the |0> state
    initialize_circuit_matrix(1);
}

const double ModelCircuit::pi;
using Complex = std::complex<double>;

// Function to create a unitary matrix U(theta, lambda) for a single qubit rotation
// gate
// U(\theta, \phi, \lambda) = \begin{bmatrix} \cos(\theta/2) & -e^{i\lambda} \sin(\theta/2) \\\e^{i\phi} \sin(\theta/2) & e^{i(\lambda + \phi)} \cos(\theta/2) \end{bmatrix}
std::vector<std::Complex> U_theta_lambda(double theta, double lambda) {
    // Create the unitary matrix U(theta, lambda)
    std::vector<std::vector<Complex>> unitary_matrix = {
        {std::cos(theta / 2), -std::exp(Complex(0, lambda)) * std::sin(theta / 2)},
        {std::exp(Complex(0, 0)) * std::sin(theta / 2), std::exp(Complex(0, lambda + 0)) * std::cos(theta / 2)}
    }; //exp(Complex(0, 0)) is used to represent e^(i*0) which is 1.
}


// apply various gates to the circuit matrix

void ModelCircuit::apply_gate(std::vector<Complex>& state_vector, const std::vector<std::vector<Complex>>& gate_matrix, int target_qubit) {
    // qubit gate to the input state vector
    int dim = 1 << target_qubit; // Dimension of the state vector
    for (int i = 0; i < dim; ++i) {
        if (((i >> target_qubit) & 1) == 0) {
            int j = i | (1 << target_qubit);
            Complex a = state[i];
            Complex b = state[j];
            state_vector [i] = gate_matrix[0][0] * a + gate_matrix[0][1] * b;
            state_vector [j] = gate_matrix[1][0] * a + gate_matrix[1][1] * b;
        }
    }
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
        apply_cnot(state_vector, i, (i + 1) % num_qubits, i);
    }
    return state_vector; // Return the transformed state vector

}

// measurement
double measure_overlap_with_zero(const std::vector<Complex>& state_vector) {
    // Measure the overlap between the state vector and the zero state
    std::vector<Complex> zero_state = {1, 0}; // |0> state
    return std::norm(std::inner_product(state_vector.begin(), state_vector.end(), zero_state.begin(), Complex(0, 0)));
}

double measure_overlap_with_one(const std::vector<Complex>& state_vector) {
    // Measure the overlap between the state vector and the one state
    std::vector<Complex> one_state = {0, 1}; // |1> state
    return std::norm(std::inner_product(state_vector.begin(), state_vector.end(), one_state.begin(), Complex(0, 0)));
}

double measure_projection_onto_zero(const std::vector<Complex>& state_vector) {
    // Measure the projection of the state vector onto the zero state
    double overlap = measure_overlap_with_zero(state_vector);
    return overlap; // Return the probability of measuring zero
}

double measure_projection_onto_one(const std::vector<Complex>& state_vector) {
    // Measure the projection of the state vector onto the one state
    double overlap = measure_overlap_with_one(state_vector);
    return overlap; // Return the probability of measuring one
}