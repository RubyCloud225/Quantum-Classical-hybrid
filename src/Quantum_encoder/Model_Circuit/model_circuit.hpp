#ifndef MODEL_CIRCUIT_HPP
#define MODEL_CIRCUIT_HPP

#include <vector>
#include "../../Quantum_encoder/converttoqubit/HybridEncoder.hpp"
#include <complex>
#include <iostream>
#include <cmath>

class ModelCircuit {
    public:
    ModelCircuit();
    static const double pi; // Define pi as a static member variable
    using Complex = std::complex<double>;
    std::vector<std::vector<Complex>> U_theta_lambda(double theta, double lambda);
    void apply_gate(std::vector<Complex>& state_vector, const std::vector<std::vector<Complex>>& gate_matrix, int target_qubit);
    void apply_hadamard(std::vector<Complex>& state_vector, int target_qubit);
    void apply_cnot(std::vector<Complex>& state_vector, int control_qubit, int target_qubit);
    std::vector<Complex> apply_unitary_to_encoded_state(const std::vector<Complex>& phi_x, const std::vector<double>& theta, const std::vector<double>& lambda);
    std::vector<Complex> apply_hybrid_encoding(const std::vector<HybridGate>& hybridGates);
    double measure_overlap_with_zero(const std::vector<Complex>& state_vector);
    double measure_overlap_with_one(const std::vector<Complex>& state_vector);
    double measure_projection_onto_zero(const std::vector<Complex>& state_vector);
    double measure_projection_onto_one(const std::vector<Complex>& state_vector);
    private:
    void initialize_circuit_matrix(int num_qubits);   
};

#endif // MODEL_CIRCUIT_HPP