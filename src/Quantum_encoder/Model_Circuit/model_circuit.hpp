#ifndef Model_Circuit_hpp
#define Model_Circuit_hpp

#include <vector>
#include <complex>
#include <iostream>
#include <cmath>

class ModelCircuit {
    public:
    ModelCircuit();
    const double pi = 3.14159265358979323846;
    using Complex = std::complex<double>;
    std::vector<std::vector<Complex>> U_theta_lambda(double theta, double lambda);
    void apply_gate(std::vector<Complex>& state_vector, const std::vector<std::vector<Complex>>& gate_matrix, int target_qubit);
    void apply_hadamard(std::vector<Complex>& state_vector, int target_qubit);
    void apply_cnot(std::vector<Complex>& state_vector, int control_qubit, int target_qubit);
    std::vector<Complex> apply_unitary_to_encoded_state(const std::vector<Complex>& phi_x, const std::vector<double>& theta, const std::vector<double<& lambda);
    double measure_overlap_with_zero(const std::vector<Complex>& state_vector);
    double measure_overlap_with_one(const std::vector<Complex>& state_vector);
    double measure_projection_onto_zero(const std::vector<Complex>& state_vector);
    double measure_projection_onto_one(const std::vector<Complex>& state_vector);
    private:
    void initialize_circuit_matrix(int num_qubits);   
}

