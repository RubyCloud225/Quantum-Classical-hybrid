#ifndef QUANTUM_COMPRESSOR_HPP
#define QUANTUM_COMPRESSOR_HPP

class QuantumCompressor {
    public:
    using Complex = std::complex<double>;
    // function to simplify using Clifford Algorithm
    std::vector<Complex> simplify_clifford(const std::vector<Complex>& state_vector);
    // Contract low-importance qubits from tensor product
    std::vector<Complex> contract_low_importance_qubits(const std::vector<Complex>& state_vector, const std::vector<int>& low_importance_qubits);
    // Simulate measurement on a specific quibit and collapse the state vector
    std::vector<Complex> measurement_collapse(const std::vector<Complex>& state_vector, int qubit_index, int measurement_result);
    // Utility: Calculate the norm of a state vector
    double probability_of_measurement(const std::vector<Complex>& state_vector, int qubit_index, int measurement_result);
}