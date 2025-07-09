#include "quantum.hpp"
#include <cmath>

// Quantum Circuit implementation for the library to use across the model
// This is a basic implementation and can be extended to support more operations
// Included are Hadamard, Pauli-X, Pauli-Z, and CNOT gates
QuantumCircuit::QuantumCircuit(int num_qubits) : n(num_qubits) {
    state = ket::Zero(1 << n);
    state[0] = 1; // |00..0>
}

QuantumCircuit::convert_to_ket(const std::vector<cpx>& initial_state) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (initial_state.size() != (1 << n)) {
        throw std::invalid_argument("Initial state size must match 2^n for n qubits.");
    }
    state = ket::Map(const_cast<cpx*>(initial_state.data()), initial_state.size());
}

void QuantumCircuit::apply_gate(const gate& g, int target) {
    // Apply the gate to the target qubit
    std::lock_guard<std::mutex> lock(mtx_);
    int dim = 1 << n;

    for (int i = 0; i < dim; ++i) {
        if (((i >> target) & 1) == 0) {
            // Apply the gate to the state vector
            int j = i | (1 << target);
            cpx a = state[i];
            cpx b = state[j];
            state[i] = g(0, 0) * a + g(0, 1) * b;
            state[j] = g(1, 0) * a + g(1, 1) * b;
        }
    }
}

// Apply a CNOT gate, flips the target qubit if the control qubit is in state |1>
// uses mask for efficient index calculation
void QuantumCircuit::apply_cnot(int control, int target) {
    // Apply CNOT gate
    std::lock_guard<std::mutex> lock(mtx_);
    int dim = 1 << n;
    ket new_state = state;
    
    for (int i = 0; i < dim; ++i ) {
        if (((i >> control) & 1) == 1) { // If control qubit is |1>
            int flip_mask = i ^ (1 << target);
            new_state[flipped] = state[i];
        }
    }
    state = new_state;
}

// Apply a list of single-qubit gates in parallel threads
// Each gate must target a distinct qubit
void QuantumCircuit::apply_parallel_gates(const std::vector<gate>& gates, const std::vector<int>& targets) {
    std::vector<std::thread> threads;

    for (size_t i = 0; i < gates.size(); ++i) {
        threads.emplace_back([&, i] () {
            apply_gate(gates[i], targets[i]);
        });
    }
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}

// Returns the fill state vector of the quantum circuit
// This is the state of the circuit after all gates have been applied

ket QuantumCircuit::get_state() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return state;
}

// reset the circuit to the initial state |00..0>
void QuantumCircuit::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    state = ket(1 << n);
    state[0] = 1; // Reset to |00..0>
}

// Quantum Gates implementation

gate QuantumGates::Hadamard() {
    gate h;
    h << 1, 1,
         1, -1;
    return h / std::sqrt(2.0); // Normalize the gate
}
gate QuantumGates::PauliX() {
    gate x;
    x << 0, 1,
         1, 0;
    return x; // Pauli-X gate
}
gate QuantumGates::PauliZ() {
    gate z;
    z << 1, 0,
         0, -1;
    return z; // Pauli-Z gate
}
gate QuantumGates::Identity() {
    gate i = gate::Identity(); // Identity gate
    return i;
}