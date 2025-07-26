#include "quantum.hpp"
#include <iostream>
#include <complex>
#include <cassert>
#include "utils/logger.hpp"
#include <cmath>

using namespace std;

// compare two complex vectors with epsilon tolerance
bool approx_equal(const ket& a, const ket& b, double eps = 1e-6) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); ++i) {
        if (abs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    Logger::log("Vectors are approximately equal within tolerance: " + std::to_string(eps), INFO);
    return true;
}

void test_initial_state() {
    QuantumCircuit qc(2);
    ket expected(4);
    expected << 1, 0, 0, 0; // |00>
    assert(approx_equal(qc.get_state(), expected));
    Logger::log("test_initial_state passed", INFO);
}

void test_hadamard_gate() {
    QuantumCircuit qc(1);
    qc.apply_gate(QuantumGates::Hadamard(), 0);
    ket result = qc.get_state();
    double amp = 1.0 / sqrt(2.0);
    ket expected(2);
    expected << amp, amp; // |+>
    assert(approx_equal(result, expected));
    Logger::log("test_hadamard_gate passed", INFO);
}

void test_pauli_x_gate() {
    QuantumCircuit qc(1);
    qc.apply_gate(QuantumGates::PauliX(), 0);
    ket result = qc.get_state();
    ket expected(2);
    expected << 0, 1; // |1>
    assert(approx_equal(result, expected));
    Logger::log("test_pauli_x_gate passed", INFO);
}

void test_cnot_gate() {
    QuantumCircuit qc(2);
    // apply hadamard to control qubit
    qc.apply_gate(QuantumGates::Hadamard(), 0);
    // apply CNOT with control qubit 0 and target qubit 1
    qc.apply_cnot(0, 1);
    ket result = qc.get_state();
    ket expected(4);
    expected << amp, 0, 0, amp; // |01> + |11>
    assert(approx_equal(result, expected));
    Logger::log("test_cnot_gate passed", INFO);
}

void test_parallel_gates() {
    QuantumCircuit qc(2);
    // Apply Hadamard to qubit 0 and Pauli-X to qubit 1 in parallel
    std::vector<gate> gates = {QuantumGates::Hadamard(), QuantumGates::PauliX()};
    std::vector<int> targets = {0, 1};
    qc.apply_parallel_gates(gates, targets);
    
    ket result = qc.get_state();
    ket expected(4);
    expected << 1.0 / sqrt(2.0), 0, 0, 1.0 / sqrt(2.0); // |00> + |11>
    assert(approx_equal(result, expected));
    Logger::log("test_parallel_gates passed", INFO);
}
void test_reset() {
    QuantumCircuit qc(2);
    // Apply some gates
    qc.apply_gate(QuantumGates::Hadamard(), 0);
    qc.apply_cnot(0, 1);
    
    // Reset the circuit
    qc.reset();
    
    // Check if the state is back to |00>
    ket expected(4);
    expected << 1, 0, 0, 0; // |00>
    assert(approx_equal(qc.get_state(), expected));
    Logger::log("test_reset passed", INFO);
}

void run_tests() {
    test_initial_state();
    test_hadamard_gate();
    test_pauli_x_gate();
    test_cnot_gate();
    test_parallel_gates();
}
