#pragma once
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <thread>
#include <mutex>

using cpx = std::complex<double>;
using ket = Eigen::vectorXcd;
using gate = Eigen::MatrixXcd;

class QuantumCircuit {
    public:
        // Constructor
        QuantumCircuit(int num_qubits);
        void convert_to_ket(const std::vector<cpx>& initial_state);
        void apply_gate(const gate& g, int target);
        void apply_cnot(int control, int target);
        void apply_parallel_gates(const std::vector<gate>& gates, const std::vector<int>& targets);
        ket get_state() const;
        void reset();
    private:
        int n;
        ket state_;
        std::mutex mtx_;
};

// Helper functions
namespace QuantumGates {
    gate Hadamard();
    gate PauliX();
    gate PauliZ();
    gate Identity();
}