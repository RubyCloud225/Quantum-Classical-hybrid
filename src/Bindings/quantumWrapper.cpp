// Pybind11 and model_circuit
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "model_circuit.hpp"
#include "../Quantum_encoder/converttoqubit/AngleEncoder.hpp"
#include "../Quantum_encoder/converttoqubit/HybridEncoder.hpp"

namespace py = pybind11;

static void apply_hybrid_encoding_wrapper(ModelCircuit& mc, const std::vector<HybridGate>& hybridGates, const Hamiltonian* hamiltonian = nullptr) {
    mc.apply_hybrid_encoding(hybridGates, hamiltonian);
}

PYBIND11_MODULE(model_circuit, m) {
    // Existing ModelCircuit binding
    py::class_<ModelCircuit>(m, "ModelCircuit")
    .def(py::init<>())
        .def_static("apply_hybrid_encoding", apply_hybrid_encoding_wrapper,
                    py::arg("hybridGates"),
                    py::arg("hamiltonian") = nullptr,
                    "Apply hybrid encoding to a circuit with optional Hamiltonian.");

    // Free functions for measurement
    m.def("measure_overlap_with_zero", &measure_overlap_with_zero,
        py::arg("state_vector"),
        "Measure overlap with |0> state.");

    m.def("measure_overlap_with_one", &measure_overlap_with_one,
        py::arg("state_vector"),
        "Measure overlap with |1> state.");

    m.def("measure_projection_onto_zero", &measure_projection_onto_zero,
        py::arg("state_vector"),
        "Measure projection onto |0> state.");

    m.def("measure_projection_onto_one", &measure_projection_onto_one,
        py::arg("state_vector"),
        "Measure projection onto |1> state.");
    
    // Add HybridGate + RotationAxis bindings
    py::enum_<RotationAxis>(m, "RotationAxis")
        .value("X", RotationAxis::X)
        .value("Y", RotationAxis::Y)
        .value("Z", RotationAxis::Z);

    py::class_<HybridGate>(m, "HybridGate")
        .def(py::init<RotationAxis, double, int>(),
            py::arg("axis"),
            py::arg("angle"),
            py::arg("target_qubit"))
        .def_readwrite("axis", &HybridGate::axis)
        .def_readwrite("angle", &HybridGate::angle)
        .def_readwrite("target_qubit", &HybridGate::target_qubit);
}