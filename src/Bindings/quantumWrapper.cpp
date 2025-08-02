// Pybind11 and model_circuit
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "model_circuit.hpp"
#include "../Quantum_encoder/converttoqubit/AngleEncoder.hpp"
#include "../Quantum_encoder/converttoqubit/HybridEncoder.hpp"

namespace py = pybind11;

PYBIND11_MODULE(model_circuit, m) {
    // Existing ModelCircuit binding
    py::class_<ModelCircuit>(m, "ModelCircuit")
        .def(py::init<>())
        .def("apply_unitary_to_encoded_state", &ModelCircuit::apply_unitary_to_encoded_state,
             py::arg("encoded_state"), py::arg("unitary_matrix"), py::arg("target_qubit"))
        .def("Measure_projection_onto_zero", &ModelCircuit::measure_projection_onto_zero,
             py::arg("state_vector"))
        .def("Measure_projection_onto_one", &ModelCircuit::measure_projection_onto_one,
             py::arg("state_vector"));

    // RotationAxis enum
    py::enum_<RotationAxis>(m, "RotationAxis")
        .value("X", RotationAxis::X)
        .value("Y", RotationAxis::Y)
        .value("Z", RotationAxis::Z)
        .export_values();

    // RotationGate struct
    py::class_<RotationGate>(m, "RotationGate")
        .def_readonly("type", &RotationGate::type)
        .def_readonly("angle", &RotationGate::angle);

    // HybridGate struct
    py::class_<HybridGate>(m, "HybridGate")
        .def_readonly("angleGate", &HybridGate::angleGate)
        .def_readonly("amplitude", &HybridGate::amplitude);

    // AngleEncoder class
    py::class_<AngleEncoder>(m, "AngleEncoder")
        .def(py::init<const std::vector<double>&, RotationAxis>(), py::arg("input"), py::arg("axis") = RotationAxis::Y)
        .def("get_gates", &AngleEncoder::get_gates)
        .def("printGates", &AngleEncoder::printGates);

    // HybridEncoder class
    py::class_<HybridEncoder>(m, "HybridEncoder")
        .def(py::init<const std::vector<double>&>())
        .def("getHybridEncodedGates", &HybridEncoder::getHybridEncodedGates)
        .def("printHybridGates", &HybridEncoder::printHybridGates);
}