#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "model_circuit.hpp"

namespace py = pybind11;

PYBIND11_MODULE(model_circuit, m) {
    py::class_<ModelCircuit>(m, "ModelCircuit")
    .def(py::init<>())
    .def("apply_unitary_to_encoded_state", &ModelCircuit::apply_unitary_to_encoded_state, 
         py::arg("encoded_state"), py::arg("unitary_matrix"), py::arg("target_qubit"))
    .def("Measure_projection_onto_zero", &ModelCircuit::measure_projection_onto_zero, 
         py::arg("state_vector"))
    .def("Measure_projection_onto_one", &ModelCircuit::measure_projection_onto_one,
            py::arg("state_vector"))
}