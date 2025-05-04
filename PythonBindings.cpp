#include <Python.h>
#include "NN/EpsilonPredictor.hpp"

// Wrapper for the EpsilonPredictor class
static PyObject* py_epsilon_predictor_new(PyObject* self, PyObject* args) {
    int input_channels, output_size;
    if (!PyArg_ParseTuple(args, "ii", &input_channels, &output_size)) {
        return nullptr;
    }

    // Create a new EpsilonPredictor instance
    auto* predictor = new EpsilonPredictor(input_channels, output_size);
    return PyCapsule_New(predictor, "EpsilonPredictor", nullptr);
}

static PyObject* py_epsilon_predictor_predict(PyObject* self, PyObject* args) {
    PyObject* capsule;
    PyObject* input_list;
    int t;

    if (!PyArg_ParseTuple(args, "OOi", &capsule, &input_list, &t)) {
        return nullptr;
    }

    // Extract the EpsilonPredictor instance
    auto* predictor = static_cast<EpsilonPredictor*>(PyCapsule_GetPointer(capsule, "EpsilonPredictor"));
    if (!predictor) {
        return nullptr;
    }

    // Convert Python list to std::vector<double>
    std::vector<double> x_t;
    for (Py_ssize_t i = 0; i < PyList_Size(input_list); ++i) {
        x_t.push_back(PyFloat_AsDouble(PyList_GetItem(input_list, i)));
    }

    // Call the predictEpilson method
    auto result = predictor->predictEpilson(x_t, t);

    // Convert std::vector<int> to Python list
    PyObject* result_list = PyList_New(result.size());
    for (size_t i = 0; i < result.size(); ++i) {
        PyList_SetItem(result_list, i, PyLong_FromLong(result[i]));
    }

    return result_list;
}

// Define module methods
static PyMethodDef EpsilonPredictorMethods[] = {
    {"new", py_epsilon_predictor_new, METH_VARARGS, "Create a new EpsilonPredictor instance"},
    {"predict", py_epsilon_predictor_predict, METH_VARARGS, "Call the predictEpilson method"},
    {nullptr, nullptr, 0, nullptr}
};

// Define the module
static struct PyModuleDef EpsilonPredictorModule = {
    PyModuleDef_HEAD_INIT,
    "epsilonpredictor",
    nullptr,
    -1,
    EpsilonPredictorMethods
};

PyMODINIT_FUNC PyInit_epsilonpredictor() {
    return PyModule_Create(&EpsilonPredictorModule);
}
