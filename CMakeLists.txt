cmake_minimum_required(VERSION 3.14)
project(HypernetworkModel LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11 packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(external/pybind11)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/NN
    ${PROJECT_SOURCE_DIR}/NN/Layers
    ${PROJECT_SOURCE_DIR}/DiT
)

# Source files
set(SOURCES
    PythonBindings.cpp
    NN/EpsilonPredictor.cpp
    NN/NeuralNetwork.cpp
    NN/Layers/ConvolutionalLayer.cpp
    NN/Layers/Flatten.cpp
    NN/Layers/FullyConnectedLayer.cpp
    NN/Layers/PoolingLayer.cpp
    NN/Layers/ReluLayer.cpp
    DiT/BetaSchedule.cpp
    DiT/Diffusion_model.cpp
    DiT/Diffusion_Sample.cpp
    DiT/GaussianDiffusion.cpp
    DiT/NormalDist.cpp
)

# Create Python module
pybind11_add_module(epsilonpredictor ${SOURCES})

# Link libraries
target_link_libraries(epsilonpredictor PRIVATE Python::Python pybind11::module)

# Set properties for Python module
set_target_properties(epsilonpredictor PROPERTIES
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
