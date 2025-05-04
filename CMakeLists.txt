cmake_minimum_required(VERSION 3.15)
project(EpsilonPredictor)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add the Python bindings
pybind11_add_module(epsilonpredictor PythonBindings.cpp)

# Link any additional libraries if needed
target_link_libraries(epsilonpredictor PRIVATE NN)

find_package(Boost REQUIRED COMPONENTS python313) # Replace python38 with your Python version

include_directories(${CMAKE_SOURCE_DIR}/NN)

# Add the NN folder
add_library(NN STATIC 
    # Add the source files for the NN library
    NN/Layers/ConvolutionalLayer.cpp
    NN/Layers/Flatten.cpp
    NN/Layers/FullyConnectedLayer.cpp
    NN/Layers/ReluLayer.cpp
    NN/Layers/PoolingLayer.cpp
    NN/EpsilonPredictor.cpp
    NN/NeuralNetwork.cpp
) # Removed NN/EpsilonPredictor.cpp as it is missing
