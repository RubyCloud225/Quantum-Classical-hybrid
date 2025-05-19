cmake_minimum_required(VERSION 3.14)
project(HypernetworkModel LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11 packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)


# Find the required libraries
include_directories(
    ${PROJECT_SOURCE_DIR}/src/DiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
)

pybind11_add_module(Hypernetwork
    src/Bindings/DiTWrapper.cpp
    src/Bindings/DataWrapper.cpp
    src/DiT/main.cpp
    src/DiT/BetaSchedule.cpp
    src/DiT/GaussianDiffusion.cpp
    src/DiT/NN/EpsilonPredictor.cpp
    src/DiT/Diffusion_model.cpp
    src/DiT/Diffusion_Sample.cpp
    src/DataProcessing/sampleData.cpp
    src/DataProcessing/GaussianNoise.cpp
    src/DataProcessing/LinearRegression.cpp
    src/DataProcessing/PreprocessingBindings.cpp
    src/DataProcessing/tokenizer.cpp
    src/DataProcessing/LayerNormalization.cpp
    src/DataProcessing/tests/LayerNormalization_performance.cpp
    src/DataProcessing/tests/LayerNormalization_edge_cases.cpp
    src/DataProcessing/tests/Integration_GaussianNoise_LayerNormalization.cpp
    src/DataProcessing/tests/LinearRegression_basic_tests.cpp
    src/DataProcessing/tests/LinearRegression_edge_cases.cpp
    src/DataProcessing/tests/Integration_LinearRegression_GaussianNoise_LayerNormalization.cpp
    src/DataProcessing/tests/sampleData_basic_tests.cpp
    src/DataProcessing/tests/gaussianNoise_edge_cases.cpp
    src/DiT/tests/BetaSchedule_tests.cpp
    src/DiT/tests/BetaSchedule_edge_cases.cpp
)

# Set the include directories for the Hypernetwork module
target_link_libraries(Hypernetwork PRIVATE
    ${Python3_LIBRARIES}
    ${Python3_INCLUDE_DIRS}
)
target_include_directories(Hypernetwork PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/DiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
)
# Set the output directory for the compiled module
set_target_properties(Hypernetwork PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_SOURCE_DIR}/lib
)


