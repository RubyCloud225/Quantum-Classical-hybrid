cmake_minimum_required(VERSION 3.14)
project(quantum_classical_hybrid LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11 packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

# Set pybind11_DIR to Homebrew path to fix find_package error
# set(pybind11_DIR /opt/homebrew/lib/cmake/pybind11)

# Ensure pybind11 is found from Homebrew path
if(NOT pybind11_FOUND)
    find_package(pybind11 REQUIRED PATHS /opt/homebrew/lib/cmake/pybind11)
endif()

# Download and configure Eigen3
#include(FetchContent)
#FetchContent_Declare(
#  eigen
#  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#  GIT_TAG 3.4.0
#)
#FetchContent_MakeAvailable(eigen)

# Find the required libraries
include_directories(
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${PROJECT_SOURCE_DIR}/src/DataProcessing/normaliser
    ${PROJECT_SOURCE_DIR}/src/Bindings
    ${PROJECT_SOURCE_DIR}/src/quantum_library
    ${PROJECT_SOURCE_DIR}/src/Qubit_graph
    /opt/homebrew/include/eigen3
    /opt/homebrew/opt/icu4c@77/include
    ${Python3_INCLUDE_DIRS}
)

file(GLOB_RECURSE DATAPROCESSING_SOURCES src/DataProcessing/*.cpp)
file(GLOB_RECURSE NORMALISER_SOURCES src/DataProcessing/normaliser/*.cpp)
file(GLOB_RECURSE NORMALISER_TESTS src/DataProcessing/normaliser/tests/*.cpp)

# Remove test files from Hypernetwork module sources
list(FILTER DATAPROCESSING_SOURCES EXCLUDE REGEX ".*tests/.*")
list(FILTER NORMALISER_SOURCES EXCLUDE REGEX ".*tests/.*")

pybind11_add_module(quantum_classical_hybrid
    ${DATAPROCESSING_SOURCES}
    ${NORMALISER_SOURCES}
)

# Add separate executable targets for normaliser tests
foreach(test_file ${NORMALISER_TESTS})
    get_filename_component(test_name ${test_file} NAME_WE)
    set(test_sources ${test_file})
    if(test_name STREQUAL "bert_test" OR test_name STREQUAL "bert_minimal_test")
        list(APPEND test_sources src/DataProcessing/normaliser/bert.cpp)
    endif()
    add_executable(${test_name} ${test_sources})
    target_include_directories(${test_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/src/DataProcessing/normaliser
        ${PROJECT_SOURCE_DIR}/src/DataProcessing
        ${Python3_INCLUDE_DIRS}
    )
    target_link_libraries(${test_name} PRIVATE ${Python3_LIBRARIES} OpenMP::OpenMP_CXX)
    target_compile_options(${test_name} PRIVATE -g)
endforeach()

# Set the include directories for the quantum_classical_hybrid module
target_link_libraries(quantum_classical_hybrid PRIVATE
    ${Python3_LIBRARIES}
    OpenMP::OpenMP_CXX
    icuuc
    icui18n
)
target_include_directories(quantum_classical_hybrid PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${PROJECT_SOURCE_DIR}/src/Bindings
    ${PROJECT_SOURCE_DIR}/src/quantum_library
    /opt/homebrew/opt/icu4c@77/include
)
# Set the output directory and output name for the compiled module
set_target_properties(quantum_classical_hybrid PROPERTIES
    OUTPUT_NAME "quantum-classical-hybrid"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# Add executable target for Diffusion Sample Test with debug flags
# add_executable(DiffusionSampleTest src/DiT/tests/Diffusion_sample_test.cpp src/DiT/Diffusion_model.cpp src/DiT/Diffusion_Sample.cpp)

# target_include_directories(DiffusionSampleTest PRIVATE
#     ${PROJECT_SOURCE_DIR}/src/DiT
#     ${PROJECT_SOURCE_DIR}/src/DataProcessing
# )

# target_compile_options(DiffusionSampleTest PRIVATE -g)

# target_link_libraries(DiffusionSampleTest PRIVATE
#     ${Python3_LIBRARIES}
#     OpenMP::OpenMP_CXX
# )

# Add executable target for Gaussian Diffusion Test with debug flags
add_executable(GaussianDiffusionTest 
    src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp 
    src/ClassicalDiT/GaussianDiffusion.cpp 
    src/ClassicalDiT/NN/EpsilonPredictor.cpp 
    src/ClassicalDiT/NN/NeuralNetwork.cpp 
    src/ClassicalDiT/NN/Layers/ConvolutionalLayer.cpp 
    src/ClassicalDiT/NN/Layers/Flatten.cpp 
    src/ClassicalDiT/NN/Layers/FullyConnectedLayer.cpp 
    src/ClassicalDiT/NN/Layers/PoolingLayer.cpp 
    src/ClassicalDiT/NN/Layers/ReluLayer.cpp 
    )

target_include_directories(GaussianDiffusionTest PRIVATE
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT/NN
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    /opt/homebrew/opt/libomp/include
    /opt/homebrew/opt/icu4c/include
)

target_compile_options(GaussianDiffusionTest PRIVATE -g)

# Enable OpenMP support for GaussianDiffusionTest
# Configure CMake to use GCC installed via Homebrew for OpenMP support

# Set C and C++ compilers to GCC and G++ from Homebrew
set(CMAKE_C_COMPILER /opt/homebrew/bin/gcc-15)
set(CMAKE_CXX_COMPILER /opt/homebrew/bin/g++-15)

# Set OpenMP flags for GCC
set(OpenMP_C_FLAGS "-fopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")
set(OpenMP_CXX_LIB_NAMES "gomp")
set(OpenMP_C_LIB_NAMES "gomp")

# Explicitly set OpenMP_gomp_LIBRARY path to GCC libgomp.dylib
set(OpenMP_gomp_LIBRARY /opt/homebrew/Cellar/gcc/15.1.0/lib/gcc/15/libgomp.dylib)

# Remove setting OpenMP_omp_LIBRARY and OpenMP_LIBRARY to libomp.dylib to avoid conflict
# set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
# set(OpenMP_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
# set(OpenMP_LIBRARIES /opt/homebrew/opt/libomp/lib/libomp.dylib)
# set(OpenMP_CXX_LIBRARIES /opt/homebrew/opt/libomp/lib/libomp.dylib)
set(OpenMP_LINK_FLAGS "-L/opt/homebrew/opt/libomp/lib -lomp")
set(OpenMP_CXX_INCLUDE_DIR /opt/homebrew/opt/libomp/include)

# Add OpenMP include directory explicitly to compiler flags
add_compile_options(
    -I/opt/homebrew/opt/libomp/include
    -I/opt/homebrew/opt/icu4c@77/include
)
add_definitions(-fopenmp)

# Also set environment variables for the build process
set(ENV{CPPFLAGS} "-I/opt/homebrew/opt/libomp/include")
set(ENV{LDFLAGS} "-L/opt/homebrew/opt/libomp/lib")

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_include_directories(GaussianDiffusionTest PRIVATE ${OpenMP_CXX_INCLUDE_DIR})
    target_link_libraries(GaussianDiffusionTest PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(GaussianDiffusionTest PRIVATE ${OpenMP_CXX_FLAGS} -I/opt/homebrew/opt/libomp/include -fopenmp)
    target_link_directories(GaussianDiffusionTest PRIVATE /opt/homebrew/opt/libomp/lib)
    target_link_options(GaussianDiffusionTest PRIVATE ${OpenMP_LINK_FLAGS})
endif()

target_link_libraries(GaussianDiffusionTest PRIVATE
    ${Python3_LIBRARIES}
    icuuc
    icui18n
)


