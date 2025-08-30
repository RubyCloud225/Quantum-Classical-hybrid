cmake_minimum_required(VERSION 3.14)
project(quantum_classical_hybrid LANGUAGES CXX)


# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
    set(OPENMP_ROOT "/opt/homebrew/opt/libomp" CACHE PATH "Path to libomp")
    set(OpenMP_ROOT "${OPENMP_ROOT}" CACHE PATH "Path to libomp")
    set(OpenMP_DIR "${OPENMP_ROOT}/lib/cmake/omp" CACHE PATH "Path to OpenMP cmake dir")
    set(OpenMP_omp_LIBRARY "${OPENMP_ROOT}/lib/libomp.dylib" CACHE FILEPATH "OpenMP libomp library")
    set(OpenMP_LIBRARIES "${OpenMP_omp_LIBRARY}" CACHE FILEPATH "OpenMP libraries")
    set(OpenMP_INCLUDE_DIR "${OPENMP_ROOT}/include" CACHE PATH "OpenMP include dir")
endif()

find_package(OpenMP REQUIRED)


project(quantum_classical_hybrid LANGUAGES CXX)

if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES arm64)
    if(OpenMP_FOUND)
        message(STATUS "OpenMP found on macOS")
    else()
        message(FATAL_ERROR "OpenMP not found on macOS")
    endif()

elseif(WIN32)
    message(STATUS "Configuring for Windows MSVC")
    # Enable OpenMP for MSVC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    add_compile_options(/openmp)

elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux")
    # Use GCC and OpenMP
    find_package(OpenMP REQUIRED)
    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR "OpenMP not found on Linux")
    endif()
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

# Set the CMAKE_BUILD_TYPE if not already set
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Link OpenMP
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenMP not found")
endif()

# Find Python and pybind11 packages
find_package(
    Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
    COMMAND python3 -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

if(NOT pybind11_FOUND)
    find_package(pybind11 REQUIRED PATHS /opt/homebrew/lib/cmake/pybind11)
endif()

# Find the required libraries
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${PROJECT_SOURCE_DIR}/src/Bindings
    ${PROJECT_SOURCE_DIR}/src/Quantum_encoder
    ${PROJECT_SOURCE_DIR}/src/Compression
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/DataProcessing/models
    /opt/homebrew/include/eigen3
    /opt/homebrew/include
    $ENV{HOME}/icu-install/include
    ${Python3_INCLUDE_DIRS}
)
link_directories(
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${PROJECT_SOURCE_DIR}/src/Bindings
    ${PROJECT_SOURCE_DIR}/src/Quantum_encoder
    ${PROJECT_SOURCE_DIR}/src/Compression
    ${PROJECT_SOURCE_DIR}/src/utils
    /opt/homebrew/lib
    $ENV{HOME}/icu-install/lib
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

# --- Universal function to add test executables with all required sources (multi-module version) ---
function(add_test_executable_with_sources test_dir module_dirs)
    file(GLOB_RECURSE TEST_FILES ${test_dir}/*.cpp)
    list(FILTER TEST_FILES EXCLUDE REGEX ".*/main.cpp$")
    set(MODULE_SOURCES "")
    foreach(module_dir IN LISTS module_dirs)
        file(GLOB_RECURSE MODULE_SOURCES_TMP RELATIVE ${PROJECT_SOURCE_DIR}
             ${PROJECT_SOURCE_DIR}/${module_dir}/*.cpp)
        list(FILTER MODULE_SOURCES_TMP EXCLUDE REGEX ".*/main.cpp$")
        list(FILTER MODULE_SOURCES_TMP EXCLUDE REGEX ".*tests/.*")
        list(APPEND MODULE_SOURCES ${MODULE_SOURCES_TMP})
    endforeach()

    foreach(test_file ${TEST_FILES})
        get_filename_component(test_name ${test_file} NAME_WE)
        set(test_sources ${test_file})

        foreach(source ${MODULE_SOURCES})
            list(APPEND test_sources ${PROJECT_SOURCE_DIR}/${source})
        endforeach()

        if(NOT TARGET ${test_name})
            add_executable(${test_name} ${test_sources})
            target_include_directories(${test_name} PRIVATE
                ${Python3_INCLUDE_DIRS}
            )
            target_link_libraries(${test_name} PRIVATE
                ${Python3_LIBRARIES}
                OpenMP::OpenMP_CXX
                icuuc
                icui18n
            )
            target_compile_options(${test_name} PRIVATE -g)
        endif()
    endforeach()
endfunction()

# Update your calls to include all relevant module directories per test

add_test_executable_with_sources(
    "${PROJECT_SOURCE_DIR}/src/ClassicalDiT/tests"
    "src/ClassicalDiT;src/DataProcessing/models"
)

add_test_executable_with_sources(
    "${PROJECT_SOURCE_DIR}/src/DataProcessing/normaliser/tests"
    "src/DataProcessing/normaliser"
)

foreach(target IN LISTS TEST_FILES)
    if(TARGET ${target})
        target_include_directories(${target} PRIVATE
            ${PROJECT_SOURCE_DIR}/src/DataProcessing/normaliser
        )
    endif()
endforeach()

add_test_executable_with_sources(
    "${PROJECT_SOURCE_DIR}/src/Quantum_encoder/tests"
    "src/Quantum_encoder"
)

# Set the include directories for the quantum_classical_hybrid module
target_link_libraries(quantum_classical_hybrid PRIVATE
    ${Python3_LIBRARIES}
    OpenMP::OpenMP_CXX
    icuuc
    icui18n
)
target_include_directories(quantum_classical_hybrid PRIVATE
    ${PROJECT_SOURCE_DIR}/src/utils
    ${Python3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${PROJECT_SOURCE_DIR}/src/Bindings
    ${PROJECT_SOURCE_DIR}/src/Quantum_encoder
    ${PROJECT_SOURCE_DIR}/src/Compression
    $ENV{HOME}/icu-install/include
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

# Add main program executable
add_executable(main_program
    src/ClassicalDiT/main.cpp
    src/ClassicalDiT/BetaSchedule.cpp
    src/ClassicalDiT/GaussianDiffusion.cpp
    src/ClassicalDiT/Diffusion_model.cpp
    src/ClassicalDiT/Diffusion_Sample.cpp
    src/ClassicalDiT/NN/EpsilonPredictor.cpp
    src/ClassicalDiT/NN/NeuralNetwork.cpp
    src/ClassicalDiT/NN/Layers/ConvolutionalLayer.cpp
    src/ClassicalDiT/NN/Layers/Flatten.cpp
    src/ClassicalDiT/NN/Layers/FullyConnectedLayer.cpp
    src/ClassicalDiT/NN/Layers/PoolingLayer.cpp
    src/ClassicalDiT/NN/Layers/ReluLayer.cpp
    src/DataProcessing/models/sampleData.cpp
)
target_link_libraries(main_program PRIVATE
    OpenMP::OpenMP_CXX
    ${Python3_LIBRARIES}
    icuuc
    icui18n
)
target_include_directories(main_program PRIVATE
    ${PROJECT_SOURCE_DIR}/src/utils
    ${Python3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${PROJECT_SOURCE_DIR}/src/Bindings
    ${PROJECT_SOURCE_DIR}/src/Quantum_encoder
    ${PROJECT_SOURCE_DIR}/src/Compression
    $ENV{HOME}/icu-install/include
)
add_library(Quantum_encoder SHARED
    src/Quantum_encoder/converttoqubit/AngleEncoder.cpp
    src/Quantum_encoder/converttoqubit/HybridEncoder.cpp
    src/Quantum_encoder/Model_Circuit/model_circuit.cpp
    src/Quantum_encoder/Hamiltonian/time_H.cpp
)
target_include_directories(Quantum_encoder PRIVATE
    ${PROJECT_SOURCE_DIR}/src/Quantum_encoder
    ${PROJECT_SOURCE_DIR}/src/utils
    ${Python3_INCLUDE_DIRS}
)

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
set(OpenMP_LINK_FLAGS "-L/opt/homebrew/opt/libomp/lib -lomp")
set(OpenMP_CXX_INCLUDE_DIR /opt/homebrew/opt/libomp/include)

# Add OpenMP include directory explicitly to compiler flags
add_compile_options(
    -I/opt/homebrew/opt/libomp/include
    -I$ENV{HOME}/icu-install/include
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
    OpenMP::OpenMP_CXX
    ${Python3_LIBRARIES}
    icuuc
    icui18n
)

add_test_executable_with_sources(
    "${PROJECT_SOURCE_DIR}/src/Quantum_encoder/convert_classical_state_to_Quantum.md/tests"
    "src/Quantum_encoder"
)
