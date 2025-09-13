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
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${PROJECT_SOURCE_DIR}/src/Bindings
    ${PROJECT_SOURCE_DIR}/src/Quantum_encoder
    ${PROJECT_SOURCE_DIR}/src/Compression
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/DataProcessing/models
    ${Python3_INCLUDE_DIRS}
)

# Platform-specific include directories
if(APPLE)
    include_directories(/opt/homebrew/include/eigen3 /opt/homebrew/include ${OpenMP_INCLUDE_DIR})
elseif(UNIX AND NOT APPLE)
    include_directories(/usr/include/eigen3 ${OpenMP_INCLUDE_DIR})
endif()
link_directories(
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${PROJECT_SOURCE_DIR}/src/Bindings
    ${PROJECT_SOURCE_DIR}/src/Quantum_encoder
    ${PROJECT_SOURCE_DIR}/src/Compression
    ${PROJECT_SOURCE_DIR}/src/utils
)

# Platform-specific link directories
if(APPLE)
    link_directories(/opt/homebrew/lib)
endif()

add_library(shared_lib STATIC
    src/DataProcessing/models/sampleData.cpp
    src/ClassicalDiT/GaussianDiffusion.cpp
    src/ClassicalDiT/Diffusion_model.cpp
    src/ClassicalDiT/Diffusion_Sample.cpp
)
target_include_directories(shared_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/src/utils
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
)
target_link_libraries(shared_lib PUBLIC OpenMP::OpenMP_CXX)

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

file(GLOB_RECURSE DATAPROCESSING_SOURCES src/DataProcessing/*.cpp)
file(GLOB_RECURSE NORMALISER_SOURCES src/DataProcessing/normaliser/*.cpp)

# Remove test files from sources
list(FILTER DATAPROCESSING_SOURCES EXCLUDE REGEX ".*tests/.*")
list(FILTER NORMALISER_SOURCES EXCLUDE REGEX ".*tests/.*")

# Consolidated pybind11 module for quantum_classical_hybrid including all normaliser sources and PreprocessingBindings.cpp
pybind11_add_module(quantum_classical_hybrid
    ${DATAPROCESSING_SOURCES}
    ${NORMALISER_SOURCES}
    src/Bindings/PreprocessingBindings.cpp
)

target_link_libraries(quantum_classical_hybrid PRIVATE
    ${Python3_LIBRARIES}
    OpenMP::OpenMP_CXX
)
target_include_directories(quantum_classical_hybrid PRIVATE
    ${PROJECT_SOURCE_DIR}/src/utils
    ${Python3_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src/ClassicalDiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${PROJECT_SOURCE_DIR}/src/Bindings
    ${PROJECT_SOURCE_DIR}/src/Quantum_encoder
    ${PROJECT_SOURCE_DIR}/src/Compression
)
set_target_properties(quantum_classical_hybrid PROPERTIES
    OUTPUT_NAME "preprocessing"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

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
            )
            target_compile_options(${test_name} PRIVATE -g)
        endif()
    endforeach()
endfunction()

if(EXISTS "${PROJECT_SOURCE_DIR}/src/ClassicalDiT/tests")
    add_test_executable_with_sources(
        "${PROJECT_SOURCE_DIR}/src/ClassicalDiT/tests"
        "src/ClassicalDiT;src/DataProcessing/models"
    )
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/src/DataProcessing/normaliser/tests")
    add_test_executable_with_sources(
        "${PROJECT_SOURCE_DIR}/src/DataProcessing/normaliser/tests"
        "src/DataProcessing/normaliser"
    )
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/src/Quantum_encoder/tests")
    add_test_executable_with_sources(
        "${PROJECT_SOURCE_DIR}/src/Quantum_encoder/tests"
        "src/Quantum_encoder"
    )
endif()

if(EXISTS "${PROJECT_SOURCE_DIR}/src/Quantum_encoder/convert_classical_state_to_Quantum.md/tests")
    add_test_executable_with_sources(
        "${PROJECT_SOURCE_DIR}/src/Quantum_encoder/convert_classical_state_to_Quantum.md/tests"
        "src/Quantum_encoder"
    )
endif()