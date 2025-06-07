cmake_minimum_required(VERSION 3.14)
project(HypernetworkModel LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11 packages
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# Download and configure Eigen3
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# Find the required libraries
include_directories(
    ${PROJECT_SOURCE_DIR}/src/DiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    ${eigen_SOURCE_DIR}
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
    src/DiT/tests/Diffusion_model_test.cpp
    src/DiT/tests/Diffusion_sample_test.cpp
    src/DiT/tests/Gaussian_Diffusion_test.cpp
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
    ${eigen_SOURCE_DIR}/Eigen
)
# Set the output directory for the compiled module
set_target_properties(Hypernetwork PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# Add executable target for Diffusion Sample Test with debug flags
add_executable(DiffusionSampleTest src/DiT/tests/Diffusion_sample_test.cpp src/DiT/Diffusion_model.cpp src/DiT/Diffusion_Sample.cpp)

target_include_directories(DiffusionSampleTest PRIVATE
    ${PROJECT_SOURCE_DIR}/src/DiT
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
)

target_compile_options(DiffusionSampleTest PRIVATE -g)

target_link_libraries(DiffusionSampleTest PRIVATE
    ${Python3_LIBRARIES}
)

# Add executable target for Gaussian Diffusion Test with debug flags
add_executable(GaussianDiffusionTest src/DiT/tests/Gaussian_Diffusion_test.cpp src/DiT/GaussianDiffusion.cpp src/DiT/NN/EpsilonPredictor.cpp src/DiT/NN/NeuralNetwork.cpp src/DiT/NN/Layers/ConvolutionalLayer.cpp src/DiT/NN/Layers/Flatten.cpp src/DiT/NN/Layers/FullyConnectedLayer.cpp src/DiT/NN/Layers/PoolingLayer.cpp src/DiT/NN/Layers/ReluLayer.cpp src/DiT/NN/NeuralNetwork.cpp)

target_include_directories(GaussianDiffusionTest PRIVATE
    ${PROJECT_SOURCE_DIR}/src/DiT
    ${PROJECT_SOURCE_DIR}/src/DiT/NN
    ${PROJECT_SOURCE_DIR}/src/DataProcessing
    /opt/homebrew/opt/libomp/include
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
add_compile_options(-I/opt/homebrew/opt/libomp/include)
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
)


