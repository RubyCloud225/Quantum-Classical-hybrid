cmake_minimum_required(VERSION 4.0.1)
set(CMAKE_MODULE_PATH "/Users/catherinepemblington/Documents/GitHub/Hypernetwork/build/_deps/eigen-src/cmake")
project(CheckFortran Fortran)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/result.cmake"
  "set(CMAKE_Fortran_COMPILER \"${CMAKE_Fortran_COMPILER}\")\n"
  
  )