# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 4.0

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/build

# Include any dependencies generated for this target.
include CMakeFiles/Gaussian_Diffusion_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Gaussian_Diffusion_test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Gaussian_Diffusion_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Gaussian_Diffusion_test.dir/flags.make

CMakeFiles/Gaussian_Diffusion_test.dir/codegen:
.PHONY : CMakeFiles/Gaussian_Diffusion_test.dir/codegen

CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.o: CMakeFiles/Gaussian_Diffusion_test.dir/flags.make
CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.o: /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp
CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.o: CMakeFiles/Gaussian_Diffusion_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.o"
	/opt/homebrew/bin/g++-15 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.o -MF CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.o.d -o CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.o -c /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp

CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.i"
	/opt/homebrew/bin/g++-15 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp > CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.i

CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.s"
	/opt/homebrew/bin/g++-15 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp -o CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.s

# Object files for target Gaussian_Diffusion_test
Gaussian_Diffusion_test_OBJECTS = \
"CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.o"

# External object files for target Gaussian_Diffusion_test
Gaussian_Diffusion_test_EXTERNAL_OBJECTS =

Gaussian_Diffusion_test: CMakeFiles/Gaussian_Diffusion_test.dir/src/ClassicalDiT/tests/Gaussian_Diffusion_test.cpp.o
Gaussian_Diffusion_test: CMakeFiles/Gaussian_Diffusion_test.dir/build.make
Gaussian_Diffusion_test: /opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib/libpython3.13.dylib
Gaussian_Diffusion_test: /opt/homebrew/Cellar/gcc/15.1.0/lib/gcc/15/libgomp.dylib
Gaussian_Diffusion_test: CMakeFiles/Gaussian_Diffusion_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Gaussian_Diffusion_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Gaussian_Diffusion_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Gaussian_Diffusion_test.dir/build: Gaussian_Diffusion_test
.PHONY : CMakeFiles/Gaussian_Diffusion_test.dir/build

CMakeFiles/Gaussian_Diffusion_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Gaussian_Diffusion_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Gaussian_Diffusion_test.dir/clean

CMakeFiles/Gaussian_Diffusion_test.dir/depend:
	cd /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/build /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/build /Users/catherinepemblington/Documents/GitHub/Quantum-Classical-hybrid/build/CMakeFiles/Gaussian_Diffusion_test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/Gaussian_Diffusion_test.dir/depend

