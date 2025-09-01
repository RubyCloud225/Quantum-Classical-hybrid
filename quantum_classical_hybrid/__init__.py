# quantum_classical_hybrid/__init__.py
import sys
import os

# Ensure lib folder is in Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../lib"))

# Import the compiled module
import quantum_classical_hybrid

# Expose the bindings in Python as preprocessing
preprocessing = quantum_classical_hybrid
