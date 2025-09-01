import sys
import os

# Add the compiled Pybind11 module path
sys.path.append(os.path.join(os.path.dirname(__file__), "../lib"))

import quantum_classical_hybrid as preprocessing  # This matches your compiled module name

env_path = os.path.join(os.path.dirname(__file__), "../.env")
preprocessing.load_dotenv(env_path)

class PreprocessingPipeline:
    def __init__(self):
        self.steps = []

    def add_step(self, step):
        self.steps.append(step)

    def run(self, data):
        for step in self.steps:
            data = step(data)
        return data