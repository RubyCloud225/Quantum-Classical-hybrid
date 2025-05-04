import sys
import os
import epsilonpredictor

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'build'))
if module_path not in sys.path:
    sys.path.append(module_path)

def main():
    # Instantiate the EpsilonPredictor class
    predictor = epsilonpredictor.EpsilonPredictor(input_channels=10, output_size=1)
    
    # Example input data
    x_t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Example input vector
    t = 1  # Example timestep
    
    # Call the predictEpilson method
    result = predictor.predictEpilson(x_t, t)
    print("Prediction result:", result)

if __name__ == "__main__":
    main()