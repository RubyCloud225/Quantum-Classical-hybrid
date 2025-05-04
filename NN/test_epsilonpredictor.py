import sys
sys.path.insert(0, '/Users/catherinepemblington/Documents/GitHub/Hypernetwork/build')
import epsilonpredictor

predictor = epsilonpredictor.EpsilonPredictor(input_channels=3, output_channels=1)

#example input
x_t = [0.1, 0.2, 0.3, 0.4, 0.5] # example input vector
t = 1 # example time step

# Call the forward method
result = predictor.predictEpilson(x_t, t)
print(result)
