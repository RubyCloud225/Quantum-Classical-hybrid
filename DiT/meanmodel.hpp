#ifndef MEAN_VARIANCE_HPP
#define MEAN_VARIANCE_HPP

#include <vector>
#include <stdexcept>

class MeanVariance {
public:
    enum PredictionType {
        PREVIOUS_X,
        START_X,
        EPSILON
    };
    // Constructor
    MeanVariance(const std::vector<double>& data);

    // Method to calculate mean
    double calculateMean() const;

    // Method to calculate variance
    double calculateVariance() const;

    // Method to calculate mean squared error (loss)
    double calculateLoss(const std::vector<double>& predictions, PredictionType type) const;
    std::vector<double> makeXPredictions(int numTimesteps) const;
    std::vector<double> makeStartXPredictions(int numTimesteps) const;

private:
    std::vector<double> data; // Store the input data
};

#endif // MEAN_VARIANCE_HPP