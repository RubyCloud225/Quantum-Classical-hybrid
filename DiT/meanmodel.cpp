#include "meanmodel.hpp"

// Constructor
MeanVariance::MeanVariance(const std::vector<double>& data) : data(data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty.");
    }
}

// Method to calculate mean
double MeanVariance::calculateMean() const {
    double sum = 0.0;
    for (double value : data) {
        sum += value;
    }
    return sum / data.size();
}

// Method to calculate variance
double MeanVariance::calculateVariance() const {
    double mean = calculateMean();
    double varianceSum = 0.0;
    for (double value : data) {
        varianceSum += (value - mean) * (value - mean);
    }
    return varianceSum / data.size();
}

// Method to calculate mean squared error (loss)
double MeanVariance::calculateLoss(const std::vector<double>& predictions, PredictionType type) const {
    if (predictions.size() != data.size()) {
        throw std::invalid_argument("Predictions vector must be the same size as data vector.");
    }

    double loss = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double error = 0.0;
        switch (type) {
            case PREVIOUS_X:
            error = data[i] - predictions[i];
            break;
            case START_X:
            error = data[i] - predictions[i];
            break;
            case EPSILON:
            error = data[i] - predictions[i];
            break;
            default:
            throw std::invalid_argument("Invalid type.");
        }
        loss += error * error; // Squared error
    }
    return loss / data.size(); // Mean squared error
}

std::vector<double> MeanVariance::makeXPredictions(int numTimesteps) const {
    std::vector<double> predictions(numTimesteps);
    for (int i = 0; i < numTimesteps; ++i) {
        double prediction = (i > 0) ? data[i - 1] : data[0];
        predictions.push_back(prediction);
    }
    return predictions;
}

std::vector<double> MeanVariance::makeStartXPredictions(int numTimesteps) const {
    std::vector<double> predictions(numTimesteps, data[0]);
    return predictions;
}