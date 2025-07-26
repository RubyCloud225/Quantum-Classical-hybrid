#include "HybridEncoder.hpp"
#include <cmath>
#include "utils/logger.hpp"
#include <iostream>

HybridEncoder::HybridEncoder(const std::vector<double>& input) {
    normaliseAndEncode(input);
}
void HybridEncoder::normaliseAndEncode(const std::vector<double>& input) {
    double norm = 0.0;
    for (double x : input) norm += x * x;
    norm = std::sqrt(norm);

    std::vector<double> normalized_input;
    for (double x : input) {
        normalized_input.push_back(x / norm);
    }
    AngleEncoder angleEncoder(normalized_input, RotationAxis::Y);
    auto angleGates = angleEncoder.get_gates();
    hybridEncodedGates.clear();
    for (size_t i = 0; i < angleGates.size(); ++i) {
        double amplitude = normalized_input[i];
        hybridEncodedGates.push_back({angleGates[i], amplitude});
    }
    Logger::log("HybridEncoder encoded " + std::to_string(hybridEncodedGates.size()) + " gates", INFO);
}

std::vector<HybridGate> HybridEncoder::getHybridEncodedGates() const {
    Logger::log("Retrieving hybrid encoded gates", INFO);
    return hybridEncodedGates;
}

void HybridEncoder::printHybridGates() const {
    for (const auto& hg : hybridEncodedGates) {
        std::string axisStr;
        switch (hg.angleGate.type) {
            case RotationType::X:
                axisStr = "X";
                break;
            case RotationType::Y:
                axisStr = "Y";
                break;
            case RotationType::Z:
                axisStr = "Z";
                break;
        }
        Logger::log("Hybrid Gate: " + axisStr + ", Angle: " + std::to_string(hg.angleGate.angle) + 
                      ", Amplitude: " + std::to_string(hg.amplitude), INFO);
    }
}