#include "HybridEncoder.hpp"
#include <cmath>
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
}

std::vector<HybridGate> HybridEncoder::getHybridEncodedGates() const {
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
        std::cout << "Hybrid Gate: " << axisStr 
                  << ", Angle: " << hg.angleGate.angle 
                  << " radians, Amplitude: " << hg.amplitude << std::endl;
    }
}