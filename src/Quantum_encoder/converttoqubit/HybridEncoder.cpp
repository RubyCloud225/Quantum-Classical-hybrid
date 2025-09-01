#include "HybridEncoder.hpp"
#include <cmath>
#include "utils/logger.hpp"
#include <iostream>
#include <omp.h>

HybridEncoder::HybridEncoder(const std::vector<double>& input) {
    normaliseAndEncode(input);
}
void HybridEncoder::normaliseAndEncode(const std::vector<double>& input) {
    double norm = 0.0;
    #pragma omp parallel for reduction(+:norm)
    for (size_t i = 0; i < input.size(); ++i) norm += input[i] * input[i];
    norm = std::sqrt(norm);

    std::vector<double> normalized_input(input.size());
    #pragma omp parallel for
    for (size_t i = 0; i < input.size(); ++i) {
        normalized_input[i] = input[i] / norm;
    }
    AngleEncoder angleEncoder(normalized_input, RotationAxis::Y);
    auto angleGates = angleEncoder.get_gates();
    hybridEncodedGates.resize(angleGates.size());
    #pragma omp parallel for
    for (size_t i = 0; i < angleGates.size(); ++i) {
        hybridEncodedGates[i] = {angleGates[i], normalized_input[i]};
    }
    Logger::log("HybridEncoder encoded " + std::to_string(hybridEncodedGates.size()) + " gates", LogLevel::INFO, __FILE__, __LINE__);
}

std::vector<HybridGate> HybridEncoder::getHybridEncodedGates() const {
    Logger::log("Retrieving hybrid encoded gates", LogLevel::INFO, __FILE__, __LINE__);
    return hybridEncodedGates;
}

void HybridEncoder::printHybridGates() const {
    for (const auto& hg : hybridEncodedGates) {
        std::string axisStr;
        switch (hg.angleGate.type) {
            case RotationAxis::X:
                axisStr = "X";
                break;
            case RotationAxis::Y:
                axisStr = "Y";
                break;
            case RotationAxis::Z:
                axisStr = "Z";
                break;
        }
        Logger::log("Hybrid Gate: " + axisStr + ", Angle: " + std::to_string(hg.angleGate.angle) + 
                      ", Amplitude: " + std::to_string(hg.amplitude), LogLevel::INFO, __FILE__, __LINE__);
    }
}