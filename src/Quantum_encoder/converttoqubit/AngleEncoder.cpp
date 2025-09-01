#include "AngleEncoder.hpp"
#include <cmath>
#include <iostream>
#include "utils/logger.hpp"
#include <omp.h>

AngleEncoder::AngleEncoder(const std::vector<double>& input, RotationAxis axis)
    : data(input), axis(axis) {
    encode();
}

void AngleEncoder::encode() {
    encoded.resize(data.size());
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        double theta = data[i] * M_PI; // Scale input to [0, π]
        encoded[i] = RotationGate{axis, theta};
    }
    Logger::log("AngleEncoder encoded " + std::to_string(data.size()) + " values", LogLevel::INFO, __FILE__, __LINE__);
}

std::vector<RotationGate> AngleEncoder::get_gates() const {
    Logger::log("Retrieving encoded gates", LogLevel::INFO, __FILE__, __LINE__);
    return encoded;
}

void AngleEncoder::printGates() const {
    for (const auto& gate : encoded) {
        std::string axisStr;
        switch (gate.type) {
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
        Logger::log("Gate: " + axisStr + ", Angle: " + std::to_string(gate.angle), LogLevel::INFO, __FILE__, __LINE__);
    }
}