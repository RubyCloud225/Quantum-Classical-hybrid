#include "AngleEncoder.hpp"
#include <cmath>
#include <iostream>
#include "utils/logger.hpp"

AngleEncoder::AngleEncoder(const std::vector<double>& input, RotationAxis axis)
    : data(input), axis(axis) {
    encode();
}

void AngleEncoder::encode() {
    encoded.clear();
    for (double val : data) {
        double theta = val * M_PI; // Scale input to [0, π]
        encoded.push_back({axis, theta});
    }
    Logger::log("AngleEncoder encoded " + std::to_string(data.size()) + " values", LogLevel::INFO, __FILE__, __LINE__);
}

std::vector<RotationGate> AngleEncoder::get_gates() const {
    Logger::log("Retrieving encoded gates", LogLevel::INFO, __FILE__, __LINE__);
    return encoded;
}

void AngelEncoder::printGates() const {
    for (const auto& gate : encoded) {
        std::string axisStr;
        switch (gate.type) {
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
        Logger::log("Gate: " + axisStr + ", Angle: " + std::to_string(gate.angle), LogLevel::INFO, __FILE__, __LINE__);
    }
}