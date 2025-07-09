#include "AngleEncoder.hpp"
#include <cmath>
#include <iostream>

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
}

std::vector<RotationGate> AngleEncoder::get_gates() const {
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
        std::cout << "Rotation Gate: " << axisStr << ", Angle: " << gate.angle << " radians" << std::endl;
    }
}