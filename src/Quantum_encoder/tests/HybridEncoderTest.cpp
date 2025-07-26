#include "HybridEncoder.hpp"
#include <cassert>
#include "utils/logger.hpp"
#include <iostream>
#include <cmath>

void testHybridEncoding() {
    std::vector<double> input = {0.5, 0.5, 0.5, 0.5};
    HybridEncoder encoder(input);
    auto hybrid = encoder.getHybridEncodedGates();

    // check vector size matches input
    assert(hybrid.size() == input.size());
    Logger::log("Hybrid encoding test: vector size matches input", INFO);

    // check that each hybrid gate has a valid angle and amplitude
    double sumSq = 0.0;
    for (const auto& hg : hybrid) {
        assert(hg.amplitude >= 0.0 && hg.amplitude <= 1.0);
        sumSq += hg.amplitude * hg.amplitude;
        assert(hg.angleGate.angle >= 0.0 && hg.angleGate.angle <= M_PI);
        Logger::log("Hybrid encoding test: valid angle and amplitude for gate", INFO);
    }
    double norm = std::sqrt(sumSq);
    assert(std::abs(norm - 1.0) < 1e-6); // Check normalization

    // check angle = amplitude * M_PI
    for (const auto& hg : hybrid) {
        double expected_angle = hg.amplitude * M_PI;
        assert(std::abs(hg.angleGate.angle - expected_angle) < 1e-6);
        assert(hg.angleGate.axis == RotationAxis::Y); // Check axis is Y
        Logger::log("Hybrid encoding test: angle matches expected value", INFO);
    }
    Logger::log("Hybrid encoding test passed successfully", INFO);
}

int main() {
    testHybridEncoding();
    return 0;
}