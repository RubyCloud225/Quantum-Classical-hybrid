#include "HybridEncoder.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

void testHybridEncoding() {
    std::vector<double> input = {0.5, 0.5, 0.5, 0.5};
    HybridEncoder encoder(input);
    auto hybrid = encoder.getHybridEncodedGates();

    // check vector size matches input
    assert(hybrid.size() == input.size());

    // check that each hybrid gate has a valid angle and amplitude
    double sumSq = 0.0;
    for (const auto& hg : hybrid) {
        assert(hg.amplitude >= 0.0 && hg.amplitude <= 1.0);
        sumSq += hg.amplitude * hg.amplitude;
        assert(hg.angleGate.angle >= 0.0 && hg.angleGate.angle <= M_PI);
    }
    double norm = std::sqrt(sumSq);
    assert(std::abs(norm - 1.0) < 1e-6); // Check normalization

    // check angle = amplitude * M_PI
    for (const auto& hg : hybrid) {
        double expected_angle = hg.amplitude * M_PI;
        assert(std::abs(hg.angleGate.angle - expected_angle) < 1e-6);
        assert(hg.angleGate.axis == RotationAxis::Y); // Check axis is Y
    }
    std::cout << "Hybrid encoding test passed!" << std::endl;
}

int main() {
    testHybridEncoding();
    return 0;
}