#ifndef HYBRID_ENCODER_HPP
#define HYBRID_ENCODER_HPP

#include "AngleEncoder.hpp"
#include <vector>

struct HybridGate {
    RotationGate angleGate;
    double amplitude; // Amplitude for the hybrid gate
};

class HybridEncoder {
public:
    HybridEncoder(const std::vector<double>& input);
    // get both angle + amplitude data
    std::vector<HybridGate> getHybridEncodedGates() const;
    void printHybridGates() const;

private:
    std::vector<HybridGate> hybridEncodedGates;
    void normaliseAndEncode(const std::vector<double>& input);
};

#endif // HYBRID_ENCODER_HPP
