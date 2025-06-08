#ifndef LAYERNOMALIZATION_HPP
#define LAYERNOMALIZATION_HPP

#include <vector>

class LayerNormalization {
    public: 
    LayerNormalization(int features, double epsilon = 1e-5);
    void resetParameters();
    std::vector<double> forward(const std::vector<double>& input);
    const std::vector<double>& getGamma() const { return gamma; }
    const std::vector<double>& getBeta() const { return beta; }

    private:
    int normal_shape;
    double epsilon;
    std::vector<double> gamma;
    std::vector<double> beta;
};

#endif // LAYERNOMALIZATION_HPP