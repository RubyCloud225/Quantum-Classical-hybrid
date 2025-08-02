#include "../BetaSchedule.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdio> // for remove()

int total_epochs = 10;
double initial_beta = 0.1;
double current_beta = 0.2;
double final_beta = 0.5;

void testBetaSchedule() {
    BetaSchedule beta_schedule(total_epochs, initial_beta);
    std::vector<double> nll_losses = {0.1, 0.2, 0.3};
    std::vector<double> entropy_losses = {0.1, 0.2, 0.3};

    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        double beta = beta_schedule.update(nll_losses, entropy_losses, epoch);
        std::cout << "Epoch: " << epoch << ", Beta: " << beta << std::endl;
        assert(beta >= initial_beta);
        double current_beta = beta_schedule.getCurrentBeta();
        double initial_beta = beta_schedule.getInitialBeta();
        int num_epochs = beta_schedule.getNumEpochs();
    }

    std::cout << "BetaSchedule test passed." << std::endl;
}
int main() {
    testBetaSchedule();
    return 0;
}