#include <random>
#include <vector>
#include <complex>

void generate_random(std::vector<std::complex<float>>& data, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1, 1);
    for(auto& v: data) {
        float r = dist(gen);
        float i = dist(gen);
        v = {r, i};
    }
}

void generate_random(std::vector<float>& data, int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1, 1);
    for(auto& v: data) {
        v = dist(gen);
    }
}