#include "policy.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
template<class T> void read(std::istream& stream, T& values) {
    for (auto& value : values) {
        if (!(stream >> value) || !std::isfinite(value)) throw std::runtime_error("Invalid input row");
    }
}

void verify(const as2w::Policy& policy, const char* fixture) {
    std::ifstream input(fixture);
    float elapsed = 0.f, max_error = 0.f;
    unsigned frames = 0;
    while (input >> elapsed) {
        as2w::Observation obs{};
        as2w::Joints action{};
        std::array<float, 12> leg{};
        std::array<float, 4> wheel{};
        read(input, obs);
        read(input, action);
        read(input, leg);
        read(input, wheel);
        int active = -1;
        if (!(input >> active) || (active != 0 && active != 1)) throw std::runtime_error("Invalid active flag");
        const auto got = policy.infer(obs, elapsed);
        if (got.active != bool(active)) throw std::runtime_error("Activation mismatch");
        for (std::size_t i = 0; i < action.size(); ++i) {
            const float expected = i < leg.size() ? leg[i] : wheel[i - leg.size()];
            const float target = i < leg.size() ? got.q[i] : got.dq[i];
            max_error = std::max({max_error, std::abs(got.action[i] - action[i]), std::abs(target - expected)});
        }
        ++frames;
    }
    if (!input.eof() || frames != 60 || max_error > .0004f) throw std::runtime_error("Reference trace mismatch");
    std::cout << "frames=" << frames << " max_abs_error=" << max_error << '\n';
}

void actorStream(const as2w::Policy& policy) {
    std::cout << std::setprecision(9);
    as2w::Observation observation{};
    while (std::cin >> observation.front()) {
        for (std::size_t i = 1; i < observation.size(); ++i) {
            if (!(std::cin >> observation[i])) throw std::runtime_error("Incomplete actor observation");
        }
        const auto result = policy.actor(observation);
        for (float value : result) std::cout << value << ' ';
        std::cout << std::endl;
    }
    if (!std::cin.eof()) throw std::runtime_error("Invalid actor stream");
}

void targetStream(const as2w::Policy& policy) {
    std::cout << std::setprecision(9);
    float elapsed;
    while (std::cin >> elapsed) {
        as2w::Observation sensors{};
        read(std::cin, sensors);
        const auto targets = policy.infer(sensors, elapsed);
        for (float value : targets.action) std::cout << value << ' ';
        for (std::size_t i = 0; i < as2w::LegCount; ++i) std::cout << targets.q[i] << ' ';
        for (std::size_t i = as2w::LegCount; i < as2w::JointCount; ++i) std::cout << targets.dq[i] << ' ';
        std::cout << int(targets.active) << std::endl;
    }
    if (!std::cin.eof()) throw std::runtime_error("Invalid target stream");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "verify") {
            verify(as2w::Policy(argv[2]), argv[3]);
        } else if (argc == 3 && std::string(argv[1]) == "actor-stream") {
            actorStream(as2w::Policy(argv[2]));
        } else if (argc == 3 && std::string(argv[1]) == "target-stream") {
            targetStream(as2w::Policy(argv[2]));
        } else {
            std::cerr << "Usage: as2w_deploy verify MODEL FIXTURE | actor-stream MODEL | target-stream MODEL\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
