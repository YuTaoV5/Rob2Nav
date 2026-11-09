#include "r152_controller.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: as2w_r152_replay deploy.yaml r152-trace.csv\n";
            return 2;
        }
        const auto config = as2w_r152::load_config(argv[1]);
        as2w_r152::Controller controller(config);
        std::ifstream stream(argv[2]);
        if (!stream) throw std::runtime_error("Cannot open the replay fixture");
        std::size_t samples = 0;
        int previous_episode = -1;
        float maximum = 0.f;
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream values(line);
            int episode = 0;
            as2w_r152::Observation observation{};
            as2w_r152::Motors expected{};
            values >> episode;
            for (auto& v : observation) values >> v;
            for (auto& v : expected) values >> v;
            if (!values) throw std::runtime_error("Invalid replay row");
            if (episode != previous_episode) controller.arm();
            previous_episode = episode;
            const auto out = controller.replay(observation);
            if (!out.active) throw std::runtime_error("R41 stopped during replay");
            for (std::size_t i = 0; i < expected.size(); ++i) {
                maximum = std::max(maximum, std::abs(out.action[i] - expected[i]));
                const auto motor = out.motors[i];
                if (i < 12) {
                    const float q = std::clamp(config.defaults[i] + .35f * out.action[i], config.lower[i], config.upper[i]);
                    if (motor.q != q || motor.dq != 0.f || motor.kp != 55.f || motor.kd != 2.f) {
                        throw std::runtime_error("Leg target mapping mismatch");
                    }
                } else {
                    const float dq = std::clamp(16.f * out.action[i], -config.velocity[i], config.velocity[i]);
                    if (motor.dq != dq || motor.q != 0.f || motor.kp != 0.f || motor.kd != 2.f) {
                        throw std::runtime_error("Wheel target mapping mismatch");
                    }
                }
            }
            ++samples;
        }
        as2w_r152::SensorState stale{};
        stale.body_to_world_wxyz = {1.f, 0.f, 0.f, 0.f};
        if (controller.step(stale, {}, .051f).active || controller.armed()) {
            throw std::runtime_error("Stale state did not disable inference");
        }
        std::cout << "rows=" << samples << " max_abs_error=" << maximum
                  << " leg_and_wheel_targets=verified stale_state=latched_off\n";
        return samples && maximum <= .001f ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
