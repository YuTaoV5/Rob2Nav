#include "policy.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        as2w::Policy recovery(argv[1]), stairs(argv[2]);
        as2w::Controller control(recovery), blind(stairs);
        as2w::State state;
        state.body_to_world_wxyz[0] = 1.f;
        for (std::size_t i = 4; i < 8; ++i) state.q[i] = .8f;
        for (std::size_t i = 8; i < 12; ++i) state.q[i] = -1.6f;
        check(control.step(state, 0.f)[0].kp == 0.f, "A disarmed controller produced gains");
        control.arm();
        control.step(state, 0.f);
        check(!control.armed(), "Recovery accepted missing odometry");
        state.velocity_valid = true;
        control.arm();
        for (int i = 0; i < 100; ++i) {
            check(control.step(state, 0.f)[0].kp == 0.f, "Recovery enabled too early");
        }
        const auto command = control.step(state, 0.f);
        check(command[0].kp == 55.f && command[12].kp == 0.f && command[12].kd == 2.f,
              "Leg and wheel gain mapping differed");
        check(control.step(state, .051f)[0].kp == 0.f && !control.armed(), "Watchdog failed");
        state.velocity_valid = false;
        blind.arm();
        check(blind.step(state, 0.f)[0].kp == 55.f, "Blind policy unexpectedly required odometry");
        state.gyro[0] = std::numeric_limits<float>::quiet_NaN();
        check(blind.step(state, 0.f)[0].kp == 0.f && !blind.armed(), "Nonfinite input watchdog failed");
        as2w::Mapping mapping{};
        for (std::size_t i = 0; i < mapping.size(); ++i) mapping[i] = {19 - i, -1.f, .25f};
        const auto encoded = as2w::encodeCommands(command, mapping, 20);
        check(encoded[19].q == .25f - command[0].q && encoded[7].dq == -command[12].dq &&
              encoded[0].kp == 0.f, "Motor-slot or sign mapping differed");
        const auto decoded = as2w::decodeEncoders(state, mapping, std::vector<float>(20, .75f),
                                                   std::vector<float>(20, 1.f));
        check(decoded.q[0] == -.5f && decoded.dq[15] == -1.f, "Encoder mapping differed");
        std::cout << "Arm, timing, odometry, watchdog and motor mapping verified\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
