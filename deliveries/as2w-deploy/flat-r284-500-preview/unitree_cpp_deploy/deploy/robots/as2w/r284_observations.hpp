#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace as2w_r284 {
using Vec3 = std::array<float, 3>;
using Motors = std::array<float, 16>;
using Frame = std::array<float, 60>;
using Features = std::array<float, 65>;
inline constexpr double Pi = 3.14159265358979323846;
inline constexpr double Dt = .02;
inline constexpr Motors Defaults{0, 0, 0, 0, .8f, .8f, .8f, .8f,
                                  -1.6f, -1.6f, -1.6f, -1.6f, 0, 0, 0, 0};
inline constexpr std::array<float, 12> Lower{-.785f, -.785f, -.785f, -.785f,
                                             -2.f, -2.f, -1.3f, -1.3f,
                                             -2.7677f, -2.7677f, -2.7677f, -2.7677f};
inline constexpr std::array<float, 12> Upper{.785f, .785f, .785f, .785f,
                                             3.492f, 3.492f, 4.19f, 4.19f,
                                             -.8133f, -.8133f, -.8133f, -.8133f};

inline double wrap_wheel(double angle) {
    double shifted = std::fmod(angle + 2. * Pi, 4. * Pi);
    if (shifted < 0.) shifted += 4. * Pi;
    return shifted - 2. * Pi;
}

class WheelMemory {
public:
    void reset() { was_zero_ = false; }

    Features push(const Frame& frame) {
        Features features{};
        std::copy(frame.begin(), frame.end(), features.begin());
        const bool zero = frame[0] == -.4f && frame[1] == 0.f && frame[2] == 0.f;
        if (zero) {
            if (!was_zero_) std::copy_n(frame.begin() + 22, 4, reference_.begin());
            for (std::size_t i = 0; i < 4; ++i) {
                const double delta = wrap_wheel(frame[22 + i] - reference_[i]);
                features[60 + i] = static_cast<float>(std::clamp(delta / .20, -3., 3.));
            }
            features[64] = 1.f;
        }
        was_zero_ = zero;
        return features;
    }

private:
    std::array<float, 4> reference_{};
    bool was_zero_ = false;
};

struct SensorFrame {
    Vec3 gyro_body{};
    Vec3 gravity_body{};
    Motors joint_q{};
    Motors joint_dq{};
};

class Observations {
public:
    void reset() {
        command_.fill(0.);
        previous_.fill(0.f);
        tick_ = 0;
        memory_.reset();
    }

    Features push(const SensorFrame& sensors, const Vec3& joystick) {
        std::array<double, 3> goal{joystick[0], joystick[1], joystick[2]};
        const double planar = std::hypot(goal[0], goal[1]);
        if (planar > .10) {
            goal[0] *= .10 / planar;
            goal[1] *= .10 / planar;
        }
        goal[2] = std::clamp(goal[2], -.30, .30);
        for (std::size_t i = 0; i < 3; ++i) command_[i] += .12 * (goal[i] - command_[i]);
        const bool released = goal[0] == 0. && goal[1] == 0. && goal[2] == 0.;
        const double peak = std::max({std::abs(command_[0]), std::abs(command_[1]), std::abs(command_[2])});
        if (released && peak < .005) command_.fill(0.);

        Frame frame{};
        frame[0] = static_cast<float>(command_[0] - .4);
        frame[1] = static_cast<float>(command_[1]);
        frame[2] = static_cast<float>(command_[2]);
        std::copy(sensors.gyro_body.begin(), sensors.gyro_body.end(), frame.begin() + 3);
        std::copy(sensors.gravity_body.begin(), sensors.gravity_body.end(), frame.begin() + 6);
        frame[9] = .4f;
        for (std::size_t i = 0; i < Defaults.size(); ++i) {
            frame[10 + i] = i < 12 ? sensors.joint_q[i] - Defaults[i]
                                   : static_cast<float>(wrap_wheel(sensors.joint_q[i]));
            frame[26 + i] = sensors.joint_dq[i];
            frame[42 + i] = previous_[i];
        }
        const double phase = 2. * Pi * static_cast<double>(tick_++) * Dt / .62;
        frame[58] = static_cast<float>(std::sin(phase));
        frame[59] = static_cast<float>(std::cos(phase));
        return memory_.push(frame);
    }

    void remember(const Motors& actions) {
        for (std::size_t i = 0; i < actions.size(); ++i)
            previous_[i] = std::clamp(actions[i], -6.f, 6.f);
    }

private:
    std::array<double, 3> command_{};
    Motors previous_{};
    std::uint64_t tick_ = 0;
    WheelMemory memory_;
};

struct MotorTargets {
    Motors q{}, dq{}, kp{}, kd{};
};

inline MotorTargets targets(const Motors& actions) {
    MotorTargets out{};
    out.kd.fill(2.f);
    for (std::size_t i = 0; i < actions.size(); ++i) {
        const float action = std::clamp(actions[i], -6.f, 6.f);
        if (i < 12) {
            out.q[i] = std::clamp(Defaults[i] + .35f * action, Lower[i], Upper[i]);
            out.kp[i] = 55.f;
        } else {
            out.dq[i] = std::clamp(16.f * action, -98.47f, 98.47f);
        }
    }
    return out;
}
}  // namespace as2w_r284
