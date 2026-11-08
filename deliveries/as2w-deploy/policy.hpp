#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace as2w {
constexpr std::size_t JointCount = 16;
constexpr std::size_t LegCount = 12;
constexpr float PolicyPeriod = .02f;
using Joints = std::array<float, JointCount>;
using Observation = std::array<float, 58>;
using Vec3 = std::array<float, 3>;

struct Targets {
    Joints action{};
    Joints q{};
    Joints dq{};
    bool active = false;
};

class Policy {
public:
    explicit Policy(const std::string& path);
    Joints actor(const Observation& observation) const;
    Targets infer(Observation sensors, float elapsed) const;
    bool stairs() const { return stairs_; }
    bool validSensors(const Joints& q, const Joints& dq) const;

private:
    struct Layer {
        std::size_t input = 0;
        std::size_t output = 0;
        std::vector<float> weights;
        std::vector<float> bias;
    };
    bool stairs_ = false;
    Joints defaults_{};
    std::array<Vec3, JointCount> limits_{};
    std::array<Layer, 4> layers_{};
};

struct State {
    Vec3 gyro{};
    std::array<float, 4> body_to_world_wxyz{};
    Vec3 body_velocity{};
    bool velocity_valid = false;
    Joints q{};
    Joints dq{};
};

struct MotorCommand {
    float q = 0.f, dq = 0.f, kp = 0.f, kd = 0.f, tau = 0.f;
};
using CommandFrame = std::array<MotorCommand, JointCount>;

class Controller {
public:
    explicit Controller(const Policy& policy) : policy_(policy) {}
    void arm();
    void stop() { armed_ = false; }
    CommandFrame step(const State& state, float age_s);
    bool armed() const { return armed_; }

private:
    const Policy& policy_;
    Joints previous_{};
    unsigned ticks_ = 0;
    bool armed_ = false;
};

struct MotorBinding {
    std::size_t index = 0;
    float sign = 1.f, offset = 0.f;
};
using Mapping = std::array<MotorBinding, JointCount>;
Mapping loadMapping(const std::string& path, std::size_t slots);
State decodeEncoders(State state, const Mapping& mapping, const std::vector<float>& q,
                     const std::vector<float>& dq);
std::vector<MotorCommand> encodeCommands(const CommandFrame& commands, const Mapping& mapping,
                                        std::size_t slots);
} // namespace as2w
