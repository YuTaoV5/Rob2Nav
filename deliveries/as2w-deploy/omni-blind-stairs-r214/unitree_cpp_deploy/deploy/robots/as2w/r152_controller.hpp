#pragma once

#include <array>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "isaaclab/algorithms/algorithms.h"

namespace as2w_r152 {
inline constexpr std::size_t MotorCount = 16;
inline constexpr std::size_t ObservationCount = 41;
inline constexpr std::size_t StateCount = 1196;
using Vector3 = std::array<float, 3>;
using Motors = std::array<float, MotorCount>;
using Observation = std::array<float, ObservationCount>;

struct SensorState {
    Vector3 gyro{};
    std::array<float, 4> body_to_world_wxyz{};
    Motors position{};
    Motors velocity{};
};

struct MotorTarget {
    float q = 0, dq = 0, kp = 0, kd = 0;
};

struct Targets {
    std::array<MotorTarget, MotorCount> motors{};
    Motors action{};
    bool active = false;
};

struct Config {
    std::filesystem::path onnx;
    std::array<std::string, MotorCount> joint_names;
    Motors defaults{}, lower{}, upper{}, velocity{};
    Vector3 command_limit{};
    float dt = 0;
};

Config load_config(const std::filesystem::path& yaml);
Observation make_observation(const SensorState& sensors, const Vector3& command);

class Controller {
public:
    explicit Controller(Config config);
    void arm();
    void stop();
    bool armed() const { return armed_; }
    Targets step(const SensorState& sensors, const Vector3& command, float age_seconds);
    Targets replay(const Observation& observation);

private:
    Config config_;
    isaaclab::OrtRunner runner_;
    std::vector<float> memory_;
    bool armed_ = false;
};
}  // namespace as2w_r152
