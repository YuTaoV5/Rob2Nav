#include "r152_controller.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <yaml-cpp/yaml.h>

namespace as2w_r152 {
namespace {
template <class Range>
bool finite(const Range& values) {
    return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}

template <class T, std::size_t N>
std::array<T, N> read_array(const YAML::Node& node) {
    if (!node.IsSequence() || node.size() != N) {
        throw std::runtime_error("Unexpected deploy.yaml array length");
    }
    std::array<T, N> result;
    for (std::size_t i = 0; i < N; ++i) result[i] = node[i].as<T>();
    return result;
}

std::string checked_model(const std::filesystem::path& path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "as2w_r152_contract");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    Ort::Session session(env, path.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    if (session.GetInputCount() != 2 || session.GetOutputCount() != 2) {
        throw std::runtime_error("R41 ONNX must expose obs/state and actions/next_state");
    }
    for (std::size_t i = 0; i < 2; ++i) {
        const auto input = session.GetInputNameAllocated(i, allocator);
        const auto output = session.GetOutputNameAllocated(i, allocator);
        const auto input_type = session.GetInputTypeInfo(i);
        const auto output_type = session.GetOutputTypeInfo(i);
        const auto in = input_type.GetTensorTypeAndShapeInfo();
        const auto out = output_type.GetTensorTypeAndShapeInfo();
        const std::vector<int64_t> input_shape{1, static_cast<int64_t>(i == 0 ? ObservationCount : StateCount)};
        const std::vector<int64_t> output_shape{1, static_cast<int64_t>(i == 0 ? MotorCount : StateCount)};
        if (std::string(input.get()) != (i == 0 ? "obs" : "state")
            || std::string(output.get()) != (i == 0 ? "actions" : "next_state")
            || in.GetShape() != input_shape || out.GetShape() != output_shape
            || in.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            || out.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error("R41 ONNX tensor contract mismatch");
        }
    }
    return path.string();
}
}  // namespace

Config load_config(const std::filesystem::path& path) {
    const auto root = YAML::LoadFile(path.string());
    Config config;
    config.onnx = std::filesystem::absolute(path.parent_path() / root["policy_file"].as<std::string>());
    config.joint_names = read_array<std::string, MotorCount>(root["joint_names"]);
    config.defaults = read_array<float, MotorCount>(root["default_joint_pos"]);
    config.lower = read_array<float, MotorCount>(root["position_lower"]);
    config.upper = read_array<float, MotorCount>(root["position_upper"]);
    config.velocity = read_array<float, MotorCount>(root["velocity_limit"]);
    config.command_limit = read_array<float, 3>(root["command_limit"]);
    config.dt = root["step_dt"].as<float>();
    if (!std::isfinite(config.dt) || std::abs(config.dt - .02f) > 1e-6f
        || !finite(config.defaults) || !finite(config.lower) || !finite(config.upper)
        || !finite(config.velocity) || !finite(config.command_limit)) {
        throw std::runtime_error("Invalid AS2-W R41 control contract");
    }
    for (std::size_t i = 0; i < MotorCount; ++i) {
        if (config.joint_names[i].empty() || config.lower[i] > config.upper[i] || config.velocity[i] <= 0.f) {
            throw std::runtime_error("Invalid AS2-W motor limits or joint name");
        }
    }
    if (std::any_of(config.command_limit.begin(), config.command_limit.end(), [](float limit) { return limit <= 0.f; })) {
        throw std::runtime_error("Invalid command limits");
    }
    return config;
}

Observation make_observation(const SensorState& state, const Vector3& command) {
    const auto& quaternion = state.body_to_world_wxyz;
    if (!finite(quaternion) || !finite(state.gyro) || !finite(state.position) || !finite(state.velocity) || !finite(command)) {
        throw std::invalid_argument("Nonfinite sensor or command");
    }
    const float norm2 = std::inner_product(quaternion.begin(), quaternion.end(), quaternion.begin(), 0.f);
    if (norm2 < .81f || norm2 > 1.21f) throw std::invalid_argument("Invalid IMU quaternion");
    const float inv_norm = 1.f / std::sqrt(norm2);
    const float w = quaternion[0] * inv_norm, x = quaternion[1] * inv_norm;
    const float y = quaternion[2] * inv_norm, z = quaternion[3] * inv_norm;
    Observation observation{};
    std::copy(command.begin(), command.end(), observation.begin());
    std::copy(state.gyro.begin(), state.gyro.end(), observation.begin() + 3);
    observation[6] = -2.f * (x*z - w*y);
    observation[7] = -2.f * (y*z + w*x);
    observation[8] = 2.f * (x*x + y*y) - 1.f;
    std::copy(state.position.begin(), state.position.end(), observation.begin() + 9);
    std::copy(state.velocity.begin(), state.velocity.end(), observation.begin() + 25);
    return observation;
}

Controller::Controller(Config config)
    : config_(std::move(config)), runner_(checked_model(config_.onnx)), memory_(StateCount, 0.f) {}

void Controller::arm() {
    std::fill(memory_.begin(), memory_.end(), 0.f);
    armed_ = true;
}

void Controller::stop() {
    armed_ = false;
    std::fill(memory_.begin(), memory_.end(), 0.f);
}

Targets Controller::step(const SensorState& sensors, const Vector3& command, float age_seconds) {
    if (!armed_) return {};
    if (!std::isfinite(age_seconds) || age_seconds < 0.f || age_seconds > .05f) {
        stop();
        return {};
    }
    try {
        return replay(make_observation(sensors, command));
    } catch (const std::invalid_argument&) {
        stop();
        return {};
    }
}

Targets Controller::replay(const Observation& observation) {
    if (!armed_) return {};
    const float gravity2 = std::inner_product(observation.begin() + 6, observation.begin() + 9,
                                             observation.begin() + 6, 0.f);
    if (!finite(observation) || gravity2 < .81f || gravity2 > 1.21f) {
        stop();
        return {};
    }
    auto values = std::vector<float>(observation.begin(), observation.end());
    for (std::size_t i = 0; i < 3; ++i) {
        values[i] = std::clamp(values[i], -config_.command_limit[i], config_.command_limit[i]);
    }
    try {
        auto result = runner_.forward({{"obs", std::move(values)}, {"state", memory_}});
        const auto& action = result.at("actions");
        const auto& next = result.at("next_state");
        if (action.size() != MotorCount || next.size() != StateCount || !finite(action) || !finite(next)) {
            throw std::runtime_error("Invalid R41 inference output");
        }
        Targets targets;
        for (std::size_t i = 0; i < MotorCount; ++i) {
            targets.action[i] = action[i];
            if (i < 12) {
                const float q = std::clamp(config_.defaults[i] + .35f * action[i], config_.lower[i], config_.upper[i]);
                targets.motors[i] = {q, 0.f, 55.f, 2.f};
            } else {
                const float dq = std::clamp(16.f * action[i], -config_.velocity[i], config_.velocity[i]);
                targets.motors[i] = {0.f, dq, 0.f, 2.f};
            }
        }
        memory_ = std::move(result.at("next_state"));
        targets.active = true;
        return targets;
    } catch (...) {
        stop();
        throw;
    }
}
}  // namespace as2w_r152
