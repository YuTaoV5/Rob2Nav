#include "policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace as2w {
namespace {
constexpr float Pi = 3.14159265358979323846f;
constexpr std::array<std::size_t, 5> Widths{{58, 128, 128, 128, 16}};
constexpr std::array<const char*, 16> Names{{
    "FL_hip", "FR_hip", "RL_hip", "RR_hip", "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf", "FL_foot", "FR_foot", "RL_foot", "RR_foot"
}};

std::uint32_t readWord(std::istream& source) {
    unsigned char b[4];
    if (!source.read(reinterpret_cast<char*>(b), 4)) {
        throw std::runtime_error("Truncated actor file");
    }
    return std::uint32_t(b[0]) | (std::uint32_t(b[1]) << 8) |
           (std::uint32_t(b[2]) << 16) | (std::uint32_t(b[3]) << 24);
}

template<class Values> bool finite(const Values& values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
}

template<class Values> void readFloats(std::istream& source, Values& values) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "IEEE float32 required");
    for (float& value : values) {
        const auto bits = readWord(source);
        std::memcpy(&value, &bits, sizeof(value));
    }
    if (!finite(values)) {
        throw std::runtime_error("Nonfinite actor parameter");
    }
}
} // namespace

Policy::Policy(const std::string& path) {
    std::ifstream source(path, std::ios::binary);
    std::array<char, 8> magic{};
    if (!source.read(magic.data(), magic.size()) ||
        std::string(magic.data(), magic.size()) != "AS2WMLP1") {
        throw std::runtime_error("Invalid AS2-W actor: " + path);
    }
    const auto mode = readWord(source);
    if (mode > 1) throw std::runtime_error("Unknown policy mode");
    stairs_ = mode == 1;
    readFloats(source, defaults_);
    for (std::size_t i = 0; i < JointCount; ++i) {
        auto& limit = limits_[i];
        readFloats(source, limit);
        if (limit[2] <= 0.f || (i < LegCount &&
            (limit[0] >= limit[1] || defaults_[i] < limit[0] || defaults_[i] > limit[1]))) {
            throw std::runtime_error("Invalid URDF limits in actor file");
        }
    }
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        auto& layer = layers_[i];
        layer.input = Widths[i];
        layer.output = Widths[i + 1];
        layer.weights.resize(layer.input * layer.output);
        layer.bias.resize(layer.output);
        readFloats(source, layer.weights);
        readFloats(source, layer.bias);
    }
    if (source.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Trailing actor data");
}

Joints Policy::actor(const Observation& obs) const {
    if (!finite(obs)) throw std::invalid_argument("Nonfinite observation");
    std::array<float, 128> input{}, output{};
    std::copy(obs.begin(), obs.end(), input.begin());
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        for (std::size_t j = 0; j < layer.output; ++j) {
            float value = layer.bias[j];
            for (std::size_t k = 0; k < layer.input; ++k) {
                value += layer.weights[j * layer.input + k] * input[k];
            }
            output[j] = i < 3 && value < 0.f ? std::expm1(value) : value;
        }
        input.swap(output);
    }
    Joints result{};
    std::copy_n(input.begin(), result.size(), result.begin());
    if (!finite(result)) throw std::runtime_error("Nonfinite actor output");
    return result;
}

Targets Policy::infer(Observation obs, float elapsed) const {
    if (!std::isfinite(elapsed) || elapsed < 0.f) throw std::invalid_argument("Invalid episode time");
    if (stairs_) std::fill_n(obs.begin(), 3, 0.f);
    obs[9] = stairs_ ? .4f : std::clamp((elapsed - 2.f) / 4.f, 0.f, 1.f);
    for (std::size_t i = 0; i < JointCount; ++i) {
        auto& position = obs[10 + i];
        if (i >= LegCount) {
            position = std::fmod(position + 2.f * Pi, 4.f * Pi);
            if (position < 0.f) position += 4.f * Pi;
            position -= 2.f * Pi;
        }
        position -= defaults_[i];
    }
    Targets result;
    result.action = actor(obs);
    result.active = stairs_ || elapsed >= 2.f;
    for (std::size_t i = 0; i < JointCount; ++i) {
        auto& action = result.action[i];
        action = std::clamp(action, -6.f, 6.f);
        const auto& lim = limits_[i];
        if (i < LegCount) result.q[i] = std::clamp(defaults_[i] + .35f * action, lim[0], lim[1]);
        else result.dq[i] = std::clamp(16.f * action, -lim[2], lim[2]);
    }
    return result;
}

bool Policy::validSensors(const Joints& q, const Joints& dq) const {
    if (!finite(q) || !finite(dq)) return false;
    for (std::size_t i = 0; i < JointCount; ++i) {
        const auto& lim = limits_[i];
        if (std::abs(dq[i]) > 1.1f * lim[2] || (i < LegCount &&
            (q[i] < lim[0] - .1f || q[i] > lim[1] + .1f))) return false;
    }
    return true;
}

void Controller::arm() {
    previous_.fill(0.f);
    ticks_ = 0;
    armed_ = true;
}

CommandFrame Controller::step(const State& state, float age_s) {
    CommandFrame output{};
    if (!armed_) return output;
    const unsigned horizon = policy_.stairs() ? 450u : 300u;
    if (ticks_ >= horizon || !std::isfinite(age_s) || age_s < 0.f || age_s > .05f ||
        !policy_.validSensors(state.q, state.dq) || !finite(state.gyro) ||
        !finite(state.body_to_world_wxyz) ||
        (!policy_.stairs() && (!state.velocity_valid || !finite(state.body_velocity)))) {
        stop();
        return output;
    }
    auto q = state.body_to_world_wxyz;
    const float norm2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3];
    if (norm2 < .81f || norm2 > 1.21f) {
        stop();
        return output;
    }
    for (float& x : q) x /= std::sqrt(norm2);
    const float w = q[0], x = q[1], y = q[2], z = q[3];
    const Vec3 gravity{{2.f*(w*y - x*z), -2.f*(w*x + y*z), -1.f + 2.f*(x*x + y*y)}};
    Observation obs{};
    if (!policy_.stairs()) std::copy(state.body_velocity.begin(), state.body_velocity.end(), obs.begin());
    std::copy(state.gyro.begin(), state.gyro.end(), obs.begin() + 3);
    std::copy(gravity.begin(), gravity.end(), obs.begin() + 6);
    std::copy(state.q.begin(), state.q.end(), obs.begin() + 10);
    std::copy(state.dq.begin(), state.dq.end(), obs.begin() + 26);
    std::copy(previous_.begin(), previous_.end(), obs.begin() + 42);
    try {
        const auto target = policy_.infer(obs, float(ticks_++) * PolicyPeriod);
        previous_ = target.action;
        if (target.active) {
            for (std::size_t i = 0; i < JointCount; ++i) {
                output[i] = {target.q[i], target.dq[i], i < LegCount ? 55.f : 0.f, 2.f, 0.f};
            }
        }
    } catch (...) {
        stop();
        return {};
    }
    return output;
}

Mapping loadMapping(const std::string& path, std::size_t slots) {
    std::ifstream input(path);
    Mapping mapping{};
    std::vector<bool> used(slots, false);
    for (std::size_t i = 0; i < JointCount; ++i) {
        auto& entry = mapping[i];
        std::string name;
        if (!(input >> name >> entry.index >> entry.sign >> entry.offset) || name != Names[i] ||
            entry.index >= slots || used[entry.index] ||
            (entry.sign != 1.f && entry.sign != -1.f) || !std::isfinite(entry.offset)) {
            throw std::runtime_error("Invalid calibration row for " + std::string(Names[i]));
        }
        used[entry.index] = true;
    }
    std::string extra;
    if (input >> extra) throw std::runtime_error("Unexpected motor calibration data");
    return mapping;
}

State decodeEncoders(State state, const Mapping& mapping, const std::vector<float>& q,
                     const std::vector<float>& dq) {
    for (std::size_t i = 0; i < JointCount; ++i) {
        const auto& entry = mapping[i];
        state.q[i] = entry.sign * (q.at(entry.index) - entry.offset);
        state.dq[i] = entry.sign * dq.at(entry.index);
    }
    return state;
}

std::vector<MotorCommand> encodeCommands(const CommandFrame& commands, const Mapping& mapping,
                                        std::size_t slots) {
    std::vector<MotorCommand> output(slots);
    for (std::size_t i = 0; i < JointCount; ++i) {
        const auto& entry = mapping[i];
        const auto& cmd = commands[i];
        if (cmd.kp == 0.f && cmd.kd == 0.f) continue;
        output.at(entry.index) = {
            cmd.kp == 0.f ? 0.f : entry.sign * cmd.q + entry.offset,
            entry.sign * cmd.dq, cmd.kp, cmd.kd, entry.sign * cmd.tau
        };
    }
    return output;
}
} // namespace as2w
