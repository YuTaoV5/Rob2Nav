#include "r41_controller.hpp"

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
using LowState = unitree_go::msg::dds_::LowState_;

struct Axis {
    std::size_t index;
    float sign, zero;
};
using Mapping = std::array<Axis, as2w_r41::MotorCount>;

Mapping read_mapping(const as2w_r41::Config& config, const char* path) {
    std::ifstream file(path);
    std::set<std::size_t> used;
    Mapping map{};
    const auto slots = LowState{}.motor_state().size();
    for (std::size_t i = 0; i < map.size(); ++i) {
        std::string name;
        auto& axis = map[i];
        if (!(file >> name >> axis.index >> axis.sign >> axis.zero)
            || name != config.joint_names[i] || axis.index >= slots
            || !used.insert(axis.index).second || std::abs(axis.sign) != 1.f || !std::isfinite(axis.zero)) {
            throw std::runtime_error("Invalid or incomplete calibrated motor map");
        }
    }
    std::string extra;
    if (file >> extra) throw std::runtime_error("Unexpected trailing motor mapping");
    return map;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6 || std::string(argv[5]) != "--imu-wxyz-body-to-world") {
            std::cerr << "Read only: as2w_r41_shadow deploy.yaml NIC VERIFIED_LOWSTATE_TOPIC calibrated.map --imu-wxyz-body-to-world\n";
            return 2;
        }
        const auto config = as2w_r41::load_config(argv[1]);
        const auto mapping = read_mapping(config, argv[4]);
        as2w_r41::Controller controller(config);
        unitree::robot::ChannelFactory::Instance()->Init(0, argv[2]);
        std::mutex mutex;
        LowState latest{};
        Clock::time_point received{};
        unitree::robot::ChannelSubscriber<LowState> subscriber(argv[3]);
        subscriber.InitChannel([&](const void* payload) {
            const auto& frame = *static_cast<const LowState*>(payload);
            std::lock_guard<std::mutex> lock(mutex);
            if (received != Clock::time_point{} && frame.tick() == latest.tick()) return;
            latest = frame;
            received = Clock::now();
        }, 1);
        auto next = Clock::now();
        std::size_t samples = 0;
        for (int i = 0; i < 500; ++i) {
            next += std::chrono::milliseconds(20);
            std::this_thread::sleep_until(next);
            LowState frame;
            Clock::time_point stamp;
            {
                std::lock_guard<std::mutex> lock(mutex);
                frame = latest;
                stamp = received;
            }
            if (stamp == Clock::time_point{}) continue;
            as2w_r41::SensorState sensors;
            const auto& imu = frame.imu_state();
            std::copy(imu.gyroscope().begin(), imu.gyroscope().end(), sensors.gyro.begin());
            std::copy(imu.quaternion().begin(), imu.quaternion().end(), sensors.body_to_world_wxyz.begin());
            for (std::size_t axis = 0; axis < mapping.size(); ++axis) {
                const auto& binding = mapping[axis];
                const auto& motor = frame.motor_state().at(binding.index);
                sensors.position[axis] = binding.sign * (motor.q() - binding.zero);
                sensors.velocity[axis] = binding.sign * motor.dq();
            }
            if (!samples++) controller.arm();
            const auto targets = controller.step(sensors, {}, std::chrono::duration<float>(Clock::now()-stamp).count());
            if (!targets.active) throw std::runtime_error("Stale or invalid LowState sample");
            if (samples % 25 == 1) {
                std::cout << "shadow_only tick=" << frame.tick()
                          << " FL_hip_q=" << targets.motors[0].q
                          << " RR_foot_dq=" << targets.motors[15].dq << '\n';
            }
        }
        if (!samples) throw std::runtime_error("No compatible LowState packets received");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
