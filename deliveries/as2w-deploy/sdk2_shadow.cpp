#include "policy.hpp"

#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
using unitree::robot::ChannelSubscriber;
using unitree_go::msg::dds_::LowState_;
using unitree_go::msg::dds_::SportModeState_;

void sport(const char* nic) {
    unitree::robot::ChannelFactory::Instance()->Init(0, nic);
    ChannelSubscriber<SportModeState_> subscriber("rt/lf/sportmodestate");
    subscriber.InitChannel([](const void* message) {
        const auto& frame = *static_cast<const SportModeState_*>(message);
        std::cout << "sport_mode=" << unsigned(frame.mode()) << " velocity="
                  << frame.velocity()[0] << ',' << frame.velocity()[1] << ',' << frame.velocity()[2] << '\n';
    }, 1);
    std::this_thread::sleep_for(std::chrono::seconds(10));
}

void low(const char* nic, const char* topic, const char* model, const char* calibration) {
    as2w::Policy policy(model);
    if (!policy.stairs()) throw std::runtime_error("Recovery shadow requires a verified body-velocity source");
    const auto mapping = as2w::loadMapping(calibration, LowState_{}.motor_state().size());
    unitree::robot::ChannelFactory::Instance()->Init(0, nic);
    std::mutex lock;
    LowState_ latest{};
    Clock::time_point received{};
    ChannelSubscriber<LowState_> subscriber(topic);
    subscriber.InitChannel([&](const void* message) {
        const auto& frame = *static_cast<const LowState_*>(message);
        std::lock_guard<std::mutex> guard(lock);
        if (received != Clock::time_point{} && latest.tick() == frame.tick()) return;
        latest = frame;
        received = Clock::now();
    }, 1);
    as2w::Controller controller(policy);
    unsigned count = 0;
    auto next = Clock::now();
    for (unsigned poll = 0; poll < 500; ++poll) {
        next += std::chrono::milliseconds(20);
        std::this_thread::sleep_until(next);
        LowState_ state;
        Clock::time_point stamp;
        {
            std::lock_guard<std::mutex> guard(lock);
            state = latest;
            stamp = received;
        }
        if (stamp == Clock::time_point{}) continue;
        as2w::State sample;
        const auto& imu = state.imu_state();
        std::copy(imu.gyroscope().begin(), imu.gyroscope().end(), sample.gyro.begin());
        std::copy(imu.quaternion().begin(), imu.quaternion().end(), sample.body_to_world_wxyz.begin());
        std::vector<float> q, dq;
        for (const auto& motor : state.motor_state()) {
            q.push_back(motor.q());
            dq.push_back(motor.dq());
        }
        if (count++ == 0) controller.arm();
        sample = as2w::decodeEncoders(sample, mapping, q, dq);
        const auto commands = controller.step(sample, std::chrono::duration<float>(Clock::now() - stamp).count());
        const auto motors = as2w::encodeCommands(commands, mapping, q.size());
        if (count % 25 == 1) {
            const auto& motor = motors.at(mapping.back().index);
            std::cout << "shadow_only tick=" << state.tick() << " active=" << controller.armed()
                      << " RR_foot_dq=" << motor.dq << '\n';
        }
        if (!controller.armed()) break;
    }
    if (count == 0) throw std::runtime_error("No compatible LowState packets received");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "sport") {
            sport(argv[2]);
        } else if (argc == 7 && std::string(argv[1]) == "low" &&
                   std::string(argv[6]) == "--imu-wxyz-body-to-world") {
            low(argv[2], argv[3], argv[4], argv[5]);
        } else {
            std::cerr << "Read only: as2w_sdk2_shadow sport NIC\n"
                         "Read only: as2w_sdk2_shadow low NIC TOPIC MODEL MAP --imu-wxyz-body-to-world\n";
            return 2;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
