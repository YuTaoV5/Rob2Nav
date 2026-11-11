#include "r284_observations.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

void verify_observations() {
    using namespace as2w_r284;
    Observations observations;
    SensorFrame sensors;
    sensors.gravity_body[2] = -1.f;
    sensors.joint_q = Defaults;
    sensors.joint_q[12] = 6.26f;
    const auto initial = observations.push(sensors, {});
    if (initial[0] != -.4f || initial[8] != -1.f || initial[21] != 0.f ||
        initial[59] != 1.f || initial[60] != 0.f || initial[64] != 1.f)
        throw std::runtime_error("Unexpected initial observation");
    sensors.joint_q[12] = 6.30f;
    const auto across_wrap = observations.push(sensors, {});
    if (std::abs(across_wrap[60] - .2f) > 1e-4f)
        throw std::runtime_error("Wheel reference did not survive the wrap");
    Motors previous{};
    previous[0] = 8.f;
    previous[12] = 1.f;
    observations.remember(previous);
    const auto moving = observations.push(sensors, {1.f, 0.f, 0.f});
    if (std::abs(moving[0] - -.388f) > 1e-6f || moving[42] != 6.f ||
        moving[54] != 1.f || moving[60] != 0.f || moving[64] != 0.f)
        throw std::runtime_error("Command filtering or action history mismatch");
    auto released = moving;
    for (int i = 0; i < 40; ++i) released = observations.push(sensors, {});
    if (released[0] != -.4f || released[60] != 0.f || released[64] != 1.f)
        throw std::runtime_error("Filtered command did not reset wheel reference");
    observations.reset();
    if (observations.push(sensors, {})[42] != 0.f)
        throw std::runtime_error("Reset did not clear action history");
    const auto motor = targets(previous);
    if (motor.q[0] != Upper[0] || motor.dq[12] != 16.f || motor.kp[12] != 0.f || motor.kd[0] != 2.f)
        throw std::runtime_error("Action mapping mismatch");
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: replay_r284_ort POLICY_ONNX REPLAY_TXT\n";
        return 2;
    }
    try {
        verify_observations();
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "r284-replay");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        Ort::Session model(env, argv[1], options);
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::ifstream replay(argv[2]);
        if (!replay) throw std::runtime_error("Cannot open replay");
        const std::array<int64_t, 2> shape{1, 65};
        const char* inputs[] = {"obs"};
        const char* outputs[] = {"actions"};
        as2w_r284::WheelMemory history;
        unsigned frames = 0;
        unsigned resets = 0;
        float maximum = 0.f;
        int reset = 0;
        while (replay >> reset) {
            as2w_r284::Frame frame{};
            std::array<float, 16> expected{};
            for (auto& value : frame) replay >> value;
            for (auto& value : expected) replay >> value;
            if (!replay || (reset != 0 && reset != 1)) throw std::runtime_error("Invalid replay frame");
            if (reset) {
                history.reset();
                ++resets;
            }
            auto features = history.push(frame);
            auto input = Ort::Value::CreateTensor<float>(
                memory, features.data(), features.size(), shape.data(), shape.size());
            const auto result = model.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
            if (result[0].GetTensorTypeAndShapeInfo().GetElementCount() != expected.size()) {
                throw std::runtime_error("Unexpected output size");
            }
            const auto* actual = result[0].GetTensorData<float>();
            for (std::size_t i = 0; i < expected.size(); ++i) {
                if (!std::isfinite(actual[i])) throw std::runtime_error("Nonfinite action");
                maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
            }
            ++frames;
        }
        if (!replay.eof()) throw std::runtime_error("Invalid replay reset marker");
        std::cout << "frames=" << frames << " resets=" << resets << " max_action_error=" << maximum << '\n';
        return frames == 16000 && resets == 16 && maximum <= 1e-4f ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
