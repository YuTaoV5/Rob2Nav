#include "r270_history.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: replay_r270_ort POLICY_ONNX REPLAY_TXT\n";
        return 2;
    }
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "r270-replay");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        Ort::Session model(env, argv[1], options);
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::ifstream replay(argv[2]);
        if (!replay) throw std::runtime_error("Cannot open replay");
        const std::array<int64_t, 2> shape{1, 720};
        const char* inputs[] = {"obs"};
        const char* outputs[] = {"actions"};
        R270History history;
        unsigned frames = 0;
        unsigned resets = 0;
        float maximum = 0.f;
        int reset = 0;
        while (replay >> reset) {
            R270History::Frame frame{};
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
        return frames == 2350 && resets == 2 && maximum <= 1e-4f ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
