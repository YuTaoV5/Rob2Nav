#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: replay_r249_ort MODEL REPLAY\n";
        return 2;
    }
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "r249-replay");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        Ort::Session model(env, argv[1], options);
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::ifstream replay(argv[2]);
        if (!replay) throw std::runtime_error("Cannot open replay");
        std::array<float, 60> obs{};
        std::array<float, 16> expected{};
        float elapsed = 0.f;
        const std::array<int64_t, 2> obs_shape{1, 60};
        const int64_t elapsed_shape = 1;
        const char* inputs[] = {"obs", "elapsed"};
        const char* outputs[] = {"actions"};
        unsigned frames = 0;
        float maximum = 0.f;
        while (replay >> obs[0]) {
            for (std::size_t i = 1; i < obs.size(); ++i) replay >> obs[i];
            replay >> elapsed;
            for (auto& value : expected) replay >> value;
            if (!replay) throw std::runtime_error("Incomplete replay frame");
            std::array<Ort::Value, 2> input{
                Ort::Value::CreateTensor<float>(memory, obs.data(), obs.size(), obs_shape.data(), obs_shape.size()),
                Ort::Value::CreateTensor<float>(memory, &elapsed, 1, &elapsed_shape, 1),
            };
            const auto result = model.Run(Ort::RunOptions{nullptr}, inputs, input.data(), input.size(), outputs, 1);
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
        std::cout << "frames=" << frames << " max_action_error=" << maximum << '\n';
        return frames && maximum <= 1e-4f ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
