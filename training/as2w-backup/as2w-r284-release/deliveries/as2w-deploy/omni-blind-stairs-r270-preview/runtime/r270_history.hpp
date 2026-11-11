#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>

class R270History {
public:
    using Frame = std::array<float, 60>;
    using Observation = std::array<float, 720>;

    void reset() { frames_.clear(); }

    Observation push(const Frame& frame) {
        if (frames_.size() == 25) frames_.pop_front();
        frames_.push_back(frame);
        Observation observation{};
        auto output = std::copy(frame.begin(), frame.end(), observation.begin());
        for (std::size_t offset = 2; offset <= 24; offset += 2) {
            const auto& past = frames_[frames_.size() - std::min(offset + 1, frames_.size())];
            output = std::copy(past.begin() + 3, past.begin() + 58, output);
        }
        return observation;
    }

private:
    std::deque<Frame> frames_;
};
