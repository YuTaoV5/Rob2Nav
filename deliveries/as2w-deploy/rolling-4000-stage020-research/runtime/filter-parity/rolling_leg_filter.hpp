#pragma once

#include <algorithm>
#include <array>
#include <cmath>

class RollingLegFilter {
public:
    using Legs = std::array<float, 12>;

    void reset() {
        filtered_.fill(0.0f);
        last_target_.fill(0.0f);
    }

    Legs step(const Legs& raw, float vx, float wz, const Legs& stand) {
        const float motion = std::min(1.0f, std::max(std::abs(vx) / 0.10f,
                                                   std::abs(wz) / 0.20f));
        for (std::size_t i = 0; i < raw.size(); ++i) {
            const float action = clamp(raw[i], 6.0f);
            filtered_[i] += alpha_ * (action - filtered_[i]);
            const float target = stand[i] + 0.18f * motion * clamp(filtered_[i], 1.0f);
            last_target_[i] += clamp(target - last_target_[i], step_limit_);
        }
        return last_target_;
    }

private:
    static float clamp(float value, float limit) {
        return std::max(-limit, std::min(value, limit));
    }

    const float alpha_ = -std::expm1(-2.0 * 3.141592653589793 * 5.0 * 0.020);
    const float step_limit_ = 0.9f * 0.020f / 0.35f;
    Legs filtered_{};
    Legs last_target_{};
};
