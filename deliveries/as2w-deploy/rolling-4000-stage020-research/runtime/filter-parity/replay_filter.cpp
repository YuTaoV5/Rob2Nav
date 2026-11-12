#include "rolling_leg_filter.hpp"

#include <iomanip>
#include <iostream>

int main() {
    RollingLegFilter filter;
    int reset;
    float vx, wz;
    RollingLegFilter::Legs raw{}, stand{};
    std::cout << std::setprecision(9);
    while (std::cin >> reset >> vx >> wz) {
        for (float& value : raw) {
            if (!(std::cin >> value)) return 1;
        }
        for (float& value : stand) {
            if (!(std::cin >> value)) return 1;
        }
        if (reset) filter.reset();
        const auto target = filter.step(raw, vx, wz, stand);
        for (float value : target) std::cout << value << ' ';
        std::cout << '\n';
    }
    return std::cin.eof() ? 0 : 1;
}
