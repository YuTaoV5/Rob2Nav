"""Shared rolling acceptance limits for a neutral speed command."""


def rolling_brake_limits(speed_m_s: float) -> tuple[float, float]:
    speed = abs(speed_m_s)
    high_speed_fraction = min(1.0, max(0.0, (speed - 1.0) / 2.0))
    reference_deceleration = 1.2 - 0.4 * high_speed_fraction
    time_limit = max(2.5, speed / reference_deceleration + 1.0)
    distance_limit = speed**2 / (2.0 * reference_deceleration) + 0.25 * speed + 0.05
    return time_limit, distance_limit
