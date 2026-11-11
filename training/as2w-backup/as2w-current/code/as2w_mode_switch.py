"""Read-only supervisor for AS2-W mode-switch validation."""
from dataclasses import dataclass
from enum import Enum
from math import isfinite


class Phase(str, Enum):
    STAND = "fixed_stand"
    ENTER = "enter_policy"
    ACTIVE = "policy_active"
    BRAKE = "brake_to_stop"
    EXIT = "exit_policy"
    FAULT = "fault"


SKILLS = ("rolling", "stepping", "getup")


@dataclass(frozen=True)
class Feedback:
    time_s: float
    age_s: float
    speed_m_s: float
    yaw_rad_s: float
    upright_cosine: float
    leg_error_rad: float
    loaded_wheels: int
    centered_stick: bool
    upright_contact_safe: bool
    recovery_pose_supported: bool = False
    guard_ok: bool = True


@dataclass(frozen=True)
class Decision:
    phase: Phase
    policy: str | None
    weight: float
    joystick_enabled: bool
    reset_policy_history: bool
    fault: str | None = None


def fixed_stand_checks(feedback):
    return {
        "speed": feedback.speed_m_s <= .025,
        "yaw": abs(feedback.yaw_rad_s) <= .06,
        "upright": feedback.upright_cosine > .95,
        "leg_error": feedback.leg_error_rad < .18,
        "wheel_load": feedback.loaded_wheels == 4,
        "contact": feedback.upright_contact_safe,
    }


class ModeSwitch:
    def __init__(self):
        self.phase = Phase.STAND
        self.policy = None
        self.phase_start = None
        self.last_time = None
        self.fixed_since = None
        self.quiet_since = None
        self.fault = None
        self.exit_start_weight = 1.

    def _enter(self, phase, now):
        self.phase = phase
        self.phase_start = now

    def _exit(self, now, weight=1.):
        self.exit_start_weight = weight
        self._enter(Phase.EXIT,now)

    def _fault(self, reason):
        self.phase = Phase.FAULT
        self.fault = reason
        return Decision(self.phase,None,0.,False,False,reason)

    def step(self, feedback, requested):
        f = feedback
        if self.fault is not None:
            return self._fault(self.fault)
        if requested is not None and requested not in SKILLS:
            return self._fault("unknown_policy_request")
        numbers = (f.time_s,f.age_s,f.speed_m_s,f.yaw_rad_s,f.upright_cosine,f.leg_error_rad)
        if (not all(isfinite(x) for x in numbers) or not 0. <= f.age_s <= .06
                or f.speed_m_s < 0. or f.leg_error_rad < 0. or not -1.02 <= f.upright_cosine <= 1.02):
            return self._fault("invalid_or_stale_feedback")
        if self.last_time is not None and not 0. < f.time_s-self.last_time <= .06:
            return self._fault("discontinuous_feedback_time")
        if not f.guard_ok or not 0 <= f.loaded_wheels <= 4:
            return self._fault("motion_guard")
        if (self.policy in ("rolling","stepping") and not f.upright_contact_safe):
            return self._fault("nonwheel_or_knee_contact")
        self.last_time = f.time_s
        if self.phase_start is None:
            self.phase_start = f.time_s
        quiet = f.speed_m_s <= .025 and abs(f.yaw_rad_s) <= .06
        fixed = all(fixed_stand_checks(f).values())
        self.quiet_since = (self.quiet_since if self.quiet_since is not None else f.time_s) if quiet else None
        self.fixed_since = (self.fixed_since if self.fixed_since is not None else f.time_s) if fixed else None
        fixed_ready = self.fixed_since is not None and f.time_s-self.fixed_since >= .40
        quiet_ready = self.quiet_since is not None and f.time_s-self.quiet_since >= .50
        age = f.time_s-self.phase_start
        reset = False
        if self.phase == Phase.STAND:
            ready = requested in ("rolling","stepping") and fixed_ready and f.centered_stick
            recover = (requested=="getup" and f.recovery_pose_supported and f.upright_cosine > 0.
                       and f.centered_stick and quiet_ready and not fixed)
            if ready or recover:
                self.policy = requested
                self._enter(Phase.ENTER,f.time_s)
                reset = True
        elif self.phase == Phase.ENTER:
            if self.policy != "getup" and (requested != self.policy or f.loaded_wheels != 4
                                          or f.upright_cosine < .90 or not f.centered_stick):
                self._exit(f.time_s,min(1.,age/.75))
            elif age >= .75:
                self._enter(Phase.ACTIVE,f.time_s)
        elif self.phase == Phase.ACTIVE:
            if self.policy == "getup":
                if fixed_ready:
                    self._exit(f.time_s)
                elif age >= 12.:
                    return self._fault("recovery_timeout")
            elif requested != self.policy:
                self.quiet_since = None
                self._enter(Phase.BRAKE,f.time_s)
        elif self.phase == Phase.BRAKE:
            if quiet_ready:
                self._exit(f.time_s)
            elif age >= 5.:
                return self._fault("brake_timeout")
        elif self.phase == Phase.EXIT:
            if age >= .75 and fixed_ready:
                self.policy = None
                self.fixed_since = None
                self._enter(Phase.STAND,f.time_s)
            elif age >= 4.:
                return self._fault("fixed_stand_timeout")
        age = f.time_s-self.phase_start
        if self.phase == Phase.ENTER:
            weight = min(1.,age/.75)
        elif self.phase == Phase.EXIT:
            weight = self.exit_start_weight*max(0.,1.-age/.75)
        else:
            weight = float(self.phase in (Phase.ACTIVE,Phase.BRAKE))
        return Decision(self.phase,self.policy,weight,
                        self.phase==Phase.ACTIVE and self.policy!="getup",reset)
