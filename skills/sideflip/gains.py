"""Scalar joint gains shared by training, export and execution."""
import math


def checked(kp=40., kd=1.):
    for name, value, low, high in [('kp', kp, 1., 80.), ('kd', kd, 0., 8.)]:
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Invalid sideflip {name}')
    return float(kp), float(kd)


def configure(cfg, kp=40., kd=1.):
    kp, kd = checked(kp, kd)
    for actuator in cfg.scene.robot.actuators.values():
        actuator.stiffness = kp
        actuator.damping = kd


def verify_export(env, contract):
    kp, kd = checked(contract.get('kp', 40.), contract.get('kd', 1.))
    actuators = env['scene']['robot']['actuators']
    if not actuators or any(a['stiffness'] != kp or a['damping'] != kd for a in actuators.values()):
        raise ValueError('Sideflip actual actuator gains disagree with export contract')
    return kp, kd
