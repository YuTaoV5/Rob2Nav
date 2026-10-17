"""Export gains from explicit actuators, not the disabled PhysX joint drives."""
import torch


def actuator_gains(robot):
    kp = torch.zeros(len(robot.joint_names), device=robot.device)
    kd = torch.zeros_like(kp)
    for actuator in robot.actuators.values():
        kp[actuator.joint_indices] = actuator.stiffness[0]
        kd[actuator.joint_indices] = actuator.damping[0]
    if not torch.all(kp > 0):
        raise ValueError("Missing positive actuator stiffness in exported joint mapping")
    return kp.cpu().tolist(), kd.cpu().tolist()
