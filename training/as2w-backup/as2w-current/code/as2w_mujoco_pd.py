"""Shared MuJoCo actuator configuration for simulator diagnostics."""
import mujoco
import numpy as np


def configure_native_pd(model, ids, kp, kd, low, high, effort, *, velocity_axes=0):
    """Integrate the PD velocity term while enforcing the same actuator force caps."""
    ids = np.asarray(ids)
    gain = np.asarray(kp).copy()
    if velocity_axes:
        gain[-velocity_axes:] = kd[-velocity_axes:]
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.actuator_gaintype[ids] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_gainprm[ids] = 0.
    model.actuator_gainprm[ids,0] = gain
    model.actuator_biastype[ids] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_biasprm[ids] = 0.
    model.actuator_biasprm[ids,1] = -kp
    model.actuator_biasprm[ids,2] = -kd
    model.actuator_forcelimited[ids] = True
    model.actuator_forcerange[ids] = np.column_stack((-effort,effort))
    model.actuator_ctrllimited[ids] = True
    model.actuator_ctrlrange[ids] = np.column_stack((low,high))
