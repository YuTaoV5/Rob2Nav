"""Command continuity is separate from measured joint velocity and contact gates."""
import numpy as np


def assess(trace, names, velocity_limits, policy_hz=50.):
    if len(trace)<2:return dict(passed=False,reason='Insufficient target samples')
    q=np.asarray([row['target'] for row in trace],dtype=float)
    speed=np.asarray(velocity_limits,dtype=float)
    if q.shape!=(len(trace),len(names)) or speed.shape!=(len(names),) or not np.isfinite(q).all() or not np.isfinite(speed).all() or np.any(speed<=0):
        raise ValueError('Invalid target continuity inputs')
    delta=np.abs(np.diff(q,axis=0));ratio=delta*policy_hz/speed
    i,j=np.unravel_index(ratio.argmax(),ratio.shape)
    return dict(passed=bool(ratio.max()<=1.),max_target_step_rad=float(delta.max()),
        max_target_slew_urdf_ratio=float(ratio[i,j]),joint=names[j],
        recorded_control_step_end_s=float(trace[i+1]['time']),policy_hz=policy_hz,
        limit_target_slew_urdf_ratio=1.,sample_pairs=len(trace)-1,
        scope='Mapped target change per control period, not measured joint speed; first target has no preceding logged target')
