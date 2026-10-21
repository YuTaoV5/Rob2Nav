"""Measured endpoint acceptance, independent of training reward and action completion."""
import numpy as np


def assess(trace,initial_end=.8,terminal_hold=1.):
    if not trace:return dict(passed=False,reason='empty physical trace')
    initial=[x for x in trace if .1<=x['time']<=initial_end]
    terminal=[x for x in trace if x['time']>=trace[-1]['time']-terminal_hold]
    def window(samples,minimum):
        if len(samples)<minimum:return dict(passed=False,samples=len(samples),reason='insufficient endpoint coverage')
        good=[x['standard_pose_error']<=.12 and x['max_joint_speed']<=1. and x['upright']>=.98
              and x['speed']<=.1 and x['angular_speed']<=.5 and x['all_feet_ground'] for x in samples]
        fraction=float(np.mean(good))
        return dict(passed=fraction>=.95,samples=len(samples),stable_fraction=fraction,
                    max_pose_error_rad=max(x['standard_pose_error'] for x in samples))
    start=window(initial,max(5,int((initial_end-.1)*50*.8)));end=window(terminal,int(terminal_hold*50*.8))
    return dict(passed=start['passed'] and end['passed'],initial=start,terminal=end,
                pose_tolerance_rad=.12,max_joint_speed_rad_s=1.,minimum_upright_cos=.98,
                required_stable_fraction=.95,terminal_hold_s=terminal_hold)
