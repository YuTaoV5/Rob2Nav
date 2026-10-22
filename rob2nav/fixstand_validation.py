"""Measured endpoint acceptance, independent of training reward and action completion."""
import numpy as np


def rotation_matches(turn,direction):
    if direction not in ('left','right','front','back'):raise ValueError('Unknown flip direction')
    return bool(turn*(-1. if direction in ('left','back') else 1.)>4.5)


def stationary_upright(trace):
    """Require a quiet, loaded rear-foot hold, never forward travel or steps."""
    hold=[x for x in trace if 2.<=x['time']<5.]
    if len(hold)<140:return dict(passed=False,reason='insufficient 2–5 second hold coverage')
    dt=float(np.median(np.diff([x['time'] for x in hold])))
    good=[];longest=0.;continuous=0.;steps=[]
    for i,x in enumerate(hold):
        rear=all(x['foot_forces_N'][leg]>4. for leg in ('RR','RL'))
        front=all(x['foot_forces_N'][leg]<4. for leg in ('FR','FL'))
        step=float(np.max(abs(np.asarray(x['target'])-hold[i-1]['target']))) if i else 0.
        steps.append(step)
        ok=(x['upright_walk_error']<.35 and x['position'][2]>.42 and rear and front and
            x['speed']<=.08 and x['angular_speed']<=.5 and x['max_joint_speed']<=2. and step<=.06)
        good.append(ok);continuous=continuous+dt if ok else 0.;longest=max(longest,continuous)
    points=np.array([x['position'][:2] for x in hold]);drift=float(np.max(np.linalg.norm(points-points[0],axis=1)))
    fraction=float(np.mean(good));passed=fraction>=.95 and longest>=2. and drift<=.10 and max(steps)<=.06
    return dict(passed=bool(passed),stable_fraction=fraction,longest_stable_hold_s=longest,
                maximum_horizontal_drift_m=drift,max_target_step_rad=max(steps),required_hold_s=2.,
                hold_window_s=[2.,5.],maximum_drift_m=.10)


def walking_action(trace):
    """Require loaded rear-foot lift/landing cycles, not merely body displacement."""
    active=[x for x in trace if 1.<=x['time']<5. and x['upright_walk_error']<.45 and x['position'][2]>.42]
    if not active:return dict(passed=False,reason='no upright active samples')
    cycles={leg:0 for leg in ('RR','RL')}
    for leg in cycles:
        supported=False;flight_start=None
        for x in active:
            loaded=x['foot_forces_N'][leg]>4.
            if loaded:
                if supported and flight_start is not None and x['time']-flight_start>=.06:
                    cycles[leg]+=1
                supported=True;flight_start=None
            elif supported and flight_start is None:flight_start=x['time']
    travel=float(np.linalg.norm(np.asarray(active[-1]['position'][:2])-active[0]['position'][:2]))
    support=float(np.mean([x['rear_ground'] and not x['front_ground'] for x in active]))
    passed=len(active)*.02>=1.5 and travel>=.20 and support>=.8 and all(v>=2 for v in cycles.values())
    return dict(passed=bool(passed),upright_seconds=len(active)*.02,upright_travel_m=travel,
                rear_support_fraction=support,rear_loaded_step_cycles=cycles,minimum_cycles_per_rear_foot=2)


def assess(trace,initial_end=.8,terminal_hold=1.):
    if not trace:return dict(passed=False,reason='empty physical trace')
    initial=[x for x in trace if .1<=x['time']<=initial_end]
    terminal=[x for x in trace if x['time']>=trace[-1]['time']-terminal_hold]
    def window(samples,minimum):
        if len(samples)<minimum:return dict(passed=False,samples=len(samples),reason='insufficient endpoint coverage')
        good=[x['standard_pose_error']<=.12 and x['max_joint_speed']<=1. and x['upright']>=.98
              and x['speed']<=.1 and x['angular_speed']<=.5 and x['all_feet_ground']
              and .26<=x['position'][2]<=.40 for x in samples]
        fraction=float(np.mean(good))
        return dict(passed=fraction>=.95,samples=len(samples),stable_fraction=fraction,
                    max_pose_error_rad=max(x['standard_pose_error'] for x in samples))
    start=window(initial,max(5,int((initial_end-.1)*50*.8)));end=window(terminal,int(terminal_hold*50*.8))
    return dict(passed=start['passed'] and end['passed'],initial=start,terminal=end,
                pose_tolerance_rad=.12,max_joint_speed_rad_s=1.,minimum_upright_cos=.98,
                required_stable_fraction=.95,terminal_hold_s=terminal_hold,base_height_range_m=[.26,.40],minimum_foot_force_N=4.)
