"""Measured in-place rear-foot stepping; distinct from static hold and walking."""
import numpy as np


def nonfoot_ground_contact(geom1,geom2,floor_ids,foot_ids):
    """Legacy geometry-ID diagnostic; GroundContactReader defines acceptance."""
    if geom1 in floor_ids and geom2 not in floor_ids:return geom2 not in foot_ids
    if geom2 in floor_ids and geom1 not in floor_ids:return geom1 not in foot_ids
    return False


def assess_in_place_steps(trace):
    hold=[x for x in trace if 2.<=x['time']<5.]
    if len(hold)<140:
        return dict(passed=False,reason='insufficient 2–5 second stepping coverage')
    if any('stepping_foot_forces_N' not in x or 'foot_clearance_m' not in x for x in hold):
        return dict(passed=False,reason='measured anatomical foot contact forces and clearance are required')
    times=np.asarray([x['time'] for x in hold])
    if np.any(np.diff(times)<=0) or np.max(np.diff(times))>.025:
        return dict(passed=False,reason='stepping samples must be continuous at 50Hz')
    cycles={leg:[] for leg in ('RR','RL')}
    for leg in cycles:
        supported=False;flight=None;peak=0.
        for x in hold:
            loaded=x['stepping_foot_forces_N'][leg]>4.
            if loaded:
                if supported and flight is not None and x['time']-flight>=.06 and peak>=.04:
                    cycles[leg].append(dict(landing_s=x['time'],flight_s=x['time']-flight,clearance_m=peak))
                supported=True;flight=None;peak=0.
            elif supported:
                if flight is None:flight=x['time']
                peak=max(peak,x['foot_clearance_m'][leg])
    events=sorted((c['landing_s'],leg) for leg,items in cycles.items() for c in items)
    # Reject both members of a simultaneous landing pair. Counting the boundary
    # between pairs would otherwise mistake repeated two-foot hops for steps.
    events=[(t,leg) for t,leg in events
            if not any(other_leg!=leg and abs(other_t-t)<.08 for other_t,other_leg in events)]
    alternations=sum(leg!=previous_leg and t-previous_t>=.08
        for (previous_t,previous_leg),(t,leg) in zip(events,events[1:]))
    target=np.asarray([x['target'] for x in hold]);steps=np.max(abs(np.diff(target,axis=0)),axis=1)
    step_max=float(steps.max())
    support=[];good=[]
    for x in hold:
        forces=x['stepping_foot_forces_N']
        rear=any(forces[leg]>4. for leg in ('RR','RL'))
        front=all(forces[leg]<4. for leg in ('FR','FL'))
        support.append(rear and front)
        good.append(rear and front and x['upright_walk_error']<.35 and x['position'][2]>.42
                    and x['speed']<=.15 and x['angular_speed']<=1.)
    points=np.asarray([x['position'][:2] for x in hold]);drift=float(np.linalg.norm(points-points[0],axis=1).max())
    stable_fraction=float(np.mean(good));support_fraction=float(np.mean(support))
    passed=(stable_fraction>=.95 and support_fraction>=.95 and drift<=.10 and step_max<=.08
            and all(len(items)>=3 for items in cycles.values()) and alternations>=4)
    return dict(passed=bool(passed),hold_window_s=[2.,5.],samples=len(hold),
        stable_fraction=stable_fraction,rear_support_fraction=support_fraction,
        maximum_horizontal_drift_m=drift,max_target_step_rad=step_max,
        rear_loaded_step_cycles={leg:len(items) for leg,items in cycles.items()},
        qualified_cycles=cycles,alternating_landings=alternations,
        limits=dict(minimum_cycles_per_rear_foot=3,minimum_clearance_m=.04,
                    minimum_flight_s=.06,minimum_alternating_landings=4,
                    maximum_horizontal_drift_m=.10,maximum_target_step_rad=.08,
                    required_stable_fraction=.95,maximum_speed_m_s=.15,
                    maximum_angular_speed_rad_s=1.))


def foot_collision_geometries(model):
    from .ground_contacts import foot_collision_geometries as locate
    return locate(model)
