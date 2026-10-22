"""Zero commanded displacement while preserving alternate rear-foot steps."""


def drift_cost(env):
    import torch
    from skills.fixstand import phase_time
    robot = env.scene['robot']
    t = phase_time(env)
    # Per-episode anchor is sampled at the start of the active hold window.
    # It is diagnostic/reward memory only; never writes simulator state.
    if not hasattr(env, '_in_place_anchor'):
        env._in_place_anchor = robot.data.root_pos_w[:, :2].clone()
    entering = t < 2.
    env._in_place_anchor[entering] = robot.data.root_pos_w[entering, :2]
    displacement = torch.sum((robot.data.root_pos_w[:, :2] - env._in_place_anchor)**2, dim=-1)
    velocity = torch.sum(robot.data.root_lin_vel_w[:, :2]**2, dim=-1)
    return ((t >= 2.) & (t < 5.)) * (displacement / .01 + velocity / .04)


def configure(cfg,qualifying_airtime=False,airtime_weight=60.):
    from isaaclab.managers import RewardTermCfg
    for key in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'):
        setattr(cfg.commands.base_velocity.ranges, key, (0., 0.))
    cfg.commands.base_velocity.rel_standing_envs = 1.
    cfg.rewards.in_place_drift = RewardTermCfg(func=drift_cost, weight=-4.)
    if qualifying_airtime:
        from .airtime import qualifying_airtime as objective
        cfg.rewards.rear_airtime=RewardTermCfg(func=objective,weight=airtime_weight)
    # rear_alternation is deliberately retained; the static double-support
    # objective would reward keeping both rear feet down instead of stepping.
