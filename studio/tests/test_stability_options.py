"""Contract tests: edited web options must reach the real trainer unchanged."""
import pytest
from studio.config import RunConfig, command
from studio.registry import registry
from studio.stability_options import validate


def config(rid, **changes):
    d = dict(next(r for r in registry()['recipes'] if r['id'] == rid)['defaults'])
    d.update(changes)
    return RunConfig(**d)


def test_frontflip_is_independent_recipe_and_direction(tmp_path):
    c = config('frontflip_native_ppo', num_envs=32, iterations=2)
    cmd = command(c, tmp_path, tmp_path/'outputs/studio/runs/PREVIEW', tmp_path/'preview.json', '/isaac/python.sh')
    assert c.capability_id == 'frontflip'
    assert cmd[cmd.index('--flip-direction') + 1] == 'front'
    assert '--fixstand-output' in cmd and '--delayed-entry' in cmd
    with pytest.raises(ValueError):
        config('backflip_native_ppo', recipe_options={'flip_direction':'front'})


def test_sideflip_web_gain_edits_reach_trainer(tmp_path):
    c=config('sideflip_native_ppo',recipe_options={'joint_kp':40.,'joint_kd':2.})
    cmd=command(c,tmp_path,tmp_path/'outputs/studio/runs/PREVIEW',tmp_path/'preview.json','/isaac/python.sh')
    assert cmd[cmd.index('--joint-kp')+1]=='40.0'
    assert cmd[cmd.index('--joint-kd')+1]=='2.0'
    with pytest.raises(ValueError):validate('sideflip_native_ppo',{'joint_kd':float('nan')})


def test_sideflip_mapped_target_rate_options(tmp_path):
    c=config('sideflip_native_ppo',recipe_options={'fixstand_roundtrip':True,'boundary_residual':True,'target_rate_weight':5.,'target_rate_margin':.8})
    cmd=command(c,tmp_path,tmp_path/'outputs/studio/runs/PREVIEW',tmp_path/'preview.json','/isaac/python.sh')
    assert cmd[cmd.index('--target-rate-weight')+1]=='5.0'
    assert cmd[cmd.index('--target-rate-margin')+1]=='0.8'
    with pytest.raises(ValueError):validate('sideflip_native_ppo',{'target_rate_weight':5.})


def test_sideflip_substep_resume_contract(tmp_path):
    ck = tmp_path/'outputs/parent/model_10.pt'; ck.parent.mkdir(parents=True); ck.write_bytes(b'parent')
    motion = ck.parent/'reference.npz'; motion.write_bytes(b'motion')
    options = dict(transfer_motors=True, fixstand_roundtrip=True, boundary_residual=True,
                   boundary_residual_limit=.2, boundary_bridge_limit=.05,
                   boundary_terminal_hip_taper_frames=50, boundary_blend_frames=10,
                   quiet_preparation=True, peak_endpoint_objective=True, landing_constraints=True,
                   substep_objective=True, finetune_learning_rate=.00001, reset_std=.03,
                   motion='outputs/parent/reference.npz')
    c = config('sideflip_native_ppo', initialization='resume', checkpoint='outputs/parent/model_10.pt', recipe_options=options)
    cmd = command(c, tmp_path, tmp_path/'outputs/studio/runs/PREVIEW', tmp_path/'preview.json', '/isaac/python.sh')
    assert '--substep-objective' in cmd and '--quiet-preparation' in cmd
    assert cmd[cmd.index('--motion')+1] == str(motion)
    assert float(cmd[cmd.index('--finetune-learning-rate')+1]) == .00001
    assert cmd[cmd.index('--resume')+1] == str(ck)


def test_backflip_resume_outputs_inside_studio(tmp_path):
    ck = tmp_path/'outputs/parent/model_10.pt'; ck.parent.mkdir(parents=True); ck.write_bytes(b'parent')
    c = config('backflip_native_ppo', initialization='resume', checkpoint='outputs/parent/model_10.pt',
               recipe_options=dict(fixstand_roundtrip=True, delayed_entry=True, entry_hold_scale=.15,
                                   urdf_speed_limits=True, fast_safety_curriculum=True, safety_start=1.,
                                   finetune_learning_rate=.00001, reset_optimizer=True,
                                   min_action_std=.05, max_action_std=.1))
    out = tmp_path/'outputs/studio/runs/PREVIEW'
    cmd = command(c, tmp_path, out, tmp_path/'preview.json', '/isaac/python.sh')
    assert cmd[cmd.index('--fixstand-output')+1] == str(out)
    assert '--reset_optimizer' in cmd and '--resume' in cmd
    assert cmd[cmd.index('--checkpoint')+1] == str(ck)


@pytest.mark.parametrize('rid,options', [
    ('sideflip_native_ppo', {'substep_objective':True}),
    ('sideflip_native_ppo', {'reset_std':.001}),
    ('sideflip_native_ppo', {'boundary_blend_frames':1}),
    ('upright_walk_ppo', {'in_place_stepping':True}),
    ('upright_walk_ppo', {'stationary_upright':True, 'stepping_gait':True}),
    ('backflip_native_ppo', {'entry_hold_scale':.15}),
    ('backflip_native_ppo', {'safety_start':.5}),
    ('sideflip_native_ppo', {'finetune_learning_rate':float('nan')}),
    ('sideflip_native_ppo', {'boundary_blend_frames':True}),
])
def test_incompatible_or_invalid_controls_rejected(rid, options):
    with pytest.raises(ValueError): validate(rid, options)


def test_in_place_stepping_keeps_alternation_and_is_not_static(tmp_path):
    options = dict(in_place_stepping=True, stepping_gait=True, alternating_rear_gait=True,
                   transfer_motors=True, fixstand_roundtrip=True)
    c = config('upright_walk_ppo', recipe_options=options)
    cmd = command(c, tmp_path, tmp_path/'outputs/studio/runs/PREVIEW', tmp_path/'preview.json', '/isaac/python.sh')
    assert all(x in cmd for x in ('--in-place-stepping', '--stepping-gait', '--alternating-rear-gait'))
    assert '--stationary-upright' not in cmd


def test_motion_cannot_escape_project(tmp_path):
    c = config('sideflip_native_ppo', recipe_options={'motion':'../outside.npz'})
    with pytest.raises(ValueError, match='Motion must'):
        command(c, tmp_path, tmp_path/'outputs/studio/runs/PREVIEW', tmp_path/'preview.json', '/isaac/python.sh')
