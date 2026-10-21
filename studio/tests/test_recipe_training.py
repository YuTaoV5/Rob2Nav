from pathlib import Path

import pytest

from studio.config import RunConfig, command


def training_config(recipe, **overrides):
    values = {
        'navigation_ppo': dict(adapter='robogauge', algorithm_id='navigation_ppo', capability_id='navigation'),
        'sideflip_native_ppo': dict(adapter='robogauge', algorithm_id='motion_imitation_ppo', capability_id='sideflip'),
        'upright_walk_ppo': dict(adapter='robogauge', algorithm_id='upright_velocity_ppo', capability_id='upright_walk'),
        'backflip_native_ppo': dict(adapter='robogauge', algorithm_id='backflip_ppo', capability_id='backflip'),
    }[recipe]
    return RunConfig(recipe_id=recipe, **(values | overrides))


def test_navigation_uses_mujoco_runtime_and_editable_steps(tmp_path):
    root = tmp_path
    config = training_config('navigation_ppo', recipe_options={'steps': 512, 'envs': 1, 'sensors': 'lidar'})
    cmd = command(config, root, root/'outputs/studio/runs/PREVIEW', root/'preview.json', '/isaac/python.sh', '/mujoco/python')
    assert cmd[:3] == ['/mujoco/python', '-m', 'go2_nav.train']
    assert cmd[cmd.index('--steps') + 1] == '512'
    assert cmd[cmd.index('--envs') + 1] == '1'


@pytest.mark.parametrize('recipe,module', [
    ('sideflip_native_ppo', 'skills.sideflip.train'),
    ('upright_walk_ppo', 'skills.upright_walk.train'),
])
def test_isaac_skill_command_is_allowlisted(tmp_path, recipe, module):
    config = training_config(recipe, num_envs=32, iterations=2, recipe_options={'transfer_motors': True})
    cmd = command(config, tmp_path, tmp_path/'outputs/studio/runs/PREVIEW', tmp_path/'preview.json', '/isaac/python.sh')
    assert cmd[:3] == ['/isaac/python.sh', '-m', module]
    assert '--transfer-motors' in cmd
    assert cmd[cmd.index('--iterations') + 1] == '2'


def test_unreviewed_option_is_rejected():
    with pytest.raises(ValueError, match='Unsupported recipe option'):
        training_config('navigation_ppo', recipe_options={'shell': 'echo unsafe'})


def test_resume_must_point_to_existing_project_artifact(tmp_path):
    config = training_config('navigation_ppo', initialization='resume', checkpoint='../outside.zip')
    with pytest.raises(ValueError, match='Resume checkpoint'):
        command(config, tmp_path, tmp_path/'outputs/studio/runs/PREVIEW', tmp_path/'preview.json', '/isaac/python.sh')
