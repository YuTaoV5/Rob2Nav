"""Shared Go2 asset contract for reference FK and actual dynamics training."""
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

BACKFLIP_COMMIT = '954aa61ceb62c515689677f44a21e32109f6b62d'
UNITREE_COMMIT = '4960b84732b0c2ec593dccbfe963fda1bcd7b1e3'
CSV_JOINTS = [f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL') for joint in ('hip', 'thigh', 'calf')]


def pinned(path, commit):
    path = Path(path).resolve()
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != commit:
        raise RuntimeError(f'Expected {commit}, got {actual}: {path}')
    return path


def setup_sources(backflip, unitree=None):
    backflip = pinned(backflip, BACKFLIP_COMMIT)
    sys.path.insert(0, str(backflip))
    if unitree:
        unitree = pinned(unitree, UNITREE_COMMIT)
        sys.path.insert(0, str(unitree/'source/unitree_rl_lab'))
    from isaaclab.sim.converters.urdf_converter import UrdfConverter
    original = UrdfConverter._get_urdf_import_config
    code = textwrap.dedent(inspect.getsource(original))
    needle = 'import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)'
    if needle in code:
        code = code.replace(needle, 'if hasattr(import_config, "set_merge_fixed_ignore_inertia"): '+needle)
        scope = dict(original.__globals__)
        exec(compile('from __future__ import annotations\n'+code, '<rob2nav-urdf-compat>', 'exec'), scope)
        UrdfConverter._get_urdf_import_config = scope['_get_urdf_import_config']
    return backflip


def robot_config(backflip):
    from isaaclab_backflip.tasks.go2_backflip.go2_backflip_env_cfg import Go2BackflipEnvCfg, GO2_URDF
    cfg = Go2BackflipEnvCfg().robot.replace(prim_path='{ENV_REGEX_NS}/Robot')
    # A dedicated converter cache never rewrites the assets of the running backflip experiment.
    cfg.spawn.usd_dir = str(Path(backflip).parent/'generated/go2_mimic_usd')
    limits = {j.attrib['name']: j.find('limit').attrib for j in ET.parse(GO2_URDF).getroot().findall('joint')
        if j.attrib['name'] in CSV_JOINTS}
    if set(limits) != set(CSV_JOINTS):
        raise RuntimeError('URDF does not contain all twelve canonical Go2 joints')
    for actuator in cfg.actuators.values():
        actuator.stiffness = 40.
        actuator.damping = 1.
        actuator.effort_limit = {n:float(v['effort']) for n,v in limits.items()}
        actuator.effort_limit_sim = dict(actuator.effort_limit)
        actuator.velocity_limit = {n:float(v['velocity']) for n,v in limits.items()}
        actuator.velocity_limit_sim = dict(actuator.velocity_limit)
    return cfg


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def enable_transfer_motors(cfg, source):
    """Explicit DC envelope, no PhysX joint-speed clamp; experimental transfer revision."""
    from isaaclab.actuators import DCMotorCfg
    limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(Path(source)/'resources/robots/go2/urdf/go2_description.urdf').getroot().findall('joint') if j.attrib['name'] in CSV_JOINTS}
    cfg.scene.robot.actuators={name:DCMotorCfg(joint_names_expr=[name],stiffness=40.,damping=1.,
        effort_limit=float(v['effort']),effort_limit_sim=float(v['effort']),
        velocity_limit=float(v['velocity']),velocity_limit_sim=1.e6,saturation_effort=float(v['effort']))
        for name,v in limits.items()}
    cfg.sim.dt=.002
    cfg.decimation=10
    cfg.sim.render_interval=10
    cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count=8
    cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count=2
