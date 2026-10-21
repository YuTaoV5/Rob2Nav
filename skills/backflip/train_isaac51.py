"""Pinned-source launcher for IsaacLab 2.3 / IsaacSim 5.1 URDF API mismatch.
Does not alter global IsaacLab or physics parameters.
"""
import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--source', required=True, type=Path)
parser.add_argument('--entry', choices=['train', 'play'], default='train')
parser.add_argument('--fixstand-roundtrip',action='store_true')
parser.add_argument('--fixstand-output',type=Path)
parser.add_argument('--delayed-entry',action='store_true')
parser.add_argument('--entry-hold-scale',type=float,default=.4)
parser.add_argument('--urdf-speed-limits',action='store_true')
parser.add_argument('--fast-safety-curriculum',action='store_true')
parser.add_argument('--safety-start',type=float)
parser.add_argument('--finetune-learning-rate',type=float)
args, rest = parser.parse_known_args()
if args.delayed_entry and not args.fixstand_roundtrip:raise ValueError('Delayed entry requires fixstand training')
if not 0.<=args.entry_hold_scale<=.4 or (args.entry_hold_scale!=.4 and not args.delayed_entry):raise ValueError('Entry hold scale requires delayed entry and [0, 0.4]')
if args.fast_safety_curriculum and not args.urdf_speed_limits:raise ValueError('Fast safety curriculum requires URDF speed limits')
if args.safety_start is not None and (not args.fast_safety_curriculum or not 0.<=args.safety_start<=1.):raise ValueError('Safety start requires fast curriculum and [0,1]')
if args.finetune_learning_rate is not None and not 0.<args.finetune_learning_rate<=.001:raise ValueError('Invalid finetune learning rate')
source = args.source.resolve()
entry = source / (args.entry + '.py')
text = entry.read_text()
needle = 'simulation_app = app_launcher.app'
if text.count(needle) != 1:
    raise RuntimeError('Upstream launcher changed; review compatibility patch')
patch = r'''
from isaaclab.sim.converters.urdf_converter import UrdfConverter
import inspect as _inspect
import textwrap as _textwrap
_original = UrdfConverter._get_urdf_import_config
_method = _textwrap.dedent(_inspect.getsource(_original))
_old = 'import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)'
if _old in _method:
    _method = _method.replace(_old, 'if hasattr(import_config, "set_merge_fixed_ignore_inertia"): ' + _old)
    _scope = dict(_original.__globals__)
    exec(compile('from __future__ import annotations\n' + _method, '<rob2nav-urdf-compat>', 'exec'), _scope)
    UrdfConverter._get_urdf_import_config = _scope['_get_urdf_import_config']
    print('[Rob2Nav] Guarded optional URDF importer inertia API; native legacy behavior retained', flush=True)
'''
text = text.replace(needle, needle + '\n' + patch)
if args.fixstand_roundtrip:
    if args.entry!='train':raise ValueError('Fixstand launcher currently targets training')
    for required in ('env_cfg = Go2BackflipEnvCfg()','init_at_random_ep_len=True'):
        if text.count(required)!=1:raise RuntimeError('Pinned training entry changed: '+required)
    text=text.replace('env_cfg = Go2BackflipEnvCfg()',
        'env_cfg = Go2BackflipEnvCfg()\n    from skills.fixstand import configure_backflip\n    '
        + f'configure_backflip(env_cfg,{str(source)!r} if {args.urdf_speed_limits!r} else None,{args.fast_safety_curriculum!r})')
    if args.safety_start is not None:
        text=text.replace('agent_cfg = Go2BackflipPPORunnerCfg()',
            f'env_cfg.rewards.safety_curriculum_start={args.safety_start!r}\n    agent_cfg = Go2BackflipPPORunnerCfg()')
    if args.finetune_learning_rate is not None:
        needle='agent_cfg = Go2BackflipPPORunnerCfg()'
        if text.count(needle)!=1:raise RuntimeError('Pinned agent config entry changed')
        text=text.replace(needle,needle+f'\n    agent_cfg.algorithm.learning_rate={args.finetune_learning_rate!r}\n    agent_cfg.algorithm.schedule="fixed"')
    if args.delayed_entry:
        text=text.replace('configure_backflip(env_cfg)', f'configure_backflip(env_cfg)\n    from skills.fixstand import configure_backflip_delayed_entry\n    configure_backflip_delayed_entry(env_cfg,{args.entry_hold_scale!r})')
    text=text.replace('init_at_random_ep_len=True','init_at_random_ep_len=False')
    text=text.replace('runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=False)',
        'runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=False)\n    from skills.fixstand import write_completion\n    write_completion(log_dir,agent_cfg.max_iterations)')
    if args.fixstand_output:
        output=args.fixstand_output.resolve()
        if output.exists():raise FileExistsError(output)
        text=text.replace('log_dir = (log_root / run_dir_name).resolve()',f'log_dir = Path({str(output)!r})')
        text=text.replace('raw_env = gym.make(TASK_ID, cfg=env_cfg)',"from skills.fixstand import write_contract\n    write_contract(log_dir / 'fixstand.json','backflip',actor_observations=60)\n    raw_env = gym.make(TASK_ID, cfg=env_cfg)")
        if args.delayed_entry:
            version='backflip_entry_v2' if args.entry_hold_scale==.4 else 'backflip_entry_v3'
            text=text.replace("'backflip',actor_observations=60)",f"'backflip',actor_observations=60,action_parameterization='{version}',entry_hold_scale={args.entry_hold_scale!r},entry_seconds=1.,release_seconds=[1.,1.2],episode_seconds=5.)")
if '--checkpoint' in rest:
    index=rest.index('--checkpoint')+1
    rest[index]=str(Path(rest[index]).expanduser().resolve())
os.chdir(source)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(source))
sys.argv = [str(entry)] + rest
exec(compile(text, str(entry), 'exec'), {'__name__': '__main__', '__file__': str(entry)})

