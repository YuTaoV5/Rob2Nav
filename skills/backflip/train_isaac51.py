"""Pinned-source launcher for IsaacLab 2.3 / IsaacSim 5.1 URDF API mismatch.
Does not alter global IsaacLab or physics parameters.
"""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

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
parser.add_argument('--flip-direction',choices=['back','front'],default='back')
parser.add_argument('--initial-std',type=float)
parser.add_argument('--peak-contact-forces',action='store_true')
parser.add_argument('--reward-scales-json',default='{}')
parser.add_argument('--rebound-cycles',type=int,choices=[1,2,3],default=1)
parser.add_argument('--rebound-max-seconds',type=float,default=.45)
parser.add_argument('--rebound-phase-version',choices=['legacy_v1','landing_v2'],default='legacy_v1')
parser.add_argument('--rebound-landing-max-seconds',type=float,default=.25)
parser.add_argument('--rebound-min-air-rotation',type=float,default=3.141592653589793)
parser.add_argument('--rebound-landing-upright-cos',type=float,default=.9)
parser.add_argument('--rebound-target-rate-weight',type=float,default=0.)
parser.add_argument('--rebound-target-rate-margin',type=float,default=.8)
parser.add_argument('--rebound-absorption-upright-weight',type=float,default=0.)
parser.add_argument('--rebound-absorption-height-weight',type=float,default=0.)
parser.add_argument('--rebound-teacher-data',default='')
parser.add_argument('--rebound-teacher-manifest-sha256',default='')
parser.add_argument('--rebound-teacher-weight',type=float,default=0.)
parser.add_argument('--rebound-teacher-learning-rate',type=float,default=.00001)
parser.add_argument('--rebound-teacher-batch-size',type=int,default=128)
args, rest = parser.parse_known_args()
from skills.backflip.reward_options import parse_scales
reward_scales=parse_scales(args.reward_scales_json)
if reward_scales and not args.fixstand_roundtrip:raise ValueError('Reward overrides require FIXSTAND training')
if args.peak_contact_forces and not args.fixstand_roundtrip:raise ValueError('Peak contacts require FIXSTAND training')
if args.delayed_entry and not args.fixstand_roundtrip:raise ValueError('Delayed entry requires fixstand training')
if not 0.<=args.entry_hold_scale<=.4 or (args.entry_hold_scale!=.4 and not args.delayed_entry):raise ValueError('Entry hold scale requires delayed entry and [0, 0.4]')
if args.fast_safety_curriculum and not args.urdf_speed_limits:raise ValueError('Fast safety curriculum requires URDF speed limits')
if args.safety_start is not None and (not args.fast_safety_curriculum or not 0.<=args.safety_start<=1.):raise ValueError('Safety start requires fast curriculum and [0,1]')
if args.finetune_learning_rate is not None and not 0.<args.finetune_learning_rate<=.001:raise ValueError('Invalid finetune learning rate')
if args.flip_direction=='front' and not (args.fixstand_roundtrip and args.delayed_entry):raise ValueError('Frontflip requires the standard delayed-entry task')
if args.initial_std is not None and not .01<=args.initial_std<=1.:raise ValueError('Invalid initial action noise')
if args.rebound_cycles != 1:
    if not (args.fixstand_roundtrip and args.delayed_entry and args.flip_direction=='back'):
        raise ValueError('Rebound requires delayed FIXSTAND backflip')
    from skills.backflip.rebound import ReboundConfig
    ReboundConfig(cycles=args.rebound_cycles,max_rebound_seconds=args.rebound_max_seconds)
elif args.rebound_max_seconds != .45:
    raise ValueError('Rebound interval requires multiple cycles')
landing_options=None
from studio.stability_options import defaults,validate
_validation=defaults('backflip_native_ppo')
_validation.update({k:getattr(args,k) for k in _validation if k.startswith('rebound_')})
_validation.update(fixstand_roundtrip=args.fixstand_roundtrip,delayed_entry=args.delayed_entry)
validate('backflip_native_ppo',_validation)
if args.rebound_phase_version=='landing_v2':
    landing_options=dict(landing_max_seconds=args.rebound_landing_max_seconds,
        min_air_rotation=args.rebound_min_air_rotation,landing_upright_cos=args.rebound_landing_upright_cos,
        target_rate_weight=args.rebound_target_rate_weight,target_rate_margin=args.rebound_target_rate_margin)
    if args.rebound_absorption_upright_weight or args.rebound_absorption_height_weight:
        landing_options.update(absorption_upright_weight=args.rebound_absorption_upright_weight,
                               absorption_height_weight=args.rebound_absorption_height_weight)
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
    from skills.backflip.launcher_patch import configure_fixstand_source
    text=configure_fixstand_source(text,source,args.urdf_speed_limits,args.fast_safety_curriculum,
        args.delayed_entry,args.entry_hold_scale,args.flip_direction,args.peak_contact_forces,reward_scales,args.rebound_cycles,args.rebound_max_seconds,args.rebound_phase_version,landing_options)
    if args.safety_start is not None:
        text=text.replace('agent_cfg = Go2BackflipPPORunnerCfg()',
            f'env_cfg.rewards.safety_curriculum_start={args.safety_start!r}\n    agent_cfg = Go2BackflipPPORunnerCfg()')
    if args.finetune_learning_rate is not None:
        needle='agent_cfg = Go2BackflipPPORunnerCfg()'
        if text.count(needle)!=1:raise RuntimeError('Pinned agent config entry changed')
        text=text.replace(needle,needle+f'\n    agent_cfg.algorithm.learning_rate={args.finetune_learning_rate!r}\n    agent_cfg.algorithm.schedule="fixed"')
    if args.initial_std is not None:
        text=text.replace('agent_cfg = Go2BackflipPPORunnerCfg()',f'agent_cfg = Go2BackflipPPORunnerCfg()\n    agent_cfg.policy.init_noise_std={args.initial_std!r}')
    text=text.replace('init_at_random_ep_len=True','init_at_random_ep_len=False')
    if args.rebound_teacher_weight:
        if not args.fixstand_output:raise ValueError('Teacher retention needs recorded FIXSTAND output')
        from skills.backflip.teacher_data import load_reference
        teacher_path=(ROOT/args.rebound_teacher_data).resolve()
        if not any(teacher_path.is_relative_to(ROOT/folder) for folder in ('outputs','artifacts','skills')):
            raise ValueError('Teacher directory outside project artifacts')
        load_reference(teacher_path,args.rebound_teacher_manifest_sha256)
        settings=dict(directory=str(teacher_path),manifest_sha256=args.rebound_teacher_manifest_sha256,
            weight=args.rebound_teacher_weight,learning_rate=args.rebound_teacher_learning_rate,
            batch_size=args.rebound_teacher_batch_size,hold_scale=args.entry_hold_scale)
        needle='runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=False)'
        if text.count(needle)!=1:raise RuntimeError('Pinned learning entry changed')
        text=text.replace(needle,f'from skills.backflip.teacher_retention import configure_runner\n    configure_runner(runner,log_dir,seed=agent_cfg.seed,**{settings!r})\n    '+needle)
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
        if args.rebound_cycles != 1:
            text=text.replace("action_parameterization='backflip_entry_v3'","action_parameterization='backflip_rebound_v1'").replace("action_parameterization='backflip_entry_v2'","action_parameterization='backflip_rebound_v1'")
            text=text.replace('episode_seconds=5.)',f'episode_seconds={3.+2.*args.rebound_cycles!r},rebound_cycles={args.rebound_cycles!r},rebound_max_seconds={args.rebound_max_seconds!r},phase_contract="contact_cycle_v1")')
        if args.rebound_phase_version=='landing_v2':
            text=text.replace("action_parameterization='backflip_rebound_v1'","action_parameterization='backflip_rebound_landing_v2'")
            text=text.replace('phase_contract="contact_cycle_v1"',f'phase_contract="landing_cycle_v2",rebound_landing_options={landing_options!r}')
        if args.flip_direction=='front':
            text=text.replace("'backflip',actor_observations=60", "'frontflip',actor_observations=60,flip_direction='front'")
        if args.peak_contact_forces:
            needle='actor_observations=60'
            if text.count(needle)!=1:raise RuntimeError('Pinned FIXSTAND contract entry changed')
            text=text.replace(needle,needle+',peak_contact_forces=True')
if '--checkpoint' in rest:
    index=rest.index('--checkpoint')+1
    rest[index]=str(Path(rest[index]).expanduser().resolve())
os.chdir(source)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(source))
sys.argv = [str(entry)] + rest
exec(compile(text, str(entry), 'exec'), {'__name__': '__main__', '__file__': str(entry)})

