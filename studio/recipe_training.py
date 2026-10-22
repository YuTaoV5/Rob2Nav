"""Allowlisted, skill-specific training contracts used by Studio jobs."""
from pathlib import Path
import subprocess
import json
from .stability_options import SCHEMAS, defaults as stability_defaults, validate as validate_stability, arguments as stability_arguments

OPTIONS = {
    'navigation_ppo': {'sensors': ('lidar', 'depth', 'both'), 'scene': ('empty', 'random', 'slalom'), 'steps': (512, 10_000_000), 'envs': (1, 64)},
    'sideflip_native_ppo': {'device': ('cuda:0',), 'transfer_motors': (False, True)},
    'upright_walk_ppo': {'device': ('cuda:0',), 'transfer_motors': (False, True)},
    'backflip_native_ppo': {'device': ('cuda:0',)},
}

DEFAULTS = {
    'navigation_ppo': {'sensors': 'lidar', 'scene': 'random', 'steps': 100000, 'envs': 4},
    'sideflip_native_ppo': {'device': 'cuda:0', 'transfer_motors': False},
    'upright_walk_ppo': {'device': 'cuda:0', 'transfer_motors': False},
    'backflip_native_ppo': {'device': 'cuda:0'},
}
OPTIONS['frontflip_native_ppo'] = dict(OPTIONS['backflip_native_ppo'])
DEFAULTS['frontflip_native_ppo'] = dict(DEFAULTS['backflip_native_ppo'])


def validate_options(recipe_id, options):
    spec = OPTIONS[recipe_id]
    unknown = set(options) - set(spec) - set(SCHEMAS.get(recipe_id, {}))
    if unknown:
        raise ValueError('Unsupported recipe option: ' + ', '.join(sorted(unknown)))
    for key, value in options.items():
        if key in SCHEMAS.get(recipe_id, {}): continue
        allowed = spec[key]
        if len(allowed) == 2 and all(type(x) is int for x in allowed):
            if type(value) is not int or not allowed[0] <= value <= allowed[1]:
                raise ValueError(f'{key} must be an integer from {allowed[0]} to {allowed[1]}')
        elif value not in allowed or type(value) is not type(allowed[0]):
            raise ValueError(f'Invalid {key} for {recipe_id}')
    validate_stability(recipe_id, {**DEFAULTS[recipe_id], **options})


def options_for(config):
    return {**DEFAULTS[config.recipe_id], **stability_defaults(config.recipe_id), **config.recipe_options}


def motion_path(config, root):
    value = options_for(config).get('motion', '')
    if not value: return Path(root).parent / 'capability-work/sideflip/reference.npz'
    root = Path(root).resolve()
    path = (root / value).resolve()
    if path.suffix != '.npz' or not path.is_file() or not any(path.is_relative_to(root / folder) for folder in ('outputs', 'artifacts', 'skills')):
        raise ValueError('Motion must be an existing project NPZ artifact')
    return path


def prerequisites(config, root):
    root = Path(root)
    rid = config.recipe_id
    required = []
    if rid == 'navigation_ppo':
        required = [root / 'skills/navigation/config.json', root / 'vendor/RoboGauge']
    elif rid in ('sideflip_native_ppo', 'upright_walk_ppo', 'backflip_native_ppo', 'frontflip_native_ppo'):
        required = [root.parent / 'Rob2Nav-upstreams/GO2_backflip']
        if rid not in ('backflip_native_ppo', 'frontflip_native_ppo'):
            required.append(root.parent / 'Rob2Nav-upstreams/unitree_rl_lab')
        if rid == 'sideflip_native_ppo':
            try:
                reference=motion_path(config, root);sidecar=reference.with_suffix('.json')
                required.extend([reference,sidecar])
                if sidecar.is_file():
                    from skills.sideflip.direction import require_reference_direction
                    require_reference_direction(json.loads(sidecar.read_text()),options_for(config)['sideflip_direction'])
            except ValueError as exc: return [str(exc)]
    missing = [str(path) for path in required if not path.exists()]
    pins = {'GO2_backflip': '954aa61ceb62c515689677f44a21e32109f6b62d',
            'unitree_rl_lab': '4960b84732b0c2ec593dccbfe963fda1bcd7b1e3'}
    for name, commit in pins.items():
        path = root.parent / 'Rob2Nav-upstreams' / name
        if path in required and path.exists():
            try:
                actual = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True,timeout=5).strip()
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                actual = ''
            if actual != commit: missing.append(f'{path}: expected commit {commit}')
    return missing


def resume_path(config, root, suffix):
    if config.initialization not in ('base','resume'):
        raise ValueError('This skill trainer supports base or resume initialization only')
    if config.initialization != 'resume':
        if config.checkpoint: raise ValueError('Choose resume before selecting a skill checkpoint')
        return None
    if not config.checkpoint: raise ValueError('Select a checkpoint for resume')
    path = (Path(root) / config.checkpoint).resolve()
    if path.suffix != suffix or not path.is_file() or not any(path.is_relative_to((Path(root)/folder).resolve()) for folder in ('outputs','artifacts','skills')):
        raise ValueError('Resume checkpoint must be an existing project artifact of the correct format')
    return path


def recipe_command(config, root, output, isaac_python, mujoco_python=None):
    root = Path(root).resolve(); output = Path(output).resolve()
    if not output.is_relative_to(root / 'outputs/studio/runs'):
        raise ValueError('Training output must be inside Studio runs')
    rid = config.recipe_id; opt = options_for(config)
    validate_options(rid, config.recipe_options)
    if config.auto_evaluate and 'evaluation_seed_start' in opt and opt['evaluation_seed_start']<=config.seed<opt['evaluation_seed_start']+4:
        raise ValueError('Independent evaluation seeds must not include the training seed')
    if rid == 'navigation_ppo':
        cmd = [str(mujoco_python or isaac_python), '-m', 'go2_nav.train', '--config', str(root/'skills/navigation/config.json'),
                '--sensors', opt['sensors'], '--scene', opt['scene'], '--envs', str(opt['envs']), '--steps', str(opt['steps']),
                '--seed', str(config.seed), '--output', str(output)]
        resume = resume_path(config, root, '.zip')
        if resume: cmd += ['--resume', str(resume)]
        return cmd
    source = root.parent / 'Rob2Nav-upstreams/GO2_backflip'
    if rid in ('backflip_native_ppo', 'frontflip_native_ppo'):
        resume = resume_path(config, root, '.pt')
        if resume and not opt['fixstand_roundtrip']: raise ValueError('Flip resume requires FIXSTAND output contract')
        if opt['reset_optimizer'] and not resume: raise ValueError('Reset optimizer requires resume')
        cmd = [str(isaac_python), str(root/'skills/backflip/train_isaac51.py'), '--source', str(source), '--headless',
                '--device', opt['device'], '--num_envs', str(config.num_envs), '--trainer', 'rsl_rl',
                '--max_iterations', str(config.iterations), '--seed', str(config.seed), '--run_name', 'studio_'+output.name]
        if opt['fixstand_roundtrip']: cmd += ['--fixstand-output', str(output)]
        if resume: cmd += ['--resume', '--checkpoint', str(resume)]
        return cmd + stability_arguments(rid, opt)
    if rid in ('sideflip_native_ppo', 'upright_walk_ppo'):
        module = 'skills.sideflip.train' if rid == 'sideflip_native_ppo' else 'skills.upright_walk.train'
        cmd = [str(isaac_python), '-m', module, '--backflip-source', str(source),
               '--unitree-source', str(root.parent/'Rob2Nav-upstreams/unitree_rl_lab'), '--output', str(output),
               '--num-envs', str(config.num_envs), '--iterations', str(config.iterations), '--seed', str(config.seed),
               '--device', opt['device'], '--headless']
        if rid == 'sideflip_native_ppo':
            cmd += ['--motion', str(motion_path(config, root))]
        resume = resume_path(config, root, '.pt')
        if resume: cmd += ['--resume', str(resume)]
        if opt['transfer_motors']: cmd += ['--transfer-motors']
        return cmd + stability_arguments(rid, opt)
    raise ValueError('Unsupported training recipe')


def skill_evaluation_command(config,root,run,cpu_python):
    """The same strict four-seed evaluator used for reviewed acrobatics runs."""
    rid=config.recipe_id
    kinds={'sideflip_native_ppo':'sideflip','backflip_native_ppo':'backflip','frontflip_native_ppo':'frontflip'}
    if not config.auto_evaluate or rid not in kinds:return None
    opt=options_for(config)
    if not opt['fixstand_roundtrip']:return None
    if not cpu_python:raise ValueError('Strict skill evaluation requires the MuJoCo CPU runtime')
    start=opt['evaluation_seed_start']
    if start<=config.seed<start+4:raise ValueError('Independent evaluation seeds must not include the training seed')
    root=Path(root).resolve()
    cmd=[str(cpu_python),str(root/'scripts/evaluate_fixstand_run.py'),'--kind',kinds[rid],
         '--run',str(run),'--seed-start',str(start),'--source',str(root.parent/'Rob2Nav-upstreams/GO2_backflip')]
    if rid=='sideflip_native_ppo':cmd+=['--motion',str(motion_path(config,root))]
    return cmd
