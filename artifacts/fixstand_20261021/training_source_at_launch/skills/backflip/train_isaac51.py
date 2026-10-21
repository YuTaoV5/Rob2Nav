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
args, rest = parser.parse_known_args()
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
    text=text.replace('env_cfg = Go2BackflipEnvCfg()', 'env_cfg = Go2BackflipEnvCfg()\n    from skills.fixstand import configure_backflip\n    configure_backflip(env_cfg)')
    text=text.replace('init_at_random_ep_len=True','init_at_random_ep_len=False')
    if args.fixstand_output:
        output=args.fixstand_output.resolve()
        if output.exists():raise FileExistsError(output)
        text=text.replace('log_dir = (log_root / run_dir_name).resolve()',f'log_dir = Path({str(output)!r})')
        text=text.replace('raw_env = gym.make(TASK_ID, cfg=env_cfg)',"from skills.fixstand import write_contract\n    write_contract(log_dir / 'fixstand.json','backflip',actor_observations=60)\n    raw_env = gym.make(TASK_ID, cfg=env_cfg)")
os.chdir(source)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(source))
sys.argv = [str(entry)] + rest
exec(compile(text, str(entry), 'exec'), {'__name__': '__main__', '__file__': str(entry)})

