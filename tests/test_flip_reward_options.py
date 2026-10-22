import json
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import sys
import pytest
from skills.backflip.reward_options import parse_scales,apply_scales
from skills.backflip.launcher_patch import configure_fixstand_source


@pytest.mark.parametrize('text',['[]','null','{"unknown":1}','{"lin_vel_z":true}','{"lin_vel_z":NaN}','{"lin_vel_z":1001}'])
def test_invalid_scales_rejected(text):
    with pytest.raises(ValueError):parse_scales(text)


def test_only_named_reward_changes_after_fixstand_direction_setup():
    cfg=NS(rewards=NS(scales={'lin_vel_z':20.,'ang_vel_y':5.,'termination':-1000.}))
    apply_scales(cfg,{'lin_vel_z':60.})
    assert cfg.rewards.scales=={'lin_vel_z':60.,'ang_vel_y':5.,'termination':-1000.}
    assert cfg.reward_scale_overrides=={'lin_vel_z':60.}
    text=configure_fixstand_source('def f():\n    env_cfg = Go2BackflipEnvCfg()',Path('/source'),
        delayed_entry=True,flip_direction='front',reward_scales={'lin_vel_z':60.})
    assert text.index('configure_frontflip(env_cfg')<text.index('apply_scales(env_cfg')
    compile(text,'test-injection','exec')
    with pytest.raises(ValueError):apply_scales(NS(rewards=NS(scales={})),{'lin_vel_z':60.})


def test_web_weight_option_is_editable_and_delivered_without_shell(tmp_path):
    from studio.config import RunConfig,command
    from studio.registry import registry
    r=next(r for r in registry()['recipes'] if r['id']=='frontflip_native_ppo')
    assert r['option_schema']['reward_scales_json']['type']=='string'
    d=dict(r['defaults']);options=dict(d['recipe_options']);options['reward_scales_json']='{"lin_vel_z":60}'
    d.update(recipe_options=options,auto_evaluate=True)
    c=RunConfig(**d);cmd=command(c,tmp_path,tmp_path/'outputs/studio/runs/PREVIEW',tmp_path/'cfg.json','/isaac/python.sh')
    assert cmd[cmd.index('--reward-scales-json')+1]=='{"lin_vel_z":60}'
    options['reward_scales_json']='{"bad_key":1}'
    with pytest.raises(ValueError):RunConfig(**{**d,'recipe_options':options})


def test_launcher_parses_reward_options_without_ambient_pythonpath(tmp_path):
    script=Path(__file__).parents[1]/'skills/backflip/train_isaac51.py'
    result=subprocess.run([sys.executable,'-I',str(script),'--source',str(tmp_path),'--reward-scales-json','{"unknown":1}'],capture_output=True,text=True)
    assert result.returncode!=0 and 'Unknown native flip reward scale' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr
