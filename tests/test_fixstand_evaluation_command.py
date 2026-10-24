from pathlib import Path
import pytest
import subprocess,sys
from scripts.evaluate_fixstand_run import simulation_arguments


def test_script_bootstraps_repository_without_ambient_pythonpath(tmp_path):
    script=Path(__file__).resolve().parents[1]/'scripts/evaluate_fixstand_run.py'
    code='import runpy,sys; m=runpy.run_path(sys.argv[1]); assert str(m["ROOT"]) in sys.path'
    subprocess.run([sys.executable,'-I','-c',code,str(script)],cwd=tmp_path,check=True)


def test_sideflip_requires_explicit_matching_reference():
    with pytest.raises(ValueError,match='--motion'):
        simulation_arguments('sideflip',Path('run'),Path('source'))
    cmd=simulation_arguments('sideflip',Path('run'),Path('source'),Path('right.npz'))
    assert cmd[cmd.index('--motion')+1]=='right.npz'


def test_stationary_steps_have_zero_velocity_and_flips_preserve_direction():
    cmd=simulation_arguments('upright_step',Path('run'),Path('source'))
    assert cmd[cmd.index('--command')+1:]==['0','0','0']
    for kind,direction in [('frontflip','front'),('backflip','back')]:
        cmd=simulation_arguments(kind,Path('run'),Path('source'))
        assert cmd[cmd.index('--direction')+1]==direction


def test_landing_v2_routes_explicitly_and_rejects_wrong_kind(tmp_path):
    import json
    from types import SimpleNamespace
    from studio.recipe_training import skill_evaluation_command
    cfg=SimpleNamespace(recipe_id='backflip_native_ppo',auto_evaluate=True,seed=73,
        recipe_options=dict(fixstand_roundtrip=True,delayed_entry=True,rebound_cycles=3,
                            rebound_phase_version='landing_v2',evaluation_seed_start=28700))
    command=skill_evaluation_command(cfg,tmp_path,tmp_path/'run','python')
    assert command[command.index('--kind')+1]=='backflip_landing'
    (tmp_path/'fixstand.json').write_text(json.dumps(dict(action_parameterization='backflip_rebound_landing_v2',episode_seconds=9)))
    args=simulation_arguments('backflip_landing',tmp_path,Path('source'))
    assert args[0]=='skills.backflip.simulate_landing'
    assert args[args.index('--seconds')+1]=='9'
    with pytest.raises(ValueError):simulation_arguments('backflip_rebound',tmp_path,Path('source'))
    cfg.recipe_options['evaluation_seed_start']=73
    with pytest.raises(ValueError):skill_evaluation_command(cfg,tmp_path,tmp_path/'run','python')
