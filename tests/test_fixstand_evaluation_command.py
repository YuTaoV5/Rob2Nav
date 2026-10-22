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
