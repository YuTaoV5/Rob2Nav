import copy
import json
from pathlib import Path
import pytest
from scripts.check_sensor_calibration import check,rigid

ROOT=Path(__file__).resolve().parents[2]
def example():return json.loads((ROOT/'deployment/sensors/calibration.synthetic_60deg.json').read_text(encoding='utf-8'))

def test_example_requires_opt_in_and_never_qualifies_hardware():
    d=example()
    with pytest.raises(ValueError):check(d)
    result=check(d,True)
    assert result['offline_checks_passed'] and result['hardware_release'] is False
    assert result['position_max_m']<1e-12
    assert all(p['inside_geometric_view'] for p in result['workspace_projections'])

def test_wrong_translation_fov_and_stale_measurements_fail():
    d=example();d['T_base_color_optical'][0][3]+=.03
    assert not check(d,True)['offline_checks_passed']
    d=example();d['workspace_points_base'].append([100,100,100])
    assert not check(d,True)['offline_checks_passed']
    d=example();d['timing']['measurement_age_p99_ms']=101
    assert not check(d,True)['offline_checks_passed']

@pytest.mark.parametrize('bad',[None,[[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]],[[float('nan'),0,0,0]]*4])
def test_missing_reflected_or_nonfinite_transform_rejected(bad):
    with pytest.raises(ValueError):rigid(bad)

def test_unmeasured_template_rejected():
    d=json.loads((ROOT/'deployment/sensors/calibration.template.json').read_text(encoding='utf-8'))
    with pytest.raises(ValueError):check(d)
