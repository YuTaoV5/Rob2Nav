import math
import pytest
from rob2nav.sea_contract import ray_angles,load_native_cbf


def test_sea_ray_contract_extremes_and_center():
    a=ray_angles()
    assert len(a)==41
    assert math.degrees(a[0])==pytest.approx(-120)
    assert math.degrees(a[20])==pytest.approx(0)
    assert math.degrees(a[-1])==pytest.approx(120)
    assert math.cos(a[0])==pytest.approx(-.5)
    assert all(math.degrees(y-x)==pytest.approx(6) for x,y in zip(a,a[1:]))


def test_invalid_source_and_geometry_rejected(tmp_path):
    for n,fov in [(40,240),(True,240),(41,float('nan')),(41,0)]:
        with pytest.raises(ValueError):ray_angles(n,fov)
    source=tmp_path/'fake.py';source.write_text('raise RuntimeError("must never execute")')
    with pytest.raises(ValueError,match='hash mismatch'):load_native_cbf(source)
