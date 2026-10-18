from pathlib import Path
import xml.etree.ElementTree as ET
from fastapi.testclient import TestClient
import pytest
from studio.api import create_app
from studio.visuals import go2_urdf,terrain_data

ROOT=Path(__file__).resolve().parents[2]


def test_display_go2_preserves_kinematic_limits_and_meshes():
    r=ET.fromstring(go2_urdf(ROOT))
    assert len(r.findall('link'))==17
    joints=r.findall("joint[@type='revolute']")
    assert len(joints)==12
    knee=r.find("joint[@name='FL_calf_joint']")
    assert knee.find('limit').get('lower')=='-2.7227'
    assert knee.find('origin').get('xyz')=='0 0 -0.213'
    assert knee.find('axis').get('xyz')=='0 1 0'
    for mesh in r.findall('.//mesh'):
        assert (ROOT/'vendor/RoboGauge/resources/robots/go2/assets'/mesh.get('filename').split('/')[-1]).is_file()


@pytest.mark.parametrize('kind',['ame_stakes','ame_alternate'])
def test_final_wide_course_is_exact_original_geometry(kind):
    from perceptive_rl.ame_terrain import boxes
    for course in ('original','wide_deep'):
        data=terrain_data(kind,course,1.)
        assert data['boxes']==boxes(1.,kind)
        assert data['parameters']['side']==pytest.approx(.2)
        assert min(b[4] for b in data['boxes'])==-2.


def test_visual_endpoints_are_read_only_and_scoped():
    with TestClient(create_app(ROOT)) as c:
        assert c.get('/api/robot/go2.urdf').status_code==200
        assert c.get('/api/robot/assets/base_0.obj').status_code==200
        assert c.get('/api/robot/assets/go2.xml').status_code==404
        assert c.get('/api/terrain-preview?kind=unknown').status_code==422
        assert c.get('/api/terrain-preview?difficulty=nan').status_code==422
        assert c.get('/api/terrain-preview?difficulty=2').status_code==422
        r=c.get('/api/terrain-preview?kind=stairs&image=true')
        assert r.status_code==200 and r.headers['content-type'].startswith('image/svg+xml')
        assert '<svg' in r.text
