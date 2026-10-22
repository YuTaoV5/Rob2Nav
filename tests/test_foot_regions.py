import numpy as np
import sys
from types import SimpleNamespace as NS
from rob2nav.ground_contacts import foot_region_leg

CENTERS={leg:np.array([i*.3,0.,.022]) for i,leg in enumerate(('FR','FL','RR','RL'))}
RADII={leg:.022 for leg in CENTERS}


def test_calf_toe_endcap_inside_own_foot_region_is_valid():
    assert foot_region_leg('FR_calf',CENTERS['FR']+[0.,0.,-.015],CENTERS,RADII)=='FR'


def test_proximal_leg_and_knee_contacts_cannot_be_foot_support():
    assert foot_region_leg('FR_calf',CENTERS['FR']+[0.,0.,.10],CENTERS,RADII) is None
    assert foot_region_leg('FR_thigh',CENTERS['FR'],CENTERS,RADII) is None


def test_matching_own_leg_prevents_borrowing_another_foot_region():
    assert foot_region_leg('FR_calf',CENTERS['FL'],CENTERS,RADII) is None


def test_body_and_head_contacts_near_a_foot_still_fail():
    for body in ('base','Head_upper','FR_hip'):
        assert foot_region_leg(body,CENTERS['FR'],CENTERS,RADII) is None


def test_region_is_derived_from_asset_radius_not_a_tuned_global_threshold():
    radii=dict(RADII);radii['FR']=.01
    assert foot_region_leg('FR_calf',CENTERS['FR']+[0.,0.,-.015],CENTERS,radii) is None


def test_force_reader_counts_toe_endcap_and_foot_sphere_but_not_knee(monkeypatch):
    from rob2nav.ground_contacts import GroundContactReader
    def force(model,data,index,out):out[:]=0.;out[0]=[12.,30.,8.][index]
    monkeypatch.setitem(sys.modules,'mujoco',NS(mj_contactForce=force))
    ids=np.zeros(60,dtype=int);ids[3]=1;ids[20]=2
    reader=object.__new__(GroundContactReader)
    reader.feet=dict(zip(('FR','FL','RR','RL'),(20,33,46,59)))
    reader.direct={gid:leg for leg,gid in reader.feet.items()};reader.floors={0};reader.radii=RADII
    reader.model=NS(geom_bodyid=ids,body=lambda i:NS(name={1:'FR_calf',2:'FR_foot'}[i]))
    positions=np.zeros((60,3))
    for leg,gid in reader.feet.items():positions[gid]=CENTERS[leg]
    contacts=[NS(geom1=0,geom2=3,pos=CENTERS['FR']+[0.,0.,-.015]),
              NS(geom1=0,geom2=3,pos=CENTERS['FR']+[0.,0.,.10]),
              NS(geom1=0,geom2=20,pos=CENTERS['FR']+[0.,0.,-.0225])]
    data=NS(contact=contacts,geom_xpos=positions)
    before=positions.copy();frame=reader.sample(data)
    assert frame.forces['FR']==20. and len(frame.unexpected)==1
    assert frame.unexpected[0]['force_N']==30.
    np.testing.assert_array_equal(positions,before)


def test_whole_sequence_requires_samples_and_complete_coverage():
    from rob2nav.ground_contacts import GroundContactValidation
    check=object.__new__(GroundContactValidation)
    check.reader=NS(radii=RADII,rule_sha256='fixture');check.hz=1000.
    check.samples=0;check.bad_samples=0;check.violations={}
    assert not check.report(0)['passed']
    check.samples=4
    assert not check.report(5)['passed']
    assert check.report(4)['passed']


def test_single_bad_substep_fails_even_with_zero_contact_force():
    from rob2nav.ground_contacts import GroundContactValidation,GroundFrame
    check=object.__new__(GroundContactValidation)
    unexpected=dict(geom=3,body='FR_calf',force_N=0.,own_foot_center_distance_m=.10)
    check.reader=NS(radii=RADII,rule_sha256='fixture',sample=lambda data:GroundFrame({},[],[unexpected]))
    check.hz=1000.;check.samples=0;check.bad_samples=0;check.violations={}
    check.sample(NS(time=.001))
    report=check.report(1)
    assert report['coverage_complete'] and not report['passed']
    assert report['unexpected_samples']==1 and report['violations']['3']['body']=='FR_calf'
