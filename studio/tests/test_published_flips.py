import hashlib
import json
from pathlib import Path
import pytest
from studio.published_flips import live_flip

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('skill,version,seconds',[('sideflip','peak_pose_v8','8'),('backflip','upstream_actor_v7','5')])
def test_published_contract_and_native_launch(skill,version,seconds):
    module,args,policy,note=live_flip(ROOT,skill,Path('/pinned/source'))
    assert version in str(policy)
    assert '--fixstand-roundtrip' in args
    assert args[args.index('--seconds')+1]==seconds
    assert args[args.index('--physics-hz')+1]=='1000'
    assert ('0/4' if skill=='sideflip' else '4/4') in note
    metadata=json.loads(policy.with_name('policy.json' if skill=='sideflip' else 'export.json').read_text())
    assert metadata['policy_sha256']==hashlib.sha256(policy.read_bytes()).hexdigest()
    if skill=='sideflip':
        assert module=='rob2nav.mujoco_skill'
        reference=Path(args[args.index('--motion')+1])
        assert hashlib.sha256(reference.read_bytes()).hexdigest()==metadata['reference_sha256']
        assert metadata['fixstand_roundtrip']['terminal_hip_taper_frames']==50
    else:
        assert module=='skills.backflip.simulate'
        assert metadata['entry_training_contract_verified']
        assert metadata['control_contract']['action_clip']==8
        assert metadata['fixstand_roundtrip']['entry_hold_scale']==.1


def test_published_resources_fail_closed_on_mismatch(tmp_path):
    (tmp_path/'studio').mkdir()
    (tmp_path/'model.pt').write_bytes(b'changed')
    row=dict(policy='model.pt',sha256={'model.pt':'0'*64})
    (tmp_path/'studio/published_flips.json').write_text(json.dumps({'sideflip':row}))
    with pytest.raises(ValueError,match='hash mismatch'):
        live_flip(tmp_path,'sideflip',tmp_path)


def test_catalog_selects_current_candidates_only():
    data=json.loads((ROOT/'studio/model_assets.json').read_text(encoding='utf-8'))
    for skill in ('sideflip','backflip'):
        rows=[m for m in data['models'] if m['skill']==skill and m['role']=='selected']
        assert rows
        assert all(('fixstand_best_flips_20261022' if skill=='sideflip' else 'backflip_v7_verified_20261022') in m['path'] for m in rows)
        assert all(('0/4' if skill=='sideflip' else '4/4') in m['validation'] for m in rows)
        assert all((ROOT/r['path']).is_file() for m in rows for r in m['resources'])
