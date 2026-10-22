import copy,json
from types import SimpleNamespace as NS
import pytest
from skills.sideflip.direction import reference_direction,require_reference_direction,inference_direction
from rob2nav.fixstand_validation import rotation_matches
from studio.config import RunConfig,command
from studio.registry import registry
from studio.recipe_training import prerequisites


def test_left_and_right_cannot_pass_each_others_rotation():
    assert reference_direction({})=='left'
    assert inference_direction({'sideflip_direction':'right'},{})=='right'
    assert rotation_matches(-6.28,'left') and not rotation_matches(-6.28,'right')
    assert rotation_matches(6.28,'right') and not rotation_matches(6.28,'left')


@pytest.mark.parametrize('reference,policy', [({'sideflip_direction':'left'},{'sideflip_direction':'right'}),
    ({'sideflip_direction':'right'},{'sideflip_direction':'left'}),({'sideflip_direction':'front'},{}),
    ({},{'sideflip_direction':True})])
def test_reference_policy_mismatch_rejected(reference,policy):
    with pytest.raises(ValueError):inference_direction(reference,policy)


def test_right_form_choice_reaches_the_trainer_and_preflight_checks_reference(tmp_path):
    ref=tmp_path/'outputs/right/reference.npz';ref.parent.mkdir(parents=True);ref.write_bytes(b'reference')
    sidecar=ref.with_suffix('.json');sidecar.write_text(json.dumps({'sideflip_direction':'right'}))
    data=copy.deepcopy(next(r for r in registry()['recipes'] if r['id']=='sideflip_native_ppo')['defaults'])
    data['recipe_options'].update(sideflip_direction='right',motion='outputs/right/reference.npz')
    cfg=RunConfig.model_validate(data)
    cmd=command(cfg,tmp_path,tmp_path/'outputs/studio/runs/PREVIEW',tmp_path/'preview.json','/isaac/python.sh')
    assert cmd[cmd.index('--sideflip-direction')+1]=='right'
    sidecar.write_text(json.dumps({'sideflip_direction':'left'}))
    assert any('reference direction is left, requested right' in x for x in prerequisites(cfg,tmp_path))
