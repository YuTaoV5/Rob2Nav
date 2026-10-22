import pytest
from studio.terrain_profiles import terrain_launch


@pytest.mark.parametrize('skill',['stairs','platform'])
def test_released_terrain_uses_the_retested_roundtrip_protocol(skill):
    args,note=terrain_launch(skill)
    assert '--roundtrip' in args and '--no-jump' in args
    assert args[args.index('--seed')+1]=='12010'
    assert '2秒' in note
    if skill=='platform':assert args[args.index('--height')+1]=='.20'


def test_mixed_scene_cannot_disable_its_required_gap_skill():
    args,note=terrain_launch('multiterrain')
    assert args[args.index('--scene')+1]=='mixed'
    assert '--no-jump' not in args
    assert '未验收' in note
