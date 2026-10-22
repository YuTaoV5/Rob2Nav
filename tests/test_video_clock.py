import pytest
from rob2nav.video_clock import VideoClock


@pytest.mark.parametrize('fps',[10,25,50])
def test_long_recording_covers_every_simulation_frame_without_phase_drift(fps):
    clock=VideoClock(fps);end=189.842
    for i in range(1,round(end/.002)+1):clock.sample(i*.002)
    report=clock.report(end)
    assert report['full_timeline_coverage']
    assert report['frames']==int(end*fps)+1
    assert report['max_alignment_error_s']<.002000001
    assert abs(report['encoded_duration_s']-end)<=1/fps


def test_clock_does_not_hide_missing_or_reversed_simulation_time():
    clock=VideoClock(50);clock.sample(.002)
    with pytest.raises(RuntimeError,match='skipped'):clock.sample(.05)
    with pytest.raises(ValueError):clock.sample(.001)
    with pytest.raises(ValueError):VideoClock(True)
