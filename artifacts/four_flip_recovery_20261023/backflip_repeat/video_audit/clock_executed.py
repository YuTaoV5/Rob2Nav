"""Simulation-clock frame scheduling; no wall-clock drift or silent frame skips."""
import math


class VideoClock:
    def __init__(self,fps):
        if type(fps) is not int or not 1<=fps<=50:raise ValueError('Invalid recording frame rate')
        self.fps=fps;self.timeline=[];self.last_time=-float('inf')

    def sample(self,time):
        if not math.isfinite(time) or time<self.last_time:raise ValueError('Invalid recording simulation time')
        self.last_time=time
        scheduled=len(self.timeline)/self.fps
        if time+1e-9<scheduled:return None
        if time-scheduled>=1/self.fps+1e-9:raise RuntimeError('Recording skipped a simulation frame')
        frame=dict(index=len(self.timeline),scheduled_sim_s=scheduled,captured_sim_s=time)
        self.timeline.append(frame)
        return frame

    def report(self,end):
        expected=math.floor(end*self.fps+1e-9)+1
        return dict(fps=self.fps,frames=len(self.timeline),expected_frames=expected,
            full_timeline_coverage=len(self.timeline)==expected,
            encoded_duration_s=len(self.timeline)/self.fps,
            first_frame_sim_s=self.timeline[0]['captured_sim_s'] if self.timeline else None,
            last_frame_sim_s=self.timeline[-1]['captured_sim_s'] if self.timeline else None,
            max_alignment_error_s=max((abs(x['captured_sim_s']-x['scheduled_sim_s']) for x in self.timeline),default=None),
            clock='frame index / fps against simulation time; no cumulative re-anchoring')
