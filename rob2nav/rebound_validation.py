"""Independent raw-physics flight segmentation; no training phase counters.

This checks motion continuity only. Endpoint stance, preparation, issued targets,
joint limits/speeds and visual review remain separate required checks.
"""
import math


class ContinuousFlipAudit:
    def __init__(self, dt, cycles=3, max_rebound_seconds=.45,
                 takeoff_confirm_seconds=.02, min_airtime_seconds=.08,
                 min_rotation=5.5, max_rotation=7.1, min_axis_dominance=.75):
        values=(dt,max_rebound_seconds,takeoff_confirm_seconds,min_airtime_seconds,
                min_rotation,max_rotation,min_axis_dominance)
        if cycles not in (2,3) or not all(math.isfinite(x) and x>0 for x in values):
            raise ValueError('Invalid continuous flip thresholds')
        if not min_rotation<max_rotation or min_axis_dominance>1 or min_airtime_seconds<takeoff_confirm_seconds:
            raise ValueError('Invalid flight acceptance thresholds')
        self.dt=dt;self.cycles=cycles;self.budget=max_rebound_seconds
        self.confirm=takeoff_confirm_seconds;self.min_air=min_airtime_seconds
        self.min_rotation=min_rotation;self.max_rotation=max_rotation;self.dominance=min_axis_dominance
        self.samples=0;self.last=None;self.covered=True;self.finite=True
        self.body_contact=False;self.flights=[];self.air=None

    def sample(self, time, foot_contact, omega_body, body_contact=False):
        if type(foot_contact) is not bool or type(body_contact) is not bool:
            raise ValueError('Contact flags must be explicit booleans')
        if len(omega_body)!=3:raise ValueError('Expected body angular velocity vector')
        if not math.isfinite(time) or not all(math.isfinite(x) for x in omega_body):
            self.finite=False;return
        expected=self.dt if self.last is None else self.last+self.dt
        self.covered &= abs(time-expected)<=max(1e-7,self.dt*1e-4)
        self.last=time;self.samples+=1;self.body_contact |= body_contact
        if not foot_contact:
            if self.air is None:
                self.air=dict(start_s=time,end_s=time,samples=0,pitch_rad=0.,angular_travel_rad=0.)
            a=self.air;a['end_s']=time;a['samples']+=1
            a['pitch_rad']+=omega_body[1]*self.dt
            a['angular_travel_rad']+=math.sqrt(sum(x*x for x in omega_body))*self.dt
        elif self.air is not None:
            a=self.air;self.air=None
            a['duration_s']=a['samples']*self.dt
            # Ignore only shorter-than-confirmation contact chatter. Confirmed
            # small hops remain candidates and therefore cannot hide extra jumps.
            if a['duration_s']+1e-7>=self.confirm:
                a['axis_dominance']=abs(a['pitch_rad'])/max(a['angular_travel_rad'],1e-12)
                a['motion_pass']=bool(a['duration_s']+1e-7>=self.min_air and
                    -self.max_rotation<=a['pitch_rad']<=-self.min_rotation and
                    a['axis_dominance']>=self.dominance)
                self.flights.append(a)

    def report(self, expected_samples):
        coverage=bool(self.covered and self.finite and self.samples==expected_samples and self.samples>0)
        unfinished=bool(self.air is not None and self.air['samples']*self.dt+1e-7>=self.confirm)
        gaps=[]
        for previous,following in zip(self.flights,self.flights[1:]):
            contact=max(0.,following['start_s']-previous['end_s']-self.dt)
            # Match the elapsed interval through confirmed next takeoff;
            # quantize confirmation to the measured physics step.
            confirmation=math.ceil(self.confirm/self.dt-1e-9)*self.dt
            elapsed=contact+confirmation
            gaps.append(dict(contact_duration_s=contact,confirmed_rebound_s=elapsed,
                             passed=elapsed<=self.budget+1e-7))
        passed=bool(coverage and not self.body_contact and not unfinished and
                    len(self.flights)==self.cycles and all(f['motion_pass'] for f in self.flights) and
                    all(g['passed'] for g in gaps))
        return dict(motion_pass=passed,scope='independent per-physics-step flight/connection audit only',
                    coverage_complete=coverage,samples=self.samples,expected_samples=expected_samples,
                    expected_cycles=self.cycles,observed_flights=len(self.flights),
                    flights=[dict(f) for f in self.flights],connections=gaps,
                    max_rebound_seconds=self.budget,unfinished_flight=unfinished,
                    unexpected_body_contact=self.body_contact,hardware_qualified=False)
