import copy
import unittest
import numpy as np
from rob2nav.fixstand_validation import stationary_upright


class StationaryUprightTests(unittest.TestCase):
    def trace(self):
        return [dict(time=float(t),position=[0.,0.,.5],upright_walk_error=.1,speed=0.,angular_speed=0.,
                     max_joint_speed=0.,target=[0.]*12,foot_forces_N=dict(RR=30.,RL=30.,FR=0.,FL=0.))
                for t in np.arange(0.,8.,.02)]

    def test_static_supported_hold_needs_no_walking_cycles(self):
        self.assertTrue(stationary_upright(self.trace())['passed'])

    def test_sliding_cannot_count_as_standing(self):
        trace=self.trace()
        for x in trace:x['position'][0]=x['time']*.1
        self.assertFalse(stationary_upright(trace)['passed'])

    def test_front_support_or_missing_rear_support_rejected(self):
        for leg,value in [('FR',10.),('RR',0.)]:
            trace=self.trace()
            for x in trace:x['foot_forces_N'][leg]=value
            self.assertFalse(stationary_upright(trace)['passed'])

    def test_jitter_and_short_hold_rejected(self):
        trace=self.trace()
        for i,x in enumerate(trace):x['target'][0]=.1*(i%2)
        self.assertFalse(stationary_upright(trace)['passed'])
        self.assertFalse(stationary_upright(self.trace()[:150])['passed'])

    def test_single_target_snap_cannot_hide_inside_stable_fraction(self):
        trace=self.trace();trace[106]['target'][0]=.2
        self.assertFalse(stationary_upright(trace)['passed'])


if __name__=='__main__':unittest.main()
