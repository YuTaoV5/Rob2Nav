from copy import deepcopy
from dataclasses import asdict
import unittest
from rob2nav.landing_validation import audit_landing_cycles
from skills.backflip.rebound_landing import LandingConfig


def trajectory():
    rows=[]
    def add(n,contact,pitch,u):
        for _ in range(n):rows.append(dict(time=(len(rows)+1)*.01,foot_contact=contact,
            body_contact=False,omega_body=[0.,pitch,0.],upright=u))
    add(100,True,0,1)
    for _ in range(3):
        add(10,True,-15,1);add(20,False,-17.5,-1)
        add(5,True,-10,.5);add(1,True,-10,.95)
    add(100,True,0,1)
    return rows


class IndependentLandingTests(unittest.TestCase):
    def check(self,rows):return audit_landing_cycles(rows,.01,asdict(LandingConfig()),len(rows))
    def test_valid_three_cycles_without_execution_counters(self):
        r=self.check(trajectory());self.assertTrue(r['motion_pass'])
        self.assertEqual(len(r['flights']),3)
        self.assertAlmostEqual(r['connections'][0]['seconds'],.18)

    def test_phase_counter_cannot_forge_success(self):
        rows=trajectory()
        for r in rows:r.update(phase=4,execution_completed=999,omega_body=[0.,0.,0.])
        self.assertFalse(self.check(rows)['motion_pass'])

    def test_reverse_and_no_inversion(self):
        for kind in ('reverse','upright'):
            rows=trajectory()
            for r in rows:
                if kind=='reverse':r['omega_body'][1]*=-1
                else:r['upright']=1.
            self.assertFalse(self.check(rows)['motion_pass'])

    def test_body_contact_missing_sample_and_extra_hop(self):
        rows=trajectory();rows[145]['body_contact']=True
        self.assertFalse(self.check(rows)['motion_pass'])
        rows=trajectory();del rows[5]
        self.assertFalse(self.check(rows)['motion_pass'])
        rows=trajectory()
        for r in rows[-30:-25]:r['foot_contact']=False
        self.assertFalse(self.check(rows)['motion_pass'])

    def test_rebound_wait_cannot_hide_in_absorption(self):
        rows=trajectory()
        insert=deepcopy(rows[135]);insert['omega_body']=[0.,0.,0.]
        rows[136:136]=[deepcopy(insert) for _ in range(50)]
        for i,r in enumerate(rows):r['time']=(i+1)*.01
        self.assertFalse(self.check(rows)['motion_pass'])

    def test_ground_rotation_plus_small_air_rotation_rejected(self):
        rows=trajectory()
        for r in rows:
            if not r['foot_contact']:r['omega_body'][1]=-.1
        self.assertFalse(self.check(rows)['motion_pass'])


if __name__=='__main__':unittest.main()
