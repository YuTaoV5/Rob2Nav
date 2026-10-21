import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from skills.fixstand import POSE,stance_score,upright_phase,initialize_expanded_policy
from skills.fixstand_reference import padded_motion
from rob2nav.fixstand_validation import assess


class FixstandTests(unittest.TestCase):
    def test_phase_input_transfer_preserves_old_policy_at_initialization(self):
        original=torch.nn.Module();original.actor=torch.nn.Linear(48,12);original.critic=torch.nn.Linear(48,1)
        expanded=torch.nn.Module();expanded.actor=torch.nn.Linear(52,12);expanded.critic=torch.nn.Linear(52,1)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'model.pt';torch.save({'model_state_dict':original.state_dict()},checkpoint)
            runner=SimpleNamespace(device='cpu',alg=SimpleNamespace(policy=expanded))
            result=initialize_expanded_policy(runner,checkpoint)
        obs=torch.randn(4,48);phase=torch.randn(4,4)
        torch.testing.assert_close(expanded.actor(torch.cat([obs,phase],dim=1)),original.actor(obs))
        self.assertEqual(len(result['expanded_parameters']),2)

    def test_reference_has_exact_standard_endpoints_and_preserves_action(self):
        raw=np.zeros((40,19));raw[:,2]=.4;raw[:,6]=1.;raw[:,7:]=.2
        result=padded_motion(raw)
        pose=np.array(list(POSE.values()))
        np.testing.assert_allclose(result[:50,7:],np.tile(pose,(50,1)))
        np.testing.assert_allclose(result[-50:,7:],np.tile(pose,(50,1)))
        np.testing.assert_allclose(result[90:130],raw)
        np.testing.assert_allclose(np.linalg.norm(result[:,3:7],axis=1),1.)

    def test_upside_down_pose_cannot_receive_standing_score(self):
        names=list(POSE)
        data=SimpleNamespace(joint_pos=torch.tensor([list(POSE.values())]),joint_vel=torch.zeros(1,12),
            projected_gravity_b=torch.tensor([[0.,0.,1.]]),root_lin_vel_w=torch.zeros(1,3),root_ang_vel_b=torch.zeros(1,3))
        robot=SimpleNamespace(data=data,joint_names=names,device='cpu')
        self.assertLess(float(stance_score(robot)[0]),1e-6)
        data.projected_gravity_b[0,2]=-1.
        self.assertAlmostEqual(float(stance_score(robot)[0]),1.)

    def test_phase_is_observable_for_start_action_and_return(self):
        env=SimpleNamespace(episode_length_buf=torch.tensor([0,100,300]),step_dt=.02)
        result=upright_phase(env)
        torch.testing.assert_close(result[:,:3],torch.eye(3))

    def test_upright_endpoint_schedule(self):
        from skills.fixstand import upright_endpoint_weight
        values=upright_endpoint_weight(torch.tensor([0.,.8,1.,1.25,1.5,4.9,5.,5.75,6.5,8.]))
        torch.testing.assert_close(values,torch.tensor([1.,1.,1.,.5,0.,0.,0.,.5,1.,1.]))

    def test_walking_rejects_sliding_and_accepts_loaded_cycles(self):
        from rob2nav.fixstand_validation import walking_action
        trace=[dict(time=1.+i*.02,upright_walk_error=.1,position=[i*.003,0.,.5],
                    rear_ground=True,front_ground=False,foot_forces_N={'RR':30.,'RL':30.}) for i in range(151)]
        self.assertFalse(walking_action(trace)['passed'])
        for i,row in enumerate(trace):
            row['foot_forces_N']['RR']=0. if 5<=i%40<12 else 30.
            row['foot_forces_N']['RL']=0. if 25<=i%40<32 else 30.
        self.assertTrue(walking_action(trace)['passed'])

    def test_endpoint_validation_rejects_terminal_motion(self):
        trace=[dict(time=i*.02,standard_pose_error=.02,max_joint_speed=.1,upright=1.,speed=.01,
                    angular_speed=.01,all_feet_ground=True,position=[0.,0.,.32]) for i in range(401)]
        self.assertTrue(assess(trace)['passed'])
        for row in trace[-50:]:row['max_joint_speed']=3.
        self.assertFalse(assess(trace)['passed'])


if __name__=='__main__':unittest.main()
