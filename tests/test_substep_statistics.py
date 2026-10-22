import unittest
from types import SimpleNamespace
import torch
from skills.substep_statistics import JointStatistics
from skills.fixstand import urdf_speed_penalty


class FakeScene(dict):
    def update(self,dt):pass


class SubstepTests(unittest.TestCase):
    def env(self):
        d=SimpleNamespace(joint_pos=torch.zeros(2,1),joint_vel=torch.zeros(2,1),root_ang_vel_b=torch.zeros(2,3))
        sensor=SimpleNamespace(body_names=['base','RR_foot'],data=SimpleNamespace(net_forces_w=torch.zeros(2,2,3)))
        scene=FakeScene(robot=SimpleNamespace(joint_names=['k'],data=d,device='cpu'),contact_forces=sensor)
        return SimpleNamespace(scene=scene,device='cpu',action_manager=SimpleNamespace(process_action=lambda a:None))

    def test_between_control_frame_peak_reaches_reward(self):
        env=self.env();stats=JointStatistics(env,{'k':-1.},{'k':1.},{'k':10.})
        env.action_manager.process_action(torch.zeros(2,1))
        env.scene['robot'].data.joint_vel[0,0]=20.;env.scene['robot'].data.joint_pos[0,0]=1.2
        env.scene.update(.002)
        env.scene['robot'].data.joint_vel.zero_();env.scene['robot'].data.joint_pos.zero_();env.scene.update(.002)
        self.assertAlmostEqual(float(urdf_speed_penalty(env,{'k':10.})[0]),1.21,places=5)
        self.assertAlmostEqual(float(stats.maximum_excess[0,0]),.2,places=5)
        self.assertEqual(stats.samples,2)
        env.action_manager.process_action(torch.zeros(2,1));self.assertEqual(float(stats.maximum_velocity.max()),0.)
        self.assertEqual(stats.report()['maximum_observed_speed_ratio'],2.)
        stats.close();self.assertFalse(hasattr(env,'_physical_joint_statistics'))

    def test_transient_contact_and_angular_speed_are_retained(self):
        env=self.env();stats=JointStatistics(env,{'k':-1.},{'k':1.},{'k':10.})
        env.action_manager.process_action(torch.zeros(2,1))
        env.scene['contact_forces'].data.net_forces_w[0,0,2]=10.
        env.scene['robot'].data.root_ang_vel_b[0,0]=3.;env.scene.update(.002)
        env.scene['contact_forces'].data.net_forces_w.zero_();env.scene['robot'].data.root_ang_vel_b.zero_();env.scene.update(.002)
        self.assertEqual(float(stats.maximum_contacts[0]),1.)
        self.assertEqual(float(stats.maximum_angular_speed[0]),3.)
        stats.close()


if __name__=='__main__':unittest.main()
