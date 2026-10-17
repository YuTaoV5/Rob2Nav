import unittest
import torch
from moe_perceptive.memory import ElevationMemory


class VisibleMemoryTests(unittest.TestCase):
    def test_unknown_reprojection_expiry_and_reset(self):
        memory=ElevationMemory(2,'cpu');position=torch.tensor([[0.,0.,.5],[0.,0.,.5]])
        returns=torch.tensor([[[0.,0.,0.]],[[0.,0.,0.]]]);yaw=torch.zeros(2)
        result=memory.update(returns,position,yaw,0.,torch.ones(2,dtype=torch.bool))
        self.assertEqual(result[:,693:1386].sum().item(),2.)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result[0,693+341].item(),1.)
        memory.reset(torch.tensor([0]));position[:,0]=.05
        result=memory.update(torch.full_like(returns,torch.inf),position,yaw,.2,torch.zeros(2,dtype=torch.bool))
        self.assertEqual(result[0,693:1386].sum().item(),0.)
        self.assertEqual(result[1,693+340].item(),1.)
        result=memory.update(returns,position,yaw,1.1,torch.zeros(2,dtype=torch.bool))
        self.assertEqual(result[:,693:1386].sum().item(),0.)

    def test_reobserved_surface_refreshes_age(self):
        memory=ElevationMemory(1,'cpu',lifetime=.3);position=torch.tensor([[0.,0.,.5]])
        returns=torch.tensor([[[0.,0.,0.]]]);yaw=torch.zeros(1)
        memory.update(returns,position,yaw,0.,torch.tensor([True]))
        memory.update(returns,position,yaw,.25,torch.tensor([True]))
        result=memory.update(returns,position,yaw,.4,torch.tensor([False]))
        self.assertEqual(result[0,693+341].item(),1.)
        self.assertAlmostEqual(result[0,1386+341].item(),.5,places=5)


if __name__=='__main__':unittest.main()
