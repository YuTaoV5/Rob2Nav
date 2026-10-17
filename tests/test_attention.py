import unittest
import torch
from perceptive_rl.attention import AttentionActor, actor_from_state


class AttentionTests(unittest.TestCase):
    def test_missing_map_and_export(self):
        actor = AttentionActor()
        obs = torch.randn(3, 419)
        obs[:, 232:] = 0
        weights = actor.attention(obs)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(float(weights[:, :, :-1].sum()), 0.)
        self.assertTrue(torch.equal(weights[:, :, -1], torch.ones(3, 4)))
        self.assertTrue(torch.equal(actor(obs), actor.base(obs)))
        exported = torch.jit.script(actor)
        torch.testing.assert_close(actor(obs), exported(obs))

    def test_grid_order_and_checkpoint(self):
        actor = AttentionActor()
        torch.testing.assert_close(actor.xy[1], torch.tensor([-.45, -.5]))
        torch.testing.assert_close(actor.xy[17], torch.tensor([-.55, -.4]))
        obs = torch.randn(2, 419); obs[:, 232:] = 1
        # Nonzero head verifies export beyond the zero-initialized branch.
        torch.nn.init.normal_(actor.head[-1].weight, std=.01)
        restored, dim = actor_from_state({'actor.'+k: v for k, v in actor.state_dict().items()})
        self.assertEqual(dim, 419)
        torch.testing.assert_close(actor(obs), restored(obs))
        torch.testing.assert_close(actor(obs), torch.jit.script(actor)(obs))
        actor(obs).square().mean().backward()
        self.assertGreater(float(actor.cnn[0].weight.grad.abs().sum()), 0)
