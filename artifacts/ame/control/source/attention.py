"""Independent AME-inspired Go2 encoder, not a reproduction or imported G1 code.

Existing 419-D sensor contract: proprio(45), height(187), valid(187).
Keep every 10-cm token to avoid erasing small support surfaces. Unknown cells
are supplied explicitly; a learned null token keeps all-missing maps finite.
"""
import torch
from torch import nn


def mlp_actor(dim):
    return nn.Sequential(nn.Linear(dim, 512), nn.ELU(), nn.Linear(512, 256),
                         nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 12))


class AttentionActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = mlp_actor(419)
        self.cnn = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.ELU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.ELU())
        x, y = torch.meshgrid(torch.linspace(-.55, 1.05, 17),
                              torch.linspace(-.5, .5, 11), indexing="ij")
        self.register_buffer("xy", torch.stack((x.flatten(), y.flatten()), -1))
        self.key = nn.Linear(34, 64)
        self.value = nn.Linear(34, 64)
        self.query = nn.Sequential(nn.Linear(45, 64), nn.ELU())
        self.null = nn.Parameter(torch.zeros(1, 1, 34))
        self.head = nn.Sequential(nn.Linear(109, 128), nn.ELU(), nn.Linear(128, 12))
        # Exact initial equality to the published student; PPO updates all branches.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def tokens(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.cnn(torch.stack((obs[:, 45:232] * obs[:, 232:419], obs[:, 232:419]), 1)
                            .reshape(-1, 2, 17, 11)).flatten(2).transpose(1, 2)
        coords = self.xy.unsqueeze(0).expand(obs.shape[0], -1, -1)
        tokens = torch.cat((features, coords), -1)
        return torch.cat((tokens, self.null.expand(obs.shape[0], -1, -1)), 1)

    def weights(self, obs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        # Four independent 16-D dot-product attention heads; no dropout/BatchNorm.
        keys = self.key(tokens).reshape(-1, 188, 4, 16).transpose(1, 2)
        query = self.query(obs[:, :45]).reshape(-1, 4, 1, 16)
        logits = (query * keys).sum(-1) / 4.
        known = torch.cat((obs[:, 232:419] > .5,
                           torch.ones_like(obs[:, :1], dtype=torch.bool)), 1)
        return logits.masked_fill(~known.unsqueeze(1), -1e4).softmax(-1)

    @torch.jit.export
    def attention(self, obs: torch.Tensor) -> torch.Tensor:
        return self.weights(obs, self.tokens(obs))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tokens = self.tokens(obs)
        values = self.value(tokens).reshape(-1, 188, 4, 16).transpose(1, 2)
        context = (self.weights(obs, tokens).unsqueeze(-1) * values).sum(2).reshape(-1, 64)
        return self.base(obs) + self.head(torch.cat((obs[:, :45], context), -1))


def actor_from_state(state):
    if "actor.base.0.weight" in state:
        actor, dim = AttentionActor(), 419
    else:
        dim = state["actor.0.weight"].shape[1]
        actor = mlp_actor(dim)
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in state.items()
                           if k.startswith("actor.")})
    return actor, dim


def register_policy():
    """Register in this process only; never modify the installed RSL-RL package."""
    from rsl_rl.modules import ActorCritic
    import rsl_rl.runners.on_policy_runner as module

    class AttentionActorCritic(ActorCritic):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.actor[0].in_features != 419 or self.state_dependent_std:
                raise ValueError("Attention requires --fused and a scalar action std")
            self.actor = AttentionActor()

    module.AttentionActorCritic = AttentionActorCritic
