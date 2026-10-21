"""Phase-conditioned joint targets; no simulator state or root writes."""
import torch
from isaaclab.envs.mdp.actions import JointPositionAction
from .fixstand import target_pose, upright_endpoint_weight


class UprightEndpointAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions)
        time=self._env.episode_length_buf*self._env.step_dt
        weight=upright_endpoint_weight(time).unsqueeze(-1)
        endpoint=target_pose(self._env.scene['robot'])+.20*torch.tanh(self._raw_actions)
        self._processed_actions[:]=torch.lerp(self._processed_actions,endpoint,weight)
