"""Imported only after the Isaac application and pinned source have been initialized."""
from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand
from isaaclab.envs.mdp.actions import JointPositionAction
import torch


class StartAtStand(MotionCommand):
    def _adaptive_sampling(self,env_ids):
        self.time_steps[env_ids]=0


class FixstandReferenceAction(JointPositionAction):
    """Bounded learned residual in padded stance/bridge regions; original actor in the action."""
    def process_actions(self,actions):
        super().process_actions(actions)
        command=self._env.command_manager.get_term('motion')
        frames=command.motion.time_step_total
        boundary=(command.time_steps<90)|(command.time_steps>=frames-90)
        from skills.fixstand import sideflip_boundary_limit
        hold=getattr(self.cfg,'residual_limit_rad',.05)
        bridge=getattr(self.cfg,'bridge_residual_limit_rad',hold)
        limit=sideflip_boundary_limit(command.time_steps,frames,hold,bridge).unsqueeze(-1)
        target=command.joint_pos+limit*torch.tanh(self._raw_actions)
        self._processed_actions[boundary]=target[boundary]
