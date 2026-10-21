"""Imported only after the Isaac application and pinned source have been initialized."""
from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand


class StartAtStand(MotionCommand):
    def _adaptive_sampling(self,env_ids):
        self.time_steps[env_ids]=0
