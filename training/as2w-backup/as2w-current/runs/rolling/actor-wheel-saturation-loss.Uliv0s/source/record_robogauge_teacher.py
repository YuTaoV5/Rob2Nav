"""Record teacher data without replacing RoboGauge's simulation or scoring."""
import os
from pathlib import Path
import runpy
import sys

import numpy as np

from as2w_go2_retarget import endpoint
from robogauge.tasks.robots.go2.go2_moe import Go2MoE

output = Path(os.environ["AS2W_GO2_TEACHER_TRACE"])
fields = ("time_s","observation","joint_q","joint_dq","joint_target","foot_hip_m",
          "contact","base_position_m","base_quaternion")
rows = {name:[] for name in fields}
build_observation = Go2MoE.build_observation
get_action = Go2MoE.get_action


def capture_observation(self, sim_data, goal_data):
    self._as2w_teacher_sim_data = sim_data
    return build_observation(self,sim_data,goal_data)


def capture_action(self, obs):
    result = get_action(self,obs)
    sim_data = self._as2w_teacher_sim_data
    state = sim_data.proprio
    q = state.joint.pos.copy()
    contacts = sim_data.dynamics.contacts
    touching = {name.rsplit("/",1)[-1] for name,force in
                zip(contacts.robot_geom_names,contacts.normal_forces) if force > 2.}
    values = (sim_data.sim_time,np.array(obs,copy=True),q,state.joint.vel.copy(),
              np.array(result[0],copy=True),endpoint(q.reshape(4,3),.0955,.213),
              np.array([leg in touching for leg in ("FL","FR","RL","RR")]),
              state.base.pos.copy(),state.base.quat.copy())
    for name,value in zip(fields,values):
        rows[name].append(value)
    return result


descending_xml = os.environ.get("AS2W_GO2_DESCENDING_XML")
if descending_xml:
    from robogauge.tasks.pipeline.base_pipeline import BasePipeline

    initialize_pipeline = BasePipeline.__init__

    def descending_pipeline(self, run_name, simulator_cfg, robot_cfg, gauge_cfg, args):
        gauge_cfg.assets.terrain_xmls = [descending_xml,*gauge_cfg.assets.terrain_xmls[1:]]
        gauge_cfg.assets.terrain_spawn_pos = [1.15,0.,.80]
        gauge_cfg.goals.target_pos_velocity.target_pos = [4.50,0.,.36]
        initialize_pipeline(self,run_name,simulator_cfg,robot_cfg,gauge_cfg,args)

    BasePipeline.__init__ = descending_pipeline

Go2MoE.build_observation = capture_observation
Go2MoE.get_action = capture_action
script = sys.argv.pop(1)
sys.argv[0] = script
runpy.run_path(script,run_name="__main__")
if not rows["time_s"]:
    raise RuntimeError("RoboGauge produced no teacher steps")
np.savez_compressed(output,**{name:np.asarray(values) for name,values in rows.items()})
