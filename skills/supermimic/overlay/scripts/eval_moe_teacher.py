"""Use the real MoE teacher in the same strict physics/gait evaluator.

The --checkpoint argument supplies only the compatible student environment
contract; its actor is not executed. Output explicitly records this distinction.
"""
import argparse,json,runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import mjlab.envs as envs
from go2_supermimic.supermimic import train
from rob2nav_moe_teacher import RoboGaugeTeacher
parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--teacher-profile',required=True)
extra,remaining=parser.parse_known_args()
environment=None
OriginalEnv=envs.ManagerBasedRlEnv
class TeacherEnv(OriginalEnv):
 def __init__(self,*a,**kw):
  global environment
  super().__init__(*a,**kw);self.teacher=RoboGaugeTeacher(self,extra.teacher_profile);environment=self
 def step(self,action):
  result=super().step(action);self.teacher.reset(result[2]|result[3]);return result
class TeacherActor(torch.nn.Module):
 def __init__(self,*a,**kw):super().__init__()
 def load_state_dict(self,*a,**kw):return None
 def forward(self,obs):return environment.teacher.act(environment.command_manager.get_term('target').gen.chase_vel)
envs.ManagerBasedRlEnv=TeacherEnv;train.Actor=TeacherActor
script=Path(__file__).with_name('eval_gait.py' if '--gait-output' in remaining else 'eval_velocity.py')
sys.argv=[str(script)]+remaining
runpy.run_path(str(script),run_name='__main__')
output=Path(remaining[remaining.index('--output')+1]);report=json.loads(output.read_text())
report.update(policy_kind='rob2nav_moe_teacher',student_actor_used=False,teacher=environment.teacher.metadata,
              teacher_mapping_error=environment.teacher.mapping_error)
output.write_text(json.dumps(report,indent=2));print('TEACHER_EVALUATION',json.dumps(report))
