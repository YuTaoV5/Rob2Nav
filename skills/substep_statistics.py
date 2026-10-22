"""Training diagnostics accumulated after physics updates, never state overrides."""
import torch


class JointStatistics:
    def __init__(self,env,lower,upper,velocity):
        if hasattr(env,'_physical_joint_statistics'):raise ValueError('Physical statistics already installed')
        self.env=env;self.robot=env.scene['robot'];self.sensor=env.scene['contact_forces']
        names=self.robot.joint_names;device=self.robot.data.joint_pos.device
        self.lower=torch.tensor([lower[n] for n in names],device=device)
        self.upper=torch.tensor([upper[n] for n in names],device=device)
        self.velocity=torch.tensor([velocity[n] for n in names],device=device)
        self.maximum_velocity=torch.zeros_like(self.robot.data.joint_vel)
        self.ever_velocity=torch.zeros_like(self.maximum_velocity)
        self.maximum_excess=torch.zeros_like(self.maximum_velocity)
        self.maximum_angular_speed=torch.zeros(self.maximum_velocity.shape[0],device=device)
        self.maximum_contacts=torch.zeros_like(self.maximum_angular_speed)
        self.contact_ids=[i for i,n in enumerate(self.sensor.body_names) if 'foot' not in n.lower() and 'calf' not in n.lower()]
        self.samples=0;self.total_samples=0;self.control_steps=0;self.collecting=False
        self.original_process=env.action_manager.process_action;self.original_update=env.scene.update
        env.action_manager.process_action=self.process;env.scene.update=self.update
        env._physical_joint_statistics=self

    def process(self,actions):
        self.maximum_velocity.zero_();self.maximum_excess.zero_();self.maximum_angular_speed.zero_();self.maximum_contacts.zero_()
        self.samples=0;self.control_steps+=1;self.collecting=True
        return self.original_process(actions)

    def update(self,dt):
        result=self.original_update(dt)
        if self.collecting:
            d=self.robot.data
            self.maximum_velocity.copy_(torch.maximum(self.maximum_velocity,d.joint_vel.abs()))
            self.ever_velocity.copy_(torch.maximum(self.ever_velocity,self.maximum_velocity))
            excess=torch.maximum(torch.relu(self.lower-d.joint_pos),torch.relu(d.joint_pos-self.upper))
            self.maximum_excess.copy_(torch.maximum(self.maximum_excess,excess))
            self.maximum_angular_speed.copy_(torch.maximum(self.maximum_angular_speed,torch.linalg.vector_norm(d.root_ang_vel_b,dim=-1)))
            count=(torch.linalg.vector_norm(self.sensor.data.net_forces_w[:,self.contact_ids],dim=-1)>4.).float().sum(-1)
            self.maximum_contacts.copy_(torch.maximum(self.maximum_contacts,count))
            self.samples+=1;self.total_samples+=1
        return result

    def report(self):
        return dict(control_steps=self.control_steps,physical_samples=self.total_samples,
                    last_control_physical_samples=self.samples,maximum_observed_speed_ratio=float((self.ever_velocity/self.velocity).max()),
                    state_writes=False)

    def close(self):
        self.env.action_manager.process_action=self.original_process;self.env.scene.update=self.original_update
        del self.env._physical_joint_statistics


def install(env,source):
    from pathlib import Path
    from xml.etree import ElementTree as ET
    limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(Path(source)/'resources/robots/go2/urdf/go2_description.urdf').getroot().findall('joint') if j.find('limit') is not None}
    return JointStatistics(env,*[{n:float(v[k]) for n,v in limits.items()} for k in ['lower','upper','velocity']])
