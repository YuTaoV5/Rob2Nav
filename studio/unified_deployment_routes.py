"""Read-only deployment metadata and plan generation. No SDK import or motor route."""
import json
from pathlib import Path
from fastapi import HTTPException
from pydantic import BaseModel,ConfigDict,Field


class SwitchPlan(BaseModel):
    model_config=ConfigDict(extra='forbid')
    source:str=Field(min_length=1,max_length=64)
    target:str=Field(min_length=1,max_length=64)


def mount_unified_deployment(app,root):
    root=Path(root)
    def profiles():
        return {p.stem:json.loads(p.read_text()) for p in (root/'deployment/skills').glob('*.json')}
    def latest(prefix):
        files=list((root/'outputs').glob(prefix+'*/summary.json'))
        if not files:return None
        path=max(files,key=lambda p:p.stat().st_mtime)
        try:
            d=json.loads(path.read_text());return dict(path=path.relative_to(root).as_posix(),passed=d.get('passed'),reason=d.get('reason'),max_torque_step_Nm=d.get('max_torque_step_Nm'),max_tilt_deg=d.get('max_tilt_deg'),commits=d.get('commits',[]))
        except (OSError,ValueError):return None
    @app.get('/api/deployment/unified')
    def metadata():
        return dict(profiles=list(profiles().values()),states=['CAPTURE','NEUTRAL','READY','ENTRY','WARMUP','ARMED','BLEND','ACTIVE','EXIT','FAULT'],
            hardware_connected=False,motor_control_from_web=False,sdk='814556d15970dd2ecf1c9984e845ca02ab07e206',
            validation=dict(switch=latest('unified_switch_'),atomic=latest('unified_atomic_')),
            document='#reports?path=docs%2FUNIFIED_GO2_DEPLOYMENT.md')
    @app.post('/api/deployment/unified/plan')
    def plan(a:SwitchPlan):
        data=profiles()
        if a.source not in data or a.target not in data:raise HTTPException(404,'Unknown deployment profile')
        old,new=data[a.source],data[a.target]
        common=sorted(old['joints'])
        def mapped(s,key):return [s[key][s['joints'].index(n)] for n in common]
        route=['异步准备候选；失败时保留当前策略',
            '等待当前原子动作结束并落地站稳' if old.get('atomic') else '速度渐归零，等待有效低速支撑窗口',
            '实测力矩/姿态接管 → 中立站姿','入口姿态与PD渐变','新历史预热',
            '提交为ARMED，显式trigger启动原子动作，拒绝旧响应' if new.get('atomic') else '输出渐入 → 提交新epoch，拒绝旧响应']
        return dict(source=a.source,target=a.target,route=route,
            max_default_pose_difference_rad=max(abs(x-y) for x,y in zip(mapped(old,'default'),mapped(new,'default'))),
            max_entry_pose_difference_rad=max(abs(x-y) for x,y in zip(mapped({**old,'entry':old.get('entry_pose') or old['default']},'entry'),mapped({**new,'entry':new.get('entry_pose') or new['default']},'entry'))),
            max_kp_difference=max(abs(x-y) for x,y in zip(mapped(old,'kp'),mapped(new,'kp'))),
            target_simulation_enabled=new.get('simulation_enabled',True),target_hardware_qualified=new.get('hardware_qualified',False),
            blocker=new.get('blocker'),executed=False,hardware_connected=False)
