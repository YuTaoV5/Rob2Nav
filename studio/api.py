"""Rob2Nav Studio API. Loopback by default; mutation requires same-origin header."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
from .visuals import ROBOT_DIR,go2_urdf,terrain_data,terrain_svg
from .catalog import Catalog,read_json,clean,normalize_evaluation
from .config import RunConfig,REWARDS,command,simulation_command
from .jobs import Jobs
from .telemetry import telemetry, checkpoints, replay
from .registry import registry,capabilities


def create_app(root=None,allow_training=False,sim_python=None,mujoco_python=None):
    root=Path(root or Path(__file__).resolve().parents[1]).resolve()
    sim_python=Path(sim_python or '/root/isaacsim/python.sh')
    mujoco_python=Path(mujoco_python or '/root/gpufree-data/envs/go2-mapless/bin/python')
    catalog=Catalog(root);jobs=Jobs(root)
    app=FastAPI(title='Rob2Nav Studio',version='0.1.0')
    from .paw_routes import mount_handshake
    mount_handshake(app, root)
    from .navigation_live_routes import mount_navigation_live
    mount_navigation_live(app, root)
    from .live_skills import mount_live_skills
    mount_live_skills(app,root,mujoco_python)
    from .unified_deployment_routes import mount_unified_deployment
    mount_unified_deployment(app,root)

    @app.get('/api/deployment')
    def deployment_framework():
        from rob2nav.contracts import profiles
        from rob2nav import __version__
        return {'version':__version__,'profiles':profiles(),'states':['standby','armed','active','fault','stopped'],
            'backends':[{'id':'mujoco','status':'source-policy inference connected'},
                        {'id':'isaac','status':'MoE training/checkpoint inference connected'},
                        {'id':'ros','status':'adapter pending'}, {'id':'unitree_dds','status':'gated SDK2 transport implemented; offline IDL/CRC only; no hardware connection'}],
            'layers':['robot/policy contracts','asset integrity','lifecycle and command gate','inference backends','navigation adapters','recording / evidence','Studio API / UI'],
            'hardware_enabled':False}

    @app.get('/api/navigation')
    def navigation_sources():
        from rob2nav.navigation import integrations,SCENES
        return {'integrations':integrations(),'scenes':list(SCENES),'native_connected':False}

    from pydantic import BaseModel,Field
    class NavigationProbe(BaseModel):
        scene:str='clutter'
        length:float=Field(default=.6,ge=.3,le=1.2)
        width:float=Field(default=.3,ge=.15,le=.6)
        margin:float=Field(default=.05,ge=0,le=.3)
        body_z:float=Field(default=.32,ge=.2,le=.8)
        alpha:float=Field(default=2,ge=.1,le=10)

    @app.post('/api/navigation/probe')
    def navigation_probe(config:NavigationProbe):
        from rob2nav.navigation import plan,shield
        try:
            result=plan(**config.model_dump(exclude={'alpha'}))
            result['shield']=dict(raw=[1.,.2],position=[2.7,3.3],obstacles=[[3.4,3.3,.25]],
                **shield([1.,.2],[2.7,3.3],[[3.4,3.3,.25]],alpha=config.alpha))
            return result
        except ValueError as error:raise HTTPException(422,str(error))

    @app.get('/api/robot/go2.urdf')
    def robot_definition():
        return Response(go2_urdf(root),media_type='application/xml')

    @app.get('/api/robot/assets/{name}')
    def robot_asset(name:str):
        assets=(root/ROBOT_DIR/'assets').resolve();p=(assets/name).resolve()
        if p.parent!=assets or p.suffix.lower()!='.obj' or not p.is_file():raise HTTPException(404,'Robot asset not found')
        return FileResponse(p,media_type='text/plain')

    @app.get('/api/terrain-preview')
    def terrain_preview(kind:str='ame_stakes',course:str='wide_deep',difficulty:float=0.,image:bool=False):
        try:data=terrain_data(kind,course,difficulty)
        except ValueError:raise HTTPException(422,'Invalid terrain preview parameters')
        return Response(terrain_svg(data),media_type='image/svg+xml') if image else data

    @app.middleware('http')
    async def protection(request,call_next):
        if request.method not in ('GET','HEAD','OPTIONS'):
            origin=request.headers.get('origin')
            if request.headers.get('x-studio-request')!='1' or (origin and urlparse(origin).netloc!=request.headers.get('host')):
                from fastapi.responses import JSONResponse
                return JSONResponse({'detail':'Same-origin Studio request required'},status_code=403)
        response=await call_next(request)
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='same-origin'
        return response

    @app.get('/api/health')
    def health():return {'ok':True,'training_enabled':allow_training,'simulator_available':sim_python.is_file(),'mujoco_available':mujoco_python.is_file(),'version':'0.1.0'}

    @app.get('/api/catalog')
    def get_catalog(refresh:bool=False):return catalog.get(refresh)

    @app.get('/api/capabilities')
    def get_capabilities():return capabilities(catalog.get(),jobs.all())

    from rob2nav.experiment_schema import METHODS, reward_schema

    @app.post('/api/experiment-preview')
    def experiment_preview(config:RunConfig):return {'rewards':reward_schema(config),'overrides':config.overrides()}

    @app.get('/api/schema')
    def schema():return {'config':RunConfig.model_json_schema(),'defaults':RunConfig().model_dump(),'rewards':REWARDS,'methods':METHODS,'registry':registry(),
        'adapters':[
            {'id':'moe','name':'RoboGauge MoE + 感知','training':True,'evaluation':True,'note':'已连接现有训练入口；执行需要Linux GPU worker。'},
            {'id':'ame','name':'AME / Go2复现','training':False,'evaluation':False,'note':'历史结果已导入；原训练控制适配器尚未接通。'},
            {'id':'nju','name':'NJU agility','training':False,'evaluation':False,'note':'现有技能视频可回放；原训练流程未接通。'},
            {'id':'robogauge','name':'RoboGauge benchmark','training':False,'evaluation':False,'note':'历史评估可查看；独立任务提交适配器待接入。'}]}

    @app.get('/api/runs/{key}/series')
    def series(key:str):
        try:return catalog.series(key)
        except (KeyError,ValueError):raise HTTPException(404,'Training series not found')

    @app.get('/api/evaluations/{key}')
    def evaluation(key:str):
        try:
            p=catalog.path(key)
            if p.name!='summary.json':raise KeyError(key)
            trace=read_json(p.parent/'trace.json') or read_json(p.parent/'trajectory_0.json',[])
            trace=[dict(t,time=t.get('time',t.get('seconds',0))) for t in trace] if isinstance(trace,list) else []
            diagnostic=read_json(p.parent/'map_diagnostic.json',[])
            return clean({'summary':normalize_evaluation(read_json(p)),'raw_summary':read_json(p),'trace':trace,'diagnostic':diagnostic})
        except KeyError:raise HTTPException(404,'Evaluation not found')

    @app.get('/api/runs/{key}/scalars')
    def scalars(key:str,tag:str|None=None):
        try:return catalog.scalars(key,tag)
        except (KeyError,ValueError):raise HTTPException(404,'Scalar series not found')

    @app.get('/api/reports/{key}')
    def report(key:str):
        try:
            p=catalog.path(key)
            if p.suffix!='.md':raise KeyError(key)
            return {'text':p.read_text(encoding='utf-8-sig',errors='replace'),'path':catalog.relative(p)}
        except KeyError:raise HTTPException(404,'Report not found')

    @app.get('/api/files/{path:path}')
    def file(path:str):
        p=(root/path).resolve()
        if not p.is_relative_to(root) or not p.is_file() or p.suffix.lower() not in {'.mp4','.png','.jpg','.jpeg','.webp','.json','.jsonl','.md','.yaml','.csv','.pt','.svg'}:
            raise HTTPException(404,'Artifact not found')
        # Never expose arbitrary files in .git, credentials or runtime control files.
        if any(s.startswith('.') for s in Path(path).parts) or not (p.is_relative_to(root/'artifacts') or p.is_relative_to(root/'outputs') or p.is_relative_to(root/'skills') or p.is_relative_to(root/'docs/evidence') or p.suffix=='.md'):
            raise HTTPException(404,'Artifact not published')
        if p.is_relative_to(root/'outputs/studio') and p.suffix not in ('.mp4','.png','.jpg','.jsonl','.pt'):
            raise HTTPException(404,'Use job API for runtime metadata')
        return FileResponse(p,filename=p.name if p.suffix=='.pt' else None)

    @app.get('/api/resources')
    def resources():
        d={'time':time.time(),'gpu':[],'compute_processes':[],'training_enabled':allow_training,'simulator_available':sim_python.is_file(),'mujoco_available':mujoco_python.is_file()}
        try:
            import psutil
            disk=psutil.disk_usage(str(root));ram=psutil.virtual_memory()
            d.update(cpu=psutil.cpu_percent(),memory_percent=ram.percent,disk_free_gb=round(disk.free/1024**3,1))
        except (ImportError,OSError,TypeError):d.update(cpu=None,memory_percent=None,disk_free_gb=None)
        try:
            p=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
            if p.returncode==0:
                d['gpu']=[dict(zip(('name','utilization','used_mb','total_mb'),[s.strip() for s in line.split(',')])) for line in p.stdout.splitlines()]
        except (OSError,subprocess.TimeoutExpired):pass
        try:
            p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
            if p.returncode==0:
                for line in p.stdout.splitlines():
                    parts=[s.strip() for s in line.split(',')]
                    if len(parts)==3 and parts[0].isdigit():
                        d['compute_processes'].append(dict(pid=int(parts[0]),name=Path(parts[1]).name,used_mb=parts[2]))
        except (OSError,subprocess.TimeoutExpired):pass
        return d

    def preflight(config):
        errors=[];warnings=[]
        output=root/'outputs/studio/runs/PREVIEW'
        from .upstreams import pie_python
        runtime=pie_python(root) if config.simulation_policy=='pie' else mujoco_python if config.task=='simulation' and config.simulation_policy!='checkpoint' else sim_python
        try:cmd=simulation_command(config,root,output,runtime) if config.task=='simulation' else command(config,root,output,root/'outputs/studio/PREVIEW.json',runtime)
        except ValueError as e:cmd=[];errors.append(str(e))
        if not runtime.is_file():errors.append('当前主机没有所选任务的执行解释器；可保存草稿，在服务器运行。')
        if config.task=='train' and not (root/'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt').is_file():
            errors.append('缺少固定版本MoE基座权重；请先核对项目资源。')
        if os.name!='posix':errors.append('训练worker需要Linux主机；本地可浏览、编辑和保存配置。')
        if not allow_training:warnings.append('当前服务为只读训练模式，提交队列已关闭。')
        if config.task=='simulation':warnings.append('选定权重在原生Isaac中推理，教师/学生身份以检查点来源为准。' if config.simulation_policy=='checkpoint' else '使用历史MoE/IK/NJU技能组合，仿真成绩不计入新感知MoE能力。' if config.simulation_policy=='baseline' else '上游策略本地推理，仅仿真；独立能力验收另行记录。')
        elif config.mode=='teacher':warnings.append('完整地图为特权教师输入，不能标为可部署相机/雷达策略。')
        if config.course=='wide_deep':warnings.append('训练前置几何与原AME严格评估分开，不计为原尺寸成功。')
        if config.body_height_target_m is not None or config.swing_clearance_target_m is not None:
            warnings.append('高度是固定训练奖励目标，不是已有策略的运行指令；需要微调与独立验收。')
        if config.warmup+config.fusion+config.actor>config.iterations:warnings.append('阶段总轮数超过本次预算，后续阶段本次可能不会执行；恢复时按全局轮次计算。')
        if config.initialization=='resume':warnings.append('恢复需匹配原参数；不恢复模拟器物理状态，不保证逐步确定性。')
        return {'valid':not errors,'errors':errors,'warnings':warnings,'command':cmd,'resolved':config.model_dump(),
                'environment_steps':config.num_envs*24*config.iterations,'estimated_hours':None,'rewards':reward_schema(config)}

    @app.post('/api/preflight')
    def validate(config:RunConfig):return preflight(config)

    @app.get('/api/jobs')
    def get_jobs():
        result=jobs.all()
        for job in result:
            job['weights']=checkpoints(root,job)
            folder=root/'outputs/studio/runs'/job['id'];status=folder/'status.json';job['progress']=read_json(status)
            try:job['progress_updated']=status.stat().st_mtime
            except OSError:job['progress_updated']=None
            if job['state']=='evaluating' and job['detail'].get('stage')=='evaluation':
                job['evaluation_completed']=sum(bool(read_json(p)) for p in (jobs.directory/'jobs'/job['id']/'evaluation').glob('*/summary.json'))
        return clean(result)

    @app.get('/api/jobs/{key}/telemetry')
    def job_telemetry(key:str,tags:str=''):
        try:return telemetry(root,jobs,key,tags.split('|') if tags else [])
        except KeyError:raise HTTPException(404,'Unknown job')

    @app.get('/api/jobs/{key}/replay')
    def job_replay(key:str,after:int=0):
        try:jobs.get(key)
        except KeyError:raise HTTPException(404,'Unknown job')
        if after<0:raise HTTPException(422,'Invalid cursor')
        return replay(root/'outputs/studio/runs'/key,after)

    @app.post('/api/jobs')
    def save(config:RunConfig):return jobs.create(config.model_dump(exclude_none=True))

    @app.post('/api/jobs/{key}/submit')
    def submit(key:str):
        if not allow_training:raise HTTPException(403,'Training disabled; start server with --allow-training on the GPU host')
        try:job=jobs.get(key)
        except KeyError:raise HTTPException(404,'Unknown job')
        check=preflight(RunConfig.model_validate(job['config']))
        if not check['valid']:raise HTTPException(422,check)
        if any(j['state']=='unknown' for j in jobs.all()):raise HTTPException(409,'A previous worker job is unknown; reconcile its process before submitting new GPU work')
        if not jobs.transition(key,'draft','queued'):raise HTTPException(409,'Only a draft can be submitted once')
        log=(jobs.directory/'worker.log').open('a')
        try:subprocess.Popen([sys.executable,'-m','studio.worker','--root',str(root),'--python',str(sim_python),'--mujoco-python',str(mujoco_python)],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except OSError as e:
            jobs.update(key,'failed',{'reason':str(e)});raise HTTPException(500,'Worker failed to launch')
        finally:log.close()
        return jobs.get(key)

    @app.post('/api/jobs/{key}/stop')
    def stop(key:str):
        try:job=jobs.get(key)
        except KeyError:raise HTTPException(404,'Unknown job')
        if job['state'] in ('draft','queued'):jobs.update(key,'canceled',{'reason':'User canceled before execution'})
        elif job['state'] in ('running','preflight','evaluating'):
            folder=jobs.folder(key);folder.mkdir(parents=True,exist_ok=True);(folder/'cancel.flag').touch()
        else:raise HTTPException(409,'Job is not active')
        return jobs.get(key)

    @app.get('/api/jobs/{key}/log')
    def job_log(key:str):
        try:folder=jobs.folder(key)
        except KeyError:raise HTTPException(404,'Unknown job')
        result={}
        for stage in ('training','evaluation','simulation'):
            p=folder/(stage+'.log')
            if p.exists():
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-24000));result[stage]=f.read().decode(errors='replace')
        return result

    @app.get('/api/events')
    async def events(request:Request):
        async def generate():
            while not await request.is_disconnected():
                yield 'event: status\ndata: '+json.dumps({'jobs':jobs.all(),'time':time.time()})+'\n\n'
                await asyncio.sleep(3)
        return StreamingResponse(generate(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    dist=Path(__file__).parent/'frontend/dist'
    @app.get('/{path:path}')
    def frontend(path:str):
        if path.startswith('api/'):raise HTTPException(404)
        p=(dist/path).resolve()
        if not p.is_relative_to(dist.resolve()):raise HTTPException(404)
        if p.is_file():return FileResponse(p)
        if (dist/'index.html').exists():return FileResponse(dist/'index.html')
        raise HTTPException(503,'Build frontend: cd studio/frontend && npm install && npm run build')
    return app


def main():
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765)
    p.add_argument('--allow-training',action='store_true');p.add_argument('--sim-python',default='/root/isaacsim/python.sh');p.add_argument('--mujoco-python',default='/root/gpufree-data/envs/go2-mapless/bin/python');a=p.parse_args()
    if a.allow_training and a.host not in ('127.0.0.1','localhost','::1'):
        p.error('Training control must bind to loopback; use an authenticated SSH tunnel/reverse proxy.')
    import uvicorn
    uvicorn.run(create_app(a.root,a.allow_training,a.sim_python,a.mujoco_python),host=a.host,port=a.port,timeout_graceful_shutdown=3)

if __name__=='__main__':main()
