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


def create_app(root=None,allow_training=False,sim_python=None,mujoco_python=None):
    root=Path(root or Path(__file__).resolve().parents[1]).resolve()
    sim_python=Path(sim_python or '/root/isaacsim/python.sh')
    mujoco_python=Path(mujoco_python or '/root/gpufree-data/envs/go2-mapless/bin/python')
    catalog=Catalog(root);jobs=Jobs(root)
    app=FastAPI(title='Rob2Nav Studio',version='0.1.0')

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

    @app.get('/api/schema')
    def schema():return {'config':RunConfig.model_json_schema(),'defaults':RunConfig().model_dump(),'rewards':REWARDS,
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
        if any(s.startswith('.') for s in Path(path).parts) or not (p.is_relative_to(root/'artifacts') or p.is_relative_to(root/'outputs') or p.suffix=='.md'):
            raise HTTPException(404,'Artifact not published')
        if p.is_relative_to(root/'outputs/studio') and p.suffix not in ('.mp4','.png','.jpg','.jsonl','.pt'):
            raise HTTPException(404,'Use job API for runtime metadata')
        return FileResponse(p,filename=p.name if p.suffix=='.pt' else None)

    @app.get('/api/resources')
    def resources():
        d={'time':time.time(),'gpu':[],'training_enabled':allow_training,'simulator_available':sim_python.is_file(),'mujoco_available':mujoco_python.is_file()}
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
        return d

    def preflight(config):
        errors=[];warnings=[]
        output=root/'outputs/studio/runs/PREVIEW'
        runtime=mujoco_python if config.task=='simulation' else sim_python
        try:cmd=simulation_command(config,root,output,runtime) if config.task=='simulation' else command(config,root,output,root/'outputs/studio/PREVIEW.json',runtime)
        except ValueError as e:cmd=[];errors.append(str(e))
        if not runtime.is_file():errors.append('当前主机没有所选任务的执行解释器；可保存草稿，在服务器运行。')
        if config.task=='train' and not (root/'vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt').is_file():
            errors.append('缺少固定版本MoE基座权重；请先核对项目资源。')
        if os.name!='posix':errors.append('训练worker需要Linux主机；本地可浏览、编辑和保存配置。')
        if not allow_training:warnings.append('当前服务为只读训练模式，提交队列已关闭。')
        if config.task=='simulation':warnings.append('使用历史MoE/IK/NJU技能组合，仿真成绩不计入新感知MoE能力。')
        elif config.mode=='teacher':warnings.append('完整地图为特权教师输入，不能标为可部署相机/雷达策略。')
        if config.course=='wide_deep':warnings.append('训练前置几何与原AME严格评估分开，不计为原尺寸成功。')
        if config.initialization=='resume':warnings.append('恢复需匹配原参数；不恢复模拟器物理状态，不保证逐步确定性。')
        return {'valid':not errors,'errors':errors,'warnings':warnings,'command':cmd,'resolved':config.model_dump(),
                'environment_steps':config.num_envs*24*config.iterations,'estimated_hours':None}

    @app.post('/api/preflight')
    def validate(config:RunConfig):return preflight(config)

    @app.get('/api/jobs')
    def get_jobs():
        result=jobs.all()
        for job in result:
            folder=root/'outputs/studio/runs'/job['id'];job['progress']=read_json(folder/'status.json')
        return clean(result)

    @app.post('/api/jobs')
    def save(config:RunConfig):return jobs.create(config.model_dump())

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
