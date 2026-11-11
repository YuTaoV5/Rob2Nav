import {useEffect,useRef,useState} from 'react';
import SimulationTelemetry,{type PhysicalFrame} from './SimulationTelemetry';
type Obj=Record<string,any>;
async function api(path:string,body?:unknown){const r=await fetch('/api/live-skills/'+path,body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json','X-Studio-Request':'1'},body:JSON.stringify(body)});if(!r.ok)throw Error(await r.text());return r.json()}
const initialCamera={azimuth:130,elevation:-25,distance:3,pan_x:0,pan_y:0};
export default function LiveSkillPanel(){
 const [skills,setSkills]=useState<Obj[]>([]),[skill,setSkill]=useState(()=>new URLSearchParams(location.hash.split('?')[1]||'').get('skill')||'sideflip'),[state,setState]=useState<Obj>({phase:'IDLE'}),[image,setImage]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false);
 const cam=useRef({...initialCamera}),drag=useRef<Obj|null>(null),cameraTimer=useRef<ReturnType<typeof setTimeout>|null>(null),imageRef=useRef('');
 const imageNode=useRef<HTMLImageElement>(null);
 const [samples,setSamples]=useState<PhysicalFrame[]>([]);
 const keys=useRef(new Set<string>());
 const [control,setControl]=useState(false);
 const session=state.manual_control&&['RUNNING','STARTING'].includes(state.phase)?state.id:'';
 useEffect(()=>{
   setSamples([]);
   setControl(false);
 },[state.id]);
 useEffect(()=>{
   if(!state.telemetry||typeof state.time!=='number')return;
   const frame={...state.telemetry,time:state.time} as PhysicalFrame;
   setSamples(old=>old.at(-1)?.time===frame.time?old:[...old.slice(-239),frame]);
 },[state.id,state.time]);
 useEffect(()=>{
   if(!session||!control)return;
   let pending=false;
   function send(vx:number,vy:number,wz:number){
     if(pending)return;
     pending=true;
     api('command',{session,vx,vy,wz}).catch(()=>{}).finally(()=>{pending=false});
   }
   const clear=()=>{keys.current.clear();send(0,0,0)};
   const onVisibility=()=>{if(document.hidden)clear()};
   window.addEventListener('blur',clear);
   document.addEventListener('visibilitychange',onVisibility);
   const timer=setInterval(()=>{
     if(!document.hasFocus()||document.hidden){clear();return}
     const k=keys.current;
     const delta=(a:string,b:string)=>Number(k.has(a))-Number(k.has(b));
     const pad=Array.from(navigator.getGamepads?.()??[]).find(p=>p?.mapping==='standard'&&p.buttons[4]?.pressed);
     const axis=(n:number)=>Math.abs(n)>.13?Math.max(-1,Math.min(1,n)):0;
     send(pad?-.35*axis(pad.axes[1]):.35*delta('w','s'),
          pad?-.20*axis(pad.axes[0]):.20*delta('a','d'),
          pad?-.50*axis(pad.axes[2]):.50*delta('q','e'));
   },85);
   return ()=>{clearInterval(timer);window.removeEventListener('blur',clear);document.removeEventListener('visibilitychange',onVisibility);
     keys.current.clear();api('command',{session,vx:0,vy:0,wz:0}).catch(()=>{});
   };
 },[session,control]);
 const embedded=['navigation','handshake','scan'].includes(skill);
 useEffect(()=>{fetch('/api/capabilities').then(r=>r.json()).then(d=>setSkills(d.capabilities)).catch(e=>setError(String(e)))},[]);
 useEffect(()=>{if(embedded)return;let alive=true,inflight=false;async function poll(){if(inflight)return;inflight=true;try{const s=await api('state');if(!alive)return;setState(s);if(s.frames){const r=await fetch('/api/live-skills/frame.jpg',{cache:'no-store'});if(r.ok){const u=URL.createObjectURL(await r.blob());if(!alive){URL.revokeObjectURL(u);return}if(imageRef.current)URL.revokeObjectURL(imageRef.current);imageRef.current=u;setImage(u)}}}catch(e){if(alive)setError(String(e))}finally{inflight=false}}poll();const t=setInterval(poll,160);return()=>{alive=false;clearInterval(t);if(imageRef.current)URL.revokeObjectURL(imageRef.current)}},[embedded]);
 useEffect(()=>()=>{if(cameraTimer.current)clearTimeout(cameraTimer.current)},[]);
 function sendCamera(){if(cameraTimer.current)return;cameraTimer.current=setTimeout(()=>{cameraTimer.current=null;api('camera',cam.current).catch(e=>setError(String(e)))},60)}
 async function launch(){setBusy(true);setError('');setImage('');try{setState(await api('start',{skill}));cam.current={...initialCamera}}catch(e){setError(String(e))}finally{setBusy(false)}}
 useEffect(()=>{const node=imageNode.current;if(!node)return;const wheel=(e:WheelEvent)=>{e.preventDefault();cam.current.distance=Math.max(.5,Math.min(18,cam.current.distance*Math.exp(e.deltaY*.001)));sendCamera()};node.addEventListener('wheel',wheel,{passive:false});return()=>node.removeEventListener('wheel',wheel)},[!!image]);
 const names:Obj={IDLE:'选择技能后启动',STARTING:'正在加载模型',RUNNING:'实时物理仿真',COMPLETED:'本次执行结束',STOPPED:'已停止',FAILED:'执行失败'};
 return <section className="card"><div className="section-header"><div><span className="eyebrow">LIVE SKILLS / MUJOCO</span><h2>部署技能到仿真</h2><p>从工作台选择技能，在服务器实际运行物理推理。视角操作只改变相机。</p></div></div><div className="online-controls"><label>工作台技能<select aria-label="工作台仿真技能" value={skill} onChange={e=>{setSkill(e.target.value);setError('')}}>{skills.map(s=><option value={s.id} key={s.id}>{s.name}</option>)}<option value="go2_teleop">Go2 步态 · 手柄与键盘</option><option value="handshake">左右爪托手互动</option><option value="scan">SCAN 原生规划仿真</option></select></label>{!embedded&&<><button className="button primary" disabled={busy} onClick={launch}>{busy?'正在部署…':'部署并启动 MuJoCo'}</button><button className="button secondary" onClick={()=>api('stop',{}).catch(e=>setError(String(e)))}>停止仿真</button></>}<a className="button quiet" href={'#reports?path='+encodeURIComponent(skill==='handshake'?'go2_nav/paw_contact/DEPLOYMENT.md':skill==='scan'?'skills/navigation/SCAN_SIMULATION.md':`skills/${skill}/SIMULATION.md`)}>仿真手册</a></div>
 {error&&<div className="alert danger">{error}</div>}
 {embedded?<iframe key={skill} title={skill+'实时仿真'} src={skill==='navigation'?'/navigation-live':skill==='handshake'?'/handshake':'/scan-live'} style={{width:'100%',height:940,border:0,borderRadius:12}}/>:<><div className="terrain-meta"><span>{names[state.phase]||state.phase}</span><span>{state.skill||'—'}</span><span>{Number(state.time||0).toFixed(2)} s · {state.frames||0} 帧</span></div><p>{state.note}</p>{image?<img ref={imageNode} src={image} alt="所选技能的实时 MuJoCo 物理画面" style={{width:'100%',maxHeight:640,objectFit:'contain',background:'#e7ebf0',touchAction:'none',cursor:'grab',borderRadius:12}} draggable={false} onContextMenu={e=>e.preventDefault()} onPointerDown={e=>{drag.current={x:e.clientX,y:e.clientY,pan:e.button===2||e.shiftKey};e.currentTarget.setPointerCapture(e.pointerId)}} onPointerUp={()=>drag.current=null} onPointerCancel={()=>drag.current=null} onPointerMove={e=>{const d=drag.current;if(!d)return;const dx=e.clientX-d.x,dy=e.clientY-d.y;d.x=e.clientX;d.y=e.clientY;if(d.pan){cam.current.pan_x=Math.max(-10,Math.min(10,cam.current.pan_x-dx*.005));cam.current.pan_y=Math.max(-10,Math.min(10,cam.current.pan_y+dy*.005))}else{cam.current.azimuth=((cam.current.azimuth-dx*.4+540)%360)-180;cam.current.elevation=Math.max(-89,Math.min(-5,cam.current.elevation-dy*.3))}sendCamera()}}/>:<div className="empty"><h3>{names[state.phase]}</h3><p>点击部署后加载真实策略；不会用历史视频替代实时结果。</p></div>}<div className="viewport-toolbar"><span>左键拖动旋转 · 右键或 Shift 拖动平移 · 滚轮缩放</span><button className="button quiet" onClick={()=>{cam.current={...initialCamera};sendCamera()}}>重置相机</button></div><SimulationTelemetry frames={samples}/>
 {state.telemetry&&<small>在线物理流未记录训练奖励。具备逐步奖励的评估请到历史回放 → 任务诊断。</small>}
 {state.manual_control&&<section className="sim-inspector">
   <header><h3>仿真速度控制</h3><span>Go2 · {state.controller?.phase??'等待控制器'}</span></header>
   <label><input type="checkbox" checked={control} disabled={!session} onChange={e=>setControl(e.target.checked)}/> 接入键盘或手柄</label>
   <div className="sim-control-pad" tabIndex={control?0:-1} aria-label="仿真键盘控制区"
     onBlur={e=>{if(!e.currentTarget.contains(e.relatedTarget))keys.current.clear()}}
     onKeyDown={e=>{const key=e.key.toLowerCase();if(control&&'wasdqe'.includes(key)&&key.length===1){e.preventDefault();keys.current.add(key)}}}
     onKeyUp={e=>keys.current.delete(e.key.toLowerCase())}>
     {Object.entries({w:'前进 W',s:'后退 S',a:'左移 A',d:'右移 D',q:'左转 Q',e:'右转 E'}).map(([key,label])=>
       <button key={key} className="button secondary" disabled={!control} onPointerDown={e=>{e.currentTarget.setPointerCapture(e.pointerId);keys.current.add(key)}}
         onPointerUp={()=>keys.current.delete(key)} onPointerCancel={()=>keys.current.delete(key)} onLostPointerCapture={()=>keys.current.delete(key)}>{label}</button>)}
   </div>
   <small>键盘先聚焦控制区。标准手柄按住 LB 启用，左摇杆平移，右摇杆横轴转向。松键、失焦或超时归零。</small>
   <p>速度请求 vx / vy / wz　{(state.command??[0,0,0]).map((n:number)=>n.toFixed(2)).join(' / ')}　m/s · m/s · rad/s</p>
   <p>控制器指令　{(state.controller?.velocity??[0,0,0]).map((n:number)=>n.toFixed(2)).join(' / ')}</p>
   {state.controller?.fault&&<p role="alert">{state.controller.fault}</p>}
 </section>}
 {state.result&&<div className="terrain-meta"><span>动作判定：{state.result.successes}/{state.result.total}</span>{state.result.hardware_constraints_pass!==undefined&&<span>电机约束：{state.result.hardware_constraints_pass?"通过":"未通过"}</span>}</div>}{state.error&&<pre className="alert danger">{state.error}</pre>}<small>一次运行结束后可重新启动。页面失联8秒停止进程；模型失败会如实显示，启动成功不表示能力验收通过。当前会话：{state.id||'—'}。</small></>}
 </section>
}
