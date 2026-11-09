import {useEffect,useRef,useState} from 'react';
import type {CloneExperiment} from './ExperimentRewards';

type TermValue={raw:number;weighted:number;label?:string};
type Frame={time:number;terms:Record<string,TermValue>;signals?:Record<string,number>};
type Trace={frames:Frame[];label:string};
type Side={passed:boolean|null;valid:boolean;stable_s?:number;video?:string;hasTrace:boolean};
type Case={id:string;label:string;sides:Record<string,Side>};
type Overview={cases:Case[];train:{step:number;terms:Frame['terms']}[];state:string;isTraining:boolean};
const sideName:Record<string,string>={baseline:'基线',candidate:'候选'};
const known:Record<string,string>={stance:'承重与站稳',impact:'非轮撞击',upright:'直立度',pose:'站姿',effort:'动作幅度',
  wheel_min_n:'最小轮载（N）',height_m:'机身高度（m）',roll_rad:'横滚（rad）',angular_speed:'角速度',speed:'速度',
  stable_streak_s:'连续站稳（s）',active:'策略控制门控',righted_gate:'翻正门控',standard_pose_error:'站姿误差',all_feet_ground:'四足接触'};
const display=(value:number|undefined)=>value===undefined?'—':value.toPrecision(4);
async function get<T>(id:string,path:string):Promise<T>{
  const res=await fetch(`/api/jobs/${id}/${path}`);
  if(!res.ok){const body=await res.json();throw new Error(typeof body.detail==='string'?body.detail:'无法读取实验数据');}
  return res.json();
}

function Curve({series,time,max,seek,label}:{series:{name:string;points:number[][]}[];time:number;max:number;seek?:(time:number)=>void;label:string}){
  const values=series.flatMap(s=>s.points.map(p=>p[1]));
  if(values.length<2)return <p className="muted">当前缺少足够的采样点。</p>;
  const low=Math.min(0,...values),high=Math.max(0,...values);
  const x=(v:number)=>v/(max||1)*560,y=(v:number)=>105-(v-low)/(high-low||1)*100;
  const colors=['#50a5ff','#e28b48'];
  return <figure className="experiment-timeline"><figcaption>{label} · {series.map((s,i)=><span key={s.name} style={{color:colors[i]}}>{s.name} </span>)}</figcaption>
    <svg viewBox="0 0 560 110" width="100%" height="125" preserveAspectRatio="none" aria-label={label} role="img"
      onPointerDown={e=>{const b=e.currentTarget.getBoundingClientRect();seek?.(Math.max(0,Math.min(max,(e.clientX-b.left)/b.width*max)));}}>
      <line x1="0" x2="560" y1={y(0)} y2={y(0)} stroke="currentColor" opacity=".3"/>
      {series.map((s,i)=><polyline key={s.name} points={s.points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')} fill="none" stroke={colors[i]} strokeWidth="2"/>)}
      {seek&&<line x1={x(time)} x2={x(time)} y1="0" y2="110" stroke="currentColor" opacity=".65"/>}
    </svg>
  </figure>;
}

export default function ExperimentReview({id,onCandidate}:{id:string;onCandidate:(next:CloneExperiment)=>void}){
  const [data,setData]=useState<Overview>();
  const [error,setError]=useState('');
  const [caseId,select]=useState('');
  const [term,setTerm]=useState('');
  const [trainTerm,setTrainTerm]=useState('stance');
  const [traces,setTraces]=useState<Record<string,Trace>>({});
  const [signal,setSignal]=useState('');
  const [time,setTime]=useState(0);
  const [duration,setDuration]=useState(0);
  const videos=useRef<Record<string,HTMLVideoElement|null>>({});
  useEffect(()=>{
    let live=true;
    const refresh=()=>get<Overview>(id,'experiment-review').then(value=>{if(live)setData(value);}).catch(e=>{if(live)setError(e.message);});
    refresh();const timer=setInterval(refresh,4000);
    return()=>{live=false;clearInterval(timer);};
  },[id]);
  const scene=data?.cases.find(c=>c.id===caseId)??data?.cases.find(c=>Object.values(c.sides).some(s=>s.passed===false))??data?.cases[0];
  useEffect(()=>{
    setTraces({});setTime(0);setDuration(0);
    if(!scene)return;
    let live=true;
    Promise.all(Object.entries(scene.sides).filter(([,side])=>side.hasTrace).map(async ([side])=>
      [side,await get<Trace>(id,`experiment-trace?case=${encodeURIComponent(scene.id)}&side=${side}`)] as const,
    )).then(entries=>{if(live)setTraces(Object.fromEntries(entries));}).catch(e=>{if(live)setError(e.message);});
    return()=>{live=false;};
  },[id,scene?.id,Object.values(scene?.sides??{}).map(s=>s.hasTrace).join(',')]);
  if(error&&!data)return <p className="alert danger">{error}</p>;
  if(!data||!data.isTraining)return null;
  const termInfo=new Map<string,string>();
  const signals=new Set<string>();
  for(const trace of Object.values(traces))for(const frame of trace.frames){
    for(const [key,value] of Object.entries(frame.terms??{}))termInfo.set(key,value.label??known[key]??key);
    for(const key of Object.keys(frame.signals??{}))signals.add(key);
  }
  const selectedTerm=termInfo.has(term)?term:[...termInfo.keys()][0];
  const selectedSignal=signals.has(signal)?signal:[...signals.keys()][0];
  const max=Math.max(duration,...Object.values(traces).map(t=>t.frames[t.frames.length-1]?.time??0));
  const events=Object.entries(traces).flatMap(([side,t])=>{
    const candidates=[
      ['控制接管',t.frames.find(f=>f.signals?.active===1)],
      ['站稳达 1 秒',t.frames.find(f=>(f.signals?.stable_streak_s??0)>=1)],
      ['站稳中断',t.frames.find((f,i)=>f.signals?.stable_streak_s===0&&(t.frames[i-1]?.signals?.stable_streak_s??0)>=.12)],
      ['四足重新接地',t.frames.find((f,i)=>f.signals?.all_feet_ground===1&&t.frames[i-1]?.signals?.all_feet_ground===0&&f.time>.4)],
    ] as const;
    return candidates.flatMap(([label,frame])=>frame?[{side,label,time:frame.time}]:[]);
  });
  function seek(value:number){
    setTime(value);
    Object.values(videos.current).forEach(v=>{if(v&&Number.isFinite(v.duration)&&Math.abs(v.currentTime-value)>.015)v.currentTime=Math.min(value,v.duration);});
  }
  async function clone(focused?:string){
    setError('');
    try{onCandidate({...await get<CloneExperiment>(id,'candidate-draft?case='+encodeURIComponent(scene?.id??'')),term:focused});}
    catch(e){setError(e instanceof Error?e.message:String(e));}
  }
  const series=(read:(f:Frame)=>number|undefined)=>Object.entries(traces).map(([side,t])=>({
    name:sideName[side],points:t.frames.flatMap(f=>{const value=read(f);return value===undefined?[]:[[f.time,value]];}),
  }));
  return <section className="experiment-diagnostics">
    <div className="section-header"><div><h3>MuJoCo 诊断</h3><p className="muted">曲线仅使用同次回放采样，训练曲线单独按训练进度显示。</p></div>
      <button className="button secondary" onClick={()=>clone()}>从此实验创建候选</button></div>
    {error&&<p className="alert danger">{error}</p>}
    {data.train.length>0&&<details><summary>训练分项趋势 · Isaac Lab</summary>
      <label>训练奖励项 <select aria-label="训练奖励项" value={trainTerm} onChange={e=>setTrainTerm(e.target.value)}>{Object.keys(data.train[0].terms).map(key=><option key={key} value={key}>{known[key]??key}</option>)}</select></label>
      <Curve series={[{name:'并行环境均值',points:data.train.map(s=>[s.step,s.terms[trainTerm]?.weighted??0])}]} max={data.train[data.train.length-1]?.step??1} time={0} label="加权贡献 / 训练控制步"/>
    </details>}
    {scene?<><label className="field"><span>评估场景</span><select aria-label="MuJoCo 评估场景" value={scene.id} onChange={e=>select(e.target.value)}>{data.cases.map(c=><option key={c.id} value={c.id}>{c.label} · {Object.entries(c.sides).map(([side,v])=>sideName[side]+' '+(v.passed?'通过':v.valid?'失败':'物理无效')).join(' / ')}</option>)}</select></label>
      <div className="experiment-videos">{Object.entries(scene.sides).map(([side,v])=><article key={side}><h4>{sideName[side]} · {v.passed?'通过':v.valid?'失败':'物理无效'}</h4>
        {v.stable_s!==undefined&&<p>连续站稳 {display(v.stable_s)} s</p>}
        {v.video?<video key={v.video} ref={node=>{videos.current[side]=node;}} controls preload="metadata" src={v.video}
          onLoadedMetadata={e=>{const seconds=e.currentTarget.duration;if(Number.isFinite(seconds))setDuration(d=>Math.max(d,seconds));}}
          onTimeUpdate={e=>{if(!e.currentTarget.paused)setTime(e.currentTarget.currentTime);}}/>:<p className="muted">此场景未录制视频。</p>}</article>)}</div>
      {max>0&&<label className="experiment-timeline">共同时间轴 · {time.toFixed(2)} s<input aria-label="同步仿真时间轴" type="range" min="0" max={max} step=".01" value={Math.min(time,max)} onChange={e=>seek(Number(e.target.value))}/></label>}
      {!!events.length&&<div className="experiment-tools" aria-label="回放事件">{events.map(event=><button className="button quiet" key={event.side+event.label} onClick={()=>seek(event.time)}>{sideName[event.side]} · {event.label} · {event.time.toFixed(2)} s</button>)}</div>}
      {selectedTerm?<><div className="experiment-tools"><label>奖励或惩罚 <select aria-label="MuJoCo 奖励项" value={selectedTerm} onChange={e=>setTerm(e.target.value)}>{[...termInfo].map(([key,label])=><option key={key} value={key}>{label}</option>)}</select></label>
        <button className="button secondary" onClick={()=>clone(selectedTerm)}>带此项创建候选</button></div>
        <p className="muted">{[...new Set(Object.values(traces).map(t=>t.label))].join(' · ')}</p>
        <Curve max={max} time={time} seek={seek} series={series(f=>f.terms?.[selectedTerm]?.weighted)} label="加权奖励 / 惩罚"/>
        <div className="experiment-tools">{Object.entries(traces).map(([side,t])=>{const point=t.frames.reduce<Frame|undefined>((best,frame)=>!best||Math.abs(frame.time-time)<Math.abs(best.time-time)?frame:best,undefined);const value=point?.terms?.[selectedTerm];return <span key={side}>{sideName[side]}原始值 {display(value?.raw)} · 加权 {display(value?.weighted)}</span>;})}</div>
      </>:<p className="muted">此场景没有可复算的逐帧奖励。已有物理量仍可查看。</p>}
      {selectedSignal&&<><label>诊断量 <select aria-label="MuJoCo 诊断量" value={selectedSignal} onChange={e=>setSignal(e.target.value)}>{[...signals].map(key=><option key={key} value={key}>{known[key]??key}</option>)}</select></label>
        <Curve max={max} time={time} seek={seek} series={series(f=>f.signals?.[selectedSignal])} label={known[selectedSignal]??selectedSignal}/></>}
    </>:<p className="muted">当前任务尚无此处可用的 MuJoCo 诊断记录。原有日志与回放入口仍可使用。</p>}
  </section>;
}
