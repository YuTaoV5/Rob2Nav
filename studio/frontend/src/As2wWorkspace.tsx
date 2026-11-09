import {useEffect, useRef, useState} from 'react';
import './as2wWorkspace.css';

type Parameter = {id:string; default:number; min:number; max:number};
type Term = {id:string; name:string; kind:string; raw:string; formula:string; help:string; parameters:Parameter[]};
type Config = {source:string; function:string; line:number; code:string; terms:Term[]; training:boolean; baseline:string};
type Plan = Config & {diff:Record<string,{before:number; after:number}>};
type Sample = {step:number; dt:number; total:number; terms:Record<string,{raw:number; weighted:number}>};
type Case = {valid:boolean; case_id:string; terrain:string; seed:number; pose:string; success:boolean; stable_s:number; max_upright_cosine?:number; peak_impact_n:number; video:string|null};
type Comparison = {backend:string; baseline:{cases:Case[]}; candidate:{cases:Case[]}};
type Status = {state:string; error?:string};
type Run = {id:string; updates:number; smoke:boolean; status?:Status};
type Detail = {parameters:Record<string,number>|null; status:Status; latest:Sample|null; samples:Sample[]; comparison:Comparison|null};

async function request<T>(path:string, body?:object):Promise<T> {
  const response = await fetch('/api/as2w'+path, body ? {
    method:'POST', headers:{'Content-Type':'application/json','X-Studio-Request':'1'},
    body:JSON.stringify(body),
  } : undefined);
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail==='string' ? result.detail : JSON.stringify(result.detail));
  return result;
}

const parameterLabel:Record<string,string> = {upright_weight:'翻正权重',pose_weight:'站姿权重',impact_weight:'撞击惩罚系数',stance_weight:'承重与站稳权重',effort_base:'基础动作惩罚',effort_righted:'翻正后的附加动作惩罚'};
const kindLabel:Record<string,string> = {reward:'奖励',penalty:'惩罚',mixed:'双向项'};
const stateLabel:Record<string,string> = {queued:'排队',training:'训练中',evaluating:'MuJoCo 评估中',succeeded:'流程完成',failed:'失败'};
const numeric = (value:number|undefined) => value===undefined ? '—' : value.toPrecision(4);

function RewardChart({samples,id}:{samples:Sample[];id:string}) {
  if (samples.length<2) return <p className="muted">等待至少两次训练采样</p>;
  const values=samples.map(sample=>sample.terms[id].weighted);
  const min=Math.min(0,...values), max=Math.max(0,...values);
  const y=(value:number)=>76-(value-min)/(max-min||1)*72;
  const points=values.map((value,i)=>`${i/(values.length-1)*280},${y(value)}`).join(' ');
  return <svg viewBox="0 0 280 80" role="img" aria-label="实际加权贡献随训练步变化">
    <line x1="0" x2="280" y1={y(0)} y2={y(0)} stroke="currentColor" opacity=".28"/>
    <polyline points={points} fill="none" stroke="#4aa9ff" strokeWidth="2"/>
  </svg>;
}

function Results({comparison,run}:{comparison:Comparison;run:string}) {
  const [selected,select]=useState('');
  const [time,setTime]=useState(0);
  const [duration,setDuration]=useState(0);
  const videos=useRef<Record<string,HTMLVideoElement|null>>({});
  function seek(value:number) {
    setTime(value);
    Object.values(videos.current).forEach(video=>{if(video && Number.isFinite(video.duration))video.currentTime=Math.min(value,video.duration);});
  }
  const candidate=new Map(comparison.candidate.cases.map(row=>[row.case_id,row]));
  const base=comparison.baseline.cases.find(row=>row.case_id===selected)??comparison.baseline.cases[0];
  const other=candidate.get(base?.case_id);
  const count=(rows:Case[],terrain:string)=>{
    const group=rows.filter(row=>row.terrain===terrain);
    const invalid=group.filter(row=>!row.valid).length;
    return `${group.filter(row=>row.success).length}/${group.length-invalid}${invalid?` · 无效 ${invalid}`:''}`;
  };
  return <section className="card as2w-section">
    <h2>同场景对比 · {comparison.backend}</h2>
    <div className="as2w-grid">{[...new Set(comparison.baseline.cases.map(row=>row.terrain))].map(terrain=>
      <div key={terrain}><b>{({flat:'平地',slope:'坡面',rough:'粗糙地面'} as Record<string,string>)[terrain]??terrain}</b>
        <p>基线 {count(comparison.baseline.cases,terrain)} → 候选 {count(comparison.candidate.cases,terrain)}</p></div>)}</div>
    <label>场景 <select value={base?.case_id??''} onChange={e=>{select(e.target.value);setTime(0);setDuration(0);}}>{comparison.baseline.cases.map(row=>
      <option key={row.case_id} value={row.case_id}>{row.case_id} · 基线{row.success?'通过':'失败'} / 候选{candidate.get(row.case_id)?.success?'通过':'失败'}</option>
    )}</select></label>
    <div className="as2w-actions">{base?.video&&other?.video&&<label>两段视频共同定位 {time.toFixed(2)} s <input aria-label="同步定位视频" type="range" min="0" max={duration} step=".02" value={time} onChange={e=>seek(Number(e.target.value))}/></label>}</div>
    <div className="as2w-grid">{([['baseline','基线',base],['candidate','候选',other]] as const).map(([id,label,row])=>
      <article key={id}><h3>{label}</h3><p>{row?.valid===false?'物理无效':row?.success?'通过':'失败'} · 站稳 {numeric(row?.stable_s)} s · 峰值非轮撞击 {numeric(row?.peak_impact_n)} N</p>
        {row?.video ? <video key={row.video} ref={element=>{videos.current[id]=element;}} controls preload="metadata" onLoadedMetadata={e=>{const length=e.currentTarget.duration; if(Number.isFinite(length))setDuration(previous=>Math.max(previous,length));}} src={`/api/as2w/runs/${run}/video/${id}/${row.video}`}/> : <p className="muted">此场景没有录制视频</p>}</article>)}</div>
  </section>;
}

export default function As2wWorkspace() {
  const [config,setConfig]=useState<Config>();
  const [values,setValues]=useState<Record<string,string>>({});
  const [updates,setUpdates]=useState(2000);
  const [envs,setEnvs]=useState(256);
  const [plan,setPlan]=useState<Plan>();
  const [runs,setRuns]=useState<Run[]>([]);
  const [run,setRun]=useState('');
  const [detail,setDetail]=useState<Detail>();
  const [error,setError]=useState('');
  const [busy,setBusy]=useState(false);
  useEffect(()=>{
    request<Config>('').then(result=>{
      setConfig(result);
      setValues(Object.fromEntries(result.terms.flatMap(term=>term.parameters.map(p=>[p.id,String(p.default)]))));
    }).catch(e=>setError(e.message));
  },[]);
  useEffect(()=>{
    let current=true;
    const refresh=()=>request<Run[]>('/runs').then(rows=>{
      if(current){setRuns(rows);setRun(id=>id||rows[0]?.id||'');}
    }).catch(e=>{if(current)setError(e.message);});
    refresh();
    const timer=setInterval(refresh,5000);
    return ()=>{current=false;clearInterval(timer);};
  },[]);
  useEffect(()=>{
    setDetail(undefined);
    if(!run)return;
    let current=true;
    const refresh=()=>request<Detail>('/runs/'+run).then(value=>{if(current)setDetail(value);}).catch(e=>{if(current)setError(e.message);});
    refresh();
    const timer=setInterval(refresh,3000);
    return ()=>{current=false;clearInterval(timer);};
  },[run]);
  function edit(id:string,value:string) {
    setValues(previous=>({...previous,[id]:value}));
    setPlan(undefined);
  }
  async function action(submit:boolean) {
    setBusy(true);setError('');
    try {
      if(Object.values(values).some(value=>value.trim()===''))throw new Error('请填写完整的参数数值');
      const payload={parameters:Object.fromEntries(Object.entries(values).map(([key,value])=>[key,Number(value)])),updates,envs};
      if(submit) {
        const result=await request<{id:string}>('/runs',payload);
        setRuns(await request<Run[]>('/runs'));setRun(result.id);setPlan(undefined);
      } else setPlan(await request<Plan>('/preview',payload));
    } catch(e) {setError(e instanceof Error?e.message:String(e));}
    finally {setBusy(false);}
  }

  return <div className="as2w-workspace">
    <header className="page-heading"><div className="eyebrow">AS2-W · ISAAC LAB → MUJOCO</div><h1>AS2-W 奖励实验</h1>
      <p>查看恢复任务的真实奖励函数，预览参数调整，再用同一批场景比较候选策略。</p></header>
    {error&&<p role="alert" className="alert danger">{error}</p>}
    {!config ? <p>正在读取服务器上的奖励模块…</p> : <>
      <section className="card as2w-section"><h2>四轮起身与站稳</h2><p>候选从服务器的 R12 权重开始，用 R13 恢复奖励训练。起身缓冲期内，本页各项贡献均为零。</p>
        <details><summary>展开服务器原始函数 · {config.function}</summary><p>{config.source} · 第 {config.line} 行</p><pre>{config.code}</pre></details>
      </section>
      <section aria-label="奖励和惩罚" className="as2w-grid">{(plan??config).terms.map(term=>
        <article className="card as2w-module" key={term.id}><span className={'as2w-kind '+term.kind}>{kindLabel[term.kind]}</span><h2>{term.name}</h2>
          <p>{term.help}</p><p className="muted">公式 · {plan?'候选预览':'服务器默认值'}</p><code>{term.formula}</code>
          {term.parameters.map(p=><label key={p.id}>{parameterLabel[p.id]} · <code>{p.id}</code>
            <input aria-label={p.id} type="number" min={p.min} max={p.max} step="any" value={values[p.id]??''} disabled={busy} onChange={e=>edit(p.id,e.target.value)}/>
            <small>默认 {p.default} · 范围 {p.min} 至 {p.max}{detail?.parameters&&` · 所选运行 ${detail.parameters[p.id]}`}</small>
          </label>)}
          <div className="as2w-readings"><span>原始均值 <b>{numeric(detail?.latest?.terms[term.id]?.raw)}</b></span>
            <span>加权贡献 <b>{numeric(detail?.latest?.terms[term.id]?.weighted)}</b></span></div>
          <RewardChart samples={detail?.samples??[]} id={term.id}/>
        </article>)}</section>
      <section className="card as2w-section"><h2>新建候选</h2>
        <p>历史训练目录与权重保留。这里只调整上面列出的参数，验收使用同一套 MuJoCo 评估条件。</p>
        <div className="as2w-actions">
          <label>训练轮数 <input type="number" min="2000" max="40000" value={updates} disabled={busy} onChange={e=>{setUpdates(Number(e.target.value));setPlan(undefined);}}/></label>
          <label>并行环境 <input type="number" min="40" max="2048" value={envs} disabled={busy} onChange={e=>{setEnvs(Number(e.target.value));setPlan(undefined);}}/></label>
          <button className="button" disabled={busy} onClick={()=>action(false)}>预览差异</button>
          <button className="button" disabled={busy||!config.training||!Object.keys(plan?.diff??{}).length} onClick={()=>action(true)}>提交训练</button>
        </div>
        {!config.training&&<p className="muted">训练提交当前不可用，请检查服务器训练模式和评估器。</p>}
        {plan&&<ul aria-label="候选参数差异">{Object.keys(plan.diff).length ? Object.entries(plan.diff).map(([key,value])=><li key={key}>{key} · {value.before} → {value.after}</li>) : <li>当前配置与默认值相同</li>}</ul>}
      </section>
    </>}
    <section className="card as2w-section"><h2>运行记录</h2>
      {runs.length ? <select aria-label="选择实验" value={run} onChange={e=>setRun(e.target.value)}>{runs.map(item=><option value={item.id} key={item.id}>{item.smoke?'链路检查':'候选实验'} · {item.updates} 轮 · {stateLabel[item.status?.state??'']??'读取中'} · {item.id.slice(0,8)}</option>)}</select> : <p>暂无运行</p>}
      {detail?.status.error&&<p className="alert danger">{detail.status.error}</p>}
      {detail?.latest&&<p>最近采样 · 仿真控制步 {detail.latest.step} · 总奖励均值 {numeric(detail.latest.total)}（乘控制步长前）</p>}
      {run&&<p className="muted">卡片里的曲线来自当前选中的运行。输入框里的值属于下一次候选。</p>}
    </section>
    {detail?.comparison&&<Results key={run} comparison={detail.comparison} run={run}/>}
  </div>;
}
