import {useEffect, useState} from 'react';
import './experimentWorkflow.css';

export type RewardParameter = {id:string; label:string; value:number; default:number; min:number; max:number; editable:boolean};
export type RewardTerm = {
  id:string; label:string; kind:'reward'|'penalty'|'mixed'; group:string; formula:string;
  description:string; source:string; code?:string; unit?:string; available:boolean; parameters:RewardParameter[];
};
export type ExtraReward = {name:string; formula:string; weight:number};
export type RewardManifest = {terms:RewardTerm[]; support?:{editable:boolean;training_terms:string;mujoco_replay:string;mujoco_reward:string}; note?:string; sourceVersion?:string; sourceCode?:string; custom?:{signals:string[];presets:[string,string,number][]}};
export type CloneExperiment = {config:Record<string,unknown>; term?:string; baseline?:Record<string,number>};

const kindLabels={reward:'奖励',penalty:'惩罚',mixed:'双向'};
export function RewardEditor({manifest, onChange, focused, pending, error, baseline, extras=[], onExtras}:{
  manifest:RewardManifest|null; onChange:(id:string, value:number)=>void; focused?:string; pending:boolean; error:string;
  baseline?:Record<string,number>;
  extras?:ExtraReward[]; onExtras?:(value:ExtraReward[])=>void;
}) {
  const [filter,setFilter]=useState('');
  const [kind,setKind]=useState('all');
  const [showUnavailable,setShowUnavailable]=useState(false);
  const [adding,setAdding]=useState(false);
  const [selection,setSelection]=useState('');
  const [weight,setWeight]=useState('');
  const [draft,setDraft]=useState<ExtraReward>({name:'',formula:'',weight:0.1});
  const library=manifest?.terms.flatMap(t=>t.parameters.filter(p=>t.available&&p.editable).map(p=>({term:t,param:p})))||[];
  const item=library.find(x=>x.param.id===selection);
  useEffect(()=>{
    if(!focused || !manifest)return;
    const frame=requestAnimationFrame(()=>document.getElementById('reward-'+focused)?.scrollIntoView({block:'center'}));
    return ()=>cancelAnimationFrame(frame);
  },[focused,manifest?.terms.length]);
  if(error&&!manifest)return <p className="alert danger">{error}</p>;
  if(!manifest)return <p className="muted">正在读取技能的奖励定义…</p>;
  const terms=manifest.terms.filter(term=>(term.available||showUnavailable)
    &&(kind==='all'||term.kind===kind)
    &&`${term.id} ${term.label} ${term.group}`.toLowerCase().includes(filter.toLowerCase()));
  return <div className="experiment-rewards" aria-busy={pending}>
    <h3>奖励与惩罚</h3>
    {error&&<p className="alert danger" role="alert">{error}</p>}
    {manifest.note&&<p className="muted">{manifest.note}</p>}
    {manifest.support&&<div className="experiment-tools" aria-label="技能支持能力">
      <small>参数 · {manifest.support.editable?'可编辑':'只读或原配方'}</small>
      <small>训练分项 · {manifest.support.training_terms==='recorded'?'记录到训练日志':'取决于原训练器'}</small>
      <small>MuJoCo 回放 · {manifest.support.mujoco_replay==='paired'?'支持同协议配对':manifest.support.mujoco_replay==='recorded'?'原配方录制':'沿用原流程'}</small>
      <small>逐步奖励 · {manifest.support.mujoco_reward==='recomputed'?'支持复算':manifest.support.mujoco_reward==='subset'?'支持已接入分项':'该配方未接入'}</small>
    </div>}
    {manifest.sourceVersion&&<small title={manifest.sourceVersion}>函数与配置 SHA-256 <code>{manifest.sourceVersion.slice(0,12)}</code></small>}
    {manifest.sourceCode&&<details><summary>查看该技能的奖励函数源码</summary><pre className="code">{manifest.sourceCode}</pre></details>}
    {!!(manifest.terms.length||manifest.custom)&&<>
      <div className="experiment-tools">
        <input aria-label="搜索奖励与惩罚" placeholder="搜索名称或分类" value={filter} onChange={e=>setFilter(e.target.value)}/>
        <select aria-label="奖励类型" value={kind} onChange={e=>setKind(e.target.value)}>
          <option value="all">全部类型</option><option value="reward">奖励</option><option value="penalty">惩罚</option><option value="mixed">双向</option>
        </select>
        <label><input type="checkbox" checked={showUnavailable} onChange={e=>setShowUnavailable(e.target.checked)}/> 显示未启用项</label>
      </div>
      {adding&&<div className="experiment-reward-card">
        <h4>函数库</h4>
        {library.length>0&&<div className="experiment-tools">
          <label>已注册函数 <select aria-label="已注册函数" value={selection} onChange={e=>{
            setSelection(e.target.value);const p=library.find(x=>x.param.id===e.target.value)?.param;setWeight(p?String(p.value):'');
          }}><option value="">选择函数…</option>{library.map(({term,param})=><option value={param.id} key={param.id}>{term.label} · {param.label}</option>)}</select></label>
          {item&&<><input aria-label="函数权重" type="number" step="any" value={weight} onChange={e=>setWeight(e.target.value)}/>
            <button type="button" className="button secondary" disabled={!weight.trim()||!(Number(weight)>=item.param.min&&Number(weight)<=item.param.max)}
              onClick={()=>{onChange(item.param.id,Number(weight));setAdding(false);}}>应用函数权重</button></>}
        </div>}
        {manifest.custom&&onExtras&&<>
          <div className="experiment-tools">{manifest.custom.presets.map(([name,formula,weight])=>
            <button type="button" className="button quiet" key={name} onClick={()=>setDraft({name,formula,weight})}>{name}</button>)}</div>
          <p>自填公式支持数字、括号、加减乘。可用信号 {manifest.custom.signals.join('、')}。</p>
          <input aria-label="自定义函数名称" placeholder="函数名称" maxLength={40} value={draft.name} onChange={e=>setDraft({...draft,name:e.target.value})}/>
          <input aria-label="自定义奖励公式" placeholder="例如 -0.5 * action_cost" maxLength={160} value={draft.formula} onChange={e=>setDraft({...draft,formula:e.target.value})}/>
          <label>自定义权重 <input type="number" min="-4" max="4" step="any" value={draft.weight} onChange={e=>setDraft({...draft,weight:e.target.valueAsNumber})}/></label>
          <button type="button" className="button primary" disabled={!draft.name.trim()||!draft.formula.trim()||!(draft.weight>=-4&&draft.weight<=4)||extras.length>=8}
            onClick={()=>{onExtras([...extras,draft]);setDraft({name:'',formula:'',weight:.1});setAdding(false);}}>添加自定义函数</button>
        </>}
      </div>}
      <div className="experiment-reward-grid">{terms.map(term=><article id={'reward-'+term.id} key={term.id} className={'experiment-reward-card '+(focused===term.id?'focused':'')}>
        <div className="experiment-tools"><span className={'experiment-kind '+term.kind}>{kindLabels[term.kind]}</span><small>{term.group}{!term.available&&' · 当前未启用'}</small></div>
        <h4>{term.label}</h4>
        <p>{term.description}</p>
        <code className="experiment-formula">{term.formula}</code>
        {!!term.parameters.length&&<small>候选代入值 · {term.parameters.map(p=>p.label+' = '+p.value).join(' · ')}</small>}
        <small>{term.unit&&term.unit+' · '}{term.source}</small>
        {term.code&&<details><summary>查看函数源码</summary><pre className="code">{term.code}</pre></details>}
        {term.parameters.map(param=><RewardInput key={param.id} parameter={param} baseline={baseline?.[param.id]} enabled={term.available} onChange={onChange}/>)}
      </article>)}
      {extras.map((extra,index)=><article className={'experiment-reward-card '+(focused==='custom_'+index?'focused':'')} id={'reward-custom_'+index} key={index}>
        <h4>{extra.name}</h4><code className="experiment-formula">clamp({extra.formula}, -10, 10)</code>
        <RewardInput parameter={{id:'custom_'+index,label:'权重',value:extra.weight,default:extra.weight,min:-4,max:4,editable:true}}
          enabled onChange={(_,value)=>onExtras?.(extras.map((x,i)=>i===index?{...x,weight:value}:x))}/>
        <button type="button" className="button quiet" onClick={()=>onExtras?.(extras.filter((_,i)=>i!==index))}>移除函数</button>
      </article>)}
      {(library.length>0||manifest.custom)&&<button type="button" className="experiment-reward-card experiment-add-reward button secondary" onClick={()=>setAdding(!adding)}>＋ 添加函数</button>}
      </div>
      {!terms.length&&<p>没有符合筛选条件的奖励项。</p>}
    </>}
  </div>;
}

function RewardInput({parameter:p,baseline,enabled,onChange}:{parameter:RewardParameter;baseline?:number;enabled:boolean;onChange:(id:string,value:number)=>void}) {
  const [draft,setDraft]=useState(String(p.value));
  useEffect(()=>setDraft(String(p.value)),[p.value]);
  const value=Number(draft);
  const invalid=!draft.trim()||!Number.isFinite(value)||value<p.min||value>p.max;
  return <label className="field experiment-parameter">
    <span>{p.label} <code>{p.id}</code></span>
    <input type="number" step="any" aria-label={p.label+' '+p.id} aria-invalid={invalid} disabled={!enabled||!p.editable}
      min={p.min} max={p.max} value={draft}
      onChange={e=>{const next=e.target.value;setDraft(next);const v=Number(next);if(next.trim()&&Number.isFinite(v)&&v>=p.min&&v<=p.max)onChange(p.id,v);}}
      onBlur={()=>{if(invalid)setDraft(String(p.value));}}/>
    <small>默认 {p.default}{baseline!==undefined&&` · 基线 ${baseline}`} · {p.editable?`范围 ${p.min} 至 ${p.max}`:'当前只读'}</small>
    {p.editable&&p.value!==(baseline??p.default)&&<button className="button quiet" type="button" onClick={()=>onChange(p.id,baseline??p.default)}>{baseline!==undefined?'恢复基线值':'恢复默认值'}</button>}
  </label>;
}
