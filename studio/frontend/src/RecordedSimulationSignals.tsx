import {useState} from 'react';
import {Curve} from './ExperimentReview';
import {PhysicalReadout,physicalSignals,metric,type PhysicalFrame} from './SimulationTelemetry';

type Reading={raw:number;weighted:number;weight?:number;gate?:boolean;label?:string};
type Frame={time:number;step?:number;telemetry?:Omit<PhysicalFrame,'time'>;signals?:Record<string,number>;terms?:Record<string,Reading>};
const seriesName={height_m:'机身高度 m',stable_streak_s:'连续站稳 s',nonwheel_impact_n:'非轮撞击 N'};
const signalValues=(f:Frame)=>({...f.signals,...(f.telemetry?physicalSignals({...f.telemetry,time:f.time}):{})});

export default function RecordedSimulationSignals({frames,time,seek}:{frames:Frame[];time:number;seek:(t:number)=>void}){
  const [selected,setSelected]=useState('');
  const [termId,setTermId]=useState('');
  if(!frames.length)return null;
  const sample=frames.reduce((best,f)=>Math.abs(f.time-time)<Math.abs(best.time-time)?f:best);
  const signals=Object.keys(signalValues(sample));
  const terms=Object.entries(sample.terms??{});
  const current=signals.includes(selected)?selected:signals[0];
  const term=sample.terms?.[termId]?termId:terms[0]?.[0];
  const max=frames.at(-1)?.time??0;
  const aligned=frames.every((f,i)=>f.step===i);
  const series=(name:string,read:(f:Frame)=>number|undefined)=>[{name,points:frames.flatMap(f=>{
    const value=read(f);
    return Number.isFinite(value)?[[f.time,value!]]:[];
  })}];
  const value=sample.terms?.[term];
  return <section className="sim-inspector">
    <header><h3>录制诊断</h3><span>{sample.time.toFixed(3)} s{sample.step!==undefined?' · 控制步 '+sample.step:''}</span></header>
    {!aligned&&<p>历史采样没有逐帧对齐契约，曲线按记录的模拟时间显示。</p>}
    {current&&<>
      <label>诊断量 <select aria-label="录制诊断量" value={current} onChange={e=>setSelected(e.target.value)}>{signals.map(k=><option key={k} value={k}>{seriesName[k as keyof typeof seriesName]??k}</option>)}</select></label>
      <Curve series={series('物理采样',f=>signalValues(f)[current])} max={max} time={time} seek={aligned?seek:undefined} label={seriesName[current as keyof typeof seriesName]??current}/>
    </>}
    {term&&value?<><label>奖励或惩罚 <select aria-label="录制奖励项" value={term} onChange={e=>setTermId(e.target.value)}>{terms.map(([k,v])=><option key={k} value={k}>{v.label??k}</option>)}</select></label>
      <Curve series={series('MuJoCo 复算',f=>f.terms?.[term]?.weighted)} max={max} time={time} seek={aligned?seek:undefined} label="加权贡献 / 秒"/>
      <p>原值 {metric(value.raw)} · 权重 {metric(value.weight)} · 门控 {value.gate===undefined?'未记录':value.gate?'生效':'关闭'} · 加权 {metric(value.weighted)}</p>
    </>:<p className="muted">该记录无可核对的逐步奖励。</p>}
    {sample.telemetry&&<PhysicalReadout frame={{...sample.telemetry,time:sample.time}}/>}
  </section>;
}
