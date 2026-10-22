import {useRef,useState} from 'react';
import {ArrowRight,ArrowUpRight,ChevronDown,Cpu,Pause,Play,Volume2,VolumeX,Maximize,Video} from 'lucide-react';

type Obj=Record<string,any>;
type Props={data:Obj|null;catalog:Obj;resources:Obj;resourceError:string;resourceLoaded:boolean;jobs:Obj[];selected:string;onSelect:(id:string)=>void;onNavigate:(page:string)=>void;onSimulate:(id:string)=>void};
const presentation:Record<string,{name:string;english:string;intro:string}>={
 backflip:{name:'后空翻',english:'BACKFLIP',intro:'从稳定站立到腾空翻转，观察一个动作如何完成，又如何回到平衡。'},
 stairs:{name:'楼梯通行',english:'STAIR TRAVERSAL',intro:'走上台阶，再稳稳走下来。观察机器人在高度变化中的落脚与姿态。'},
 platform:{name:'高台通行',english:'PLATFORM TRAVERSAL',intro:'从平地登上高台，再返回地面。把每一次高度变化，变成可观察的动作。'},
 multiterrain:{name:'多地形行走',english:'TERRAIN ADAPTATION',intro:'当地形发生变化，观察技能如何衔接，以及在哪些地方仍需要改进。'},
 sideflip:{name:'侧空翻',english:'SIDE FLIP',intro:'从侧向起跳到落地恢复，查看实验中的翻转表现与当前验证边界。'},
 navigation:{name:'目标导航',english:'GOAL NAVIGATION',intro:'给定一个目标，观察机器人如何规划路径、绕过障碍并接近终点。'},
 upright_walk:{name:'直立行走',english:'UPRIGHT LOCOMOTION',intro:'探索更少支撑点下的平衡与移动。当前实验进度与历史演示分别记录。'},
 frontflip:{name:'前空翻',english:'FRONT FLIP',intro:'向前腾空，完成翻转。查看已有演示与从稳定起止到完整动作的验收要求。'},
};
const stateNames:Obj={draft:'草稿',queued:'排队中',preflight:'预检中',running:'训练中',evaluating:'评估中',succeeded:'已完成',completed:'已完成',failed:'失败',canceled:'已停止',unknown:'待核对',snapshot:'历史快照'};
const runNames:Obj={manual_reproduction:'手动复现实验',multiterrain_reproduction:'多地形复现实验',platform_reproduction:'高台复现实验',stairs_reproduction:'楼梯复现实验'};
const format=(value:unknown)=>typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('zh-CN',{maximumFractionDigits:1}):'—';

function Film({media,name}:{media:Obj;name:string}){
 const ref=useRef<HTMLVideoElement>(null);
 const [playing,setPlaying]=useState(false),[muted,setMuted]=useState(true),[failed,setFailed]=useState(false),[time,setTime]=useState(0),[duration,setDuration]=useState(0);
 const [auto]=useState(()=>!window.matchMedia('(prefers-reduced-motion: reduce)').matches);
 const clock=(s:number)=>`${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}`;
 return <div className="home-film">
  <video ref={ref} src={media.url} aria-label={name+'仿真回放'} autoPlay={auto} loop muted={muted} playsInline preload="metadata" onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onError={()=>setFailed(true)} onTimeUpdate={e=>setTime(e.currentTarget.currentTime)} onLoadedMetadata={e=>setDuration(Number.isFinite(e.currentTarget.duration)?e.currentTarget.duration:0)}/>
  <span className="home-film-label"><span/>{media.origin==='local'?'本地仿真记录':'上游作者演示'}</span>
  {failed&&<div className="home-film-error"><Video size={28}/><p>暂时无法播放这段记录</p><a href={media.url} target="_blank" rel="noreferrer">打开原始视频 <ArrowUpRight size={13}/></a></div>}
  <div className="home-film-controls">
   <button aria-label={playing?'暂停回放':'播放回放'} onClick={()=>{const v=ref.current;if(v){if(v.paused)v.play().catch(()=>setFailed(true));else v.pause()}}}>{playing?<Pause size={16}/>:<Play size={16}/>}</button>
   <span>{clock(time)}</span><input aria-label="回放进度" type="range" min={0} max={duration||1} step={.1} value={Math.min(time,duration||1)} disabled={!duration} onChange={e=>{if(ref.current)ref.current.currentTime=Number(e.target.value)}}/><span>{clock(duration)}</span>
   <button aria-label={muted?'开启回放声音':'静音回放'} onClick={()=>setMuted(v=>!v)}>{muted?<VolumeX size={15}/>:<Volume2 size={15}/>}</button>
   <button aria-label="全屏回放" onClick={()=>ref.current?.requestFullscreen?.().catch(()=>{})}><Maximize size={15}/></button>
  </div>
 </div>;
}

function Featured({skill,projects,onSimulate}:{skill:Obj;projects:Obj[];onSimulate:(id:string)=>void}){
 const [index,setIndex]=useState(0);
 const info=presentation[skill.id]||{name:skill.name,english:'ROBOT CAPABILITY',intro:skill.description};
 const related=projects.filter(p=>skill.projects?.includes(p.id));
 const media=[...(skill.local_videos||[]),...related.flatMap(p=>(p.media||[]).filter((m:Obj)=>!m.capabilities||m.capabilities.includes(skill.id)))];
 const current=media[Math.min(index,Math.max(0,media.length-1))];
 const local=current?.origin==='local';
 return <>
 <div className="home-feature">
  <div className="home-screen">
   {current?.type==='video'?<Film key={current.url} media={current} name={info.name}/>:current?.type==='image'?<div className="home-film"><img src={current.url} alt={current.title}/></div>:<div className="home-no-film"><Video size={30}/><h3>{current?'作者演示':'等待新的动作记录'}</h3><p>{current?'这段演示来自外部平台，可在原始页面观看。':'该能力暂无可嵌入回放，验证进展仍可查阅。'}</p>{current&&<a href={current.url} target="_blank" rel="noreferrer">观看作者演示 <ArrowUpRight size={14}/></a>}</div>}
   <div className="home-screen-caption"><span>{local?'实验回放':'演示资料'}<i>·</i>{info.name}</span>{media.length>1?<label className="home-media-picker"><span className="home-sr-only">选择回放</span><select aria-label="选择回放" value={index} onChange={e=>setIndex(Number(e.target.value))}>{media.map((m:Obj,i:number)=><option value={i} key={m.url+i}>{m.origin==='local'?'本地回放':'作者演示'} {i+1}{m.evaluation_id?' · 含评估':''}</option>)}</select><ChevronDown size={12}/></label>:<span>{current?'原始记录':'暂无回放'}</span>}</div>
  </div>
  <div className="home-story">
   <span className="home-overline">{info.english}</span><h2>{info.name}</h2><p className="home-intro">{info.intro}</p>
   <div className="home-verification"><span className="home-small-label">当前验证</span><p>{skill.local_status}</p><small>{local?'演示为本地仿真记录，实机资格需独立验证。':'作者演示与本地验收分别记录。'}</small></div>
   <div className="home-story-actions"><button className="home-primary" onClick={()=>onSimulate(skill.id)}>进入在线仿真<ArrowRight size={16}/></button><a className="home-text-action" href={'#capabilities?skill='+encodeURIComponent(skill.id)}>查看能力详情<ArrowUpRight size={13}/></a></div>
  </div>
 </div>
 <details className="home-evidence"><summary>回放来源与实验详情<ChevronDown size={13}/></summary><div><p>{skill.description}</p>{current&&<p><b>原始记录：</b>{current.title} · {current.label} <a href={current.url} target="_blank" rel="noreferrer">查看原始媒体 ↗</a></p>}{current?.evaluation_id&&<a href={'#simulation?evaluation='+encodeURIComponent(current.evaluation_id)}>完整轨迹与评估 →</a>}<div className="home-evidence-links"><a href={'#models?skill='+encodeURIComponent(skill.id)}>模型与证据</a>{skill.manuals?.map((m:Obj)=><a key={m.kind} href={m.url}>{m.label}</a>)}</div></div></details>
 </>;
}

export default function HomeOverview({data,catalog,resources,resourceError,resourceLoaded,jobs,selected,onSelect,onNavigate,onSimulate}:Props){
 const skills:Obj[]=data?.capabilities||[];
 const ordered=[...skills].sort((a,b)=>['backflip','stairs','platform','sideflip','multiterrain','navigation','upright_walk','frontflip'].indexOf(a.id)-['backflip','stairs','platform','sideflip','multiterrain','navigation','upright_walk','frontflip'].indexOf(b.id));
 const skill=skills.find(s=>s.id===selected)||skills[0];
 const gpu=resources.gpu?.[0],active=jobs.filter(j=>['running','preflight','evaluating'].includes(j.state));
 const recent=active.length?active.slice(0,3):catalog.runs.slice(0,3);
 return <div className="home-v2">
  <header className="home-heading"><div><span className="home-overline">ROB2NAV STUDIO · GO2</span><h1>让能力，<span>看得见。</span></h1><p>从一次动作开始，连接训练、仿真与验证。</p></div><a href="#capabilities" className="home-all">探索全部能力<ArrowUpRight size={15}/></a></header>
  <section className="home-showcase" aria-label="机器人能力预览">
   <div className="home-capability-tabs" aria-label="选择预览技能">{ordered.map(s=><button key={s.id} aria-pressed={skill?.id===s.id} className={skill?.id===s.id?'is-selected':''} onClick={()=>onSelect(s.id)}>{presentation[s.id]?.name||s.name}</button>)}</div>
   {skill?<Featured key={skill.id} skill={skill} projects={data?.projects||[]} onSimulate={onSimulate}/>:<div className="home-empty">正在读取能力与真实回放…</div>}
  </section>
  <div className="home-lower">
   <section className="home-activity"><div className="home-section-title"><h2>{active.length?'正在进行':'最近实验'}</h2><button className="home-text-action" onClick={()=>onNavigate('experiments')}>任务队列<ArrowRight size={13}/></button></div>
   {recent.length?recent.map((r:Obj)=><button className="home-run" key={r.id} onClick={()=>onNavigate('experiments')}><div><strong title={r.name}>{runNames[r.name]||r.name}</strong><span>{r.config?.task==='simulation'?'仿真任务':active.length?'训练任务':'历史训练记录'}{r.iteration!=null&&` · ${format(r.iteration)} 次迭代`}</span></div><span className="home-run-state">{stateNames[r.state]||r.state}<ArrowUpRight size={13}/></span></button>):<p className="home-empty">创建实验后，进展会显示在这里。</p>}
   </section>
   <section className="home-compute"><div className="home-section-title"><h2>计算资源</h2><Cpu size={16}/></div><div className="home-device">{gpu?.name||(resourceLoaded?'暂无GPU设备':'正在连接…')}<span>{resourceError?'连接异常':'服务器遥测'}</span></div><div className="home-compute-metrics"><div><span>GPU 利用率</span><strong>{gpu?format(Number(gpu.utilization)):'—'}<small>%</small></strong></div><div><span>系统内存</span><strong>{format(resources.memory_percent)}<small>%</small></strong></div><div><span>可用磁盘</span><strong>{format(resources.disk_free_gb)}<small>GB</small></strong></div></div><p className="home-resource-note">{resourceError?resourceError:'每 8 秒更新 · '+(resources.training_enabled?'训练需手动提交':'当前训练提交已关闭')}</p></section>
  </div>
  <div className="home-archive"><span>已收录的实验资产</span>{[[catalog.runs.length,'训练记录','experiments'],[catalog.evaluations.length,'评估回合组','compare'],[catalog.videos.length,'仿真视频','simulation'],[catalog.reports.length,'研究报告','reports']].map(([n,label,page])=><button key={String(page)} onClick={()=>onNavigate(String(page))}><b>{format(n)}</b>{label}<ArrowUpRight size={12}/></button>)}</div>
 </div>;
}
