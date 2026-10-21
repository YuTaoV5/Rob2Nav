import {useState} from 'react';
import {ArrowUpRight,Play,Video} from 'lucide-react';
type Obj=Record<string,any>;
export default function WorkbenchEvidence({data,models,selected}:{data:Obj|null;models:Obj[];selected:string}){
 const [choice,setChoice]=useState(0);
 const skill=data?.capabilities?.find((s:Obj)=>s.id===selected)||data?.capabilities?.[0];
 const projects=(data?.projects||[]).filter((p:Obj)=>skill?.projects?.includes(p.id));
 const media=[...(skill?.local_videos||[]),...projects.flatMap((p:Obj)=>(p.media||[]).filter((m:Obj)=>!m.capabilities||m.capabilities.includes(skill?.id)))];
 const current=media[Math.min(choice,Math.max(0,media.length-1))];
 const weights=models.filter(m=>m.skill===skill?.id&&['selected','queue_selected','failed_baseline','reproduction'].includes(m.role));
 return <section className="card featured" aria-label="工作台技能证据"><div className="section-header"><div><span className="eyebrow">SKILL EVIDENCE</span><h2>{skill?.name||'技能证据'}</h2></div><span className="badge neutral">{skill?.local_status||'正在读取'}</span></div>
 <div className="feature-video">{current?.type==='video'?<video key={current.url} src={current.url} controls autoPlay muted playsInline preload="metadata"/>:current?.type==='image'?<img src={current.url} alt={current.title}/>:current?.type==='youtube'?<iframe title={current.title} src={'https://www.youtube.com/embed/'+current.url.split('/').at(-1)} allowFullScreen/>:current?.type==='bilibili'?<iframe title={current.title} src={'https://player.bilibili.com/player.html?bvid='+current.url.match(/BV[\w]+/)?.[0]+'&autoplay=0'} allowFullScreen/>:<div className="empty"><Video size={30}/><p>该技能暂无可播放视频，验收状态与手册见下方。</p></div>}</div>
 {current&&<div className="media-attribution"><b>{current.origin==='local'?'本项目仿真记录':'上游作者演示 · 非本地验收'}</b><span>{current.title} · {current.label}</span><a href={current.url} target="_blank" rel="noreferrer">原始媒体</a></div>}
 {media.length>1&&<div className="media-choices">{media.map((m:Obj,i:number)=><button key={m.url+i} className={choice===i?'selected':''} onClick={()=>setChoice(i)}><Play size={12}/>{m.title}</button>)}</div>}
 <div className="workbench-evidence-details"><p>{skill?.description}</p><p><b>验收要求：</b>{skill?.acceptance}</p><div className="skill-manuals">{skill?.manuals?.map((m:Obj)=><a className="button quiet" key={m.kind} href={m.url}>{m.label}<ArrowUpRight size={12}/></a>)}</div>
 <h3>对应权重与验证记录</h3>{weights.length?weights.map(m=><div className="workbench-weight" key={m.id}><b>{m.name} · {m.format}</b><small>{m.selection_basis}</small><small>{m.validation}</small><a href={'#models?path='+encodeURIComponent(m.path)}>查看权重、下载与证据 →</a></div>):<p className="muted">暂无已登记的技能候选权重；不会用其他技能权重替代。</p>}
 <h3>关联任务与方法来源</h3>{skill?.jobs?.length?skill.jobs.map((j:Obj)=><p key={j.id}>{j.name} · {j.state}</p>):<p className="muted">暂无 Studio 关联任务，专用 CLI 记录见手册。</p>}{projects.map((p:Obj)=><a className="source-link" key={p.id} href={p.url+'/tree/'+(p.commit||p.branch)} target="_blank" rel="noreferrer">{p.name} · 固定源码<ArrowUpRight size={12}/></a>)}
 <div className="online-controls"><a className="button primary" href={'#capabilities?skill='+encodeURIComponent(skill?.id||'')}>查看对应能力矩阵<ArrowUpRight size={14}/></a><a className="button secondary" href={'#models?skill='+encodeURIComponent(skill?.id||'')}>全部对应模型</a><a className="button secondary" href={'#simulation?skill='+encodeURIComponent(skill?.id||'')}>在线仿真</a></div>
 </div></section>
}
