import {useState} from 'react';
import './SuperMimicDemos.css';
const demos=[
 {"id": "expression_actorlow_v4_model800_playful_dance", "title": "节律舞动 · SuperMimic学生800轮（待验收）", "detail": "12秒原速未剪辑，教师关闭的自训800轮学生；独立四种子功能检查4/4。节律编排研究演示，待人工完整验收；不代表情绪感知或实机资格。", "mp4Only": true},
 {"id": "expression_actorlow_v4_model800_slow_sway", "title": "缓慢摆动 · SuperMimic学生800轮（待验收）", "detail": "12秒原速未剪辑，教师关闭的自训800轮学生；独立四种子功能检查4/4。节律编排研究演示，待人工完整验收；不代表情绪感知或实机资格。", "mp4Only": true},
 {"id": "expression_actorlow_v4_model800_joyful_steps", "title": "轻快踏步 · SuperMimic学生800轮（待验收）", "detail": "12秒原速未剪辑，教师关闭的自训800轮学生；独立四种子功能检查4/4。节律编排研究演示，待人工完整验收；不代表情绪感知或实机资格。", "mp4Only": true},
 {"id": "expression_actorlow_v4_model800_firm_march", "title": "节律行进 · SuperMimic学生800轮（待验收）", "detail": "12秒原速未剪辑，教师关闭的自训800轮学生；独立四种子功能检查4/4。节律编排研究演示，待人工完整验收；不代表情绪感知或实机资格。", "mp4Only": true},

 {"id": "recovery_student_v1_model4000", "title": "自恢复 · SuperMimic学生第4000轮", "detail": "12秒原速，教师已关闭；独立四种子恢复并稳定2秒 365/512。关节约束另报，研究候选、未获实机资格。"},
 {id:"velocity_moe_continuity_tradeoff_v3_model800",title:"MoE学生跟踪权衡对照 · 第800轮",detail:"32秒原速：站立、前后行走、左右侧移、双向转向与停步。步态和指令跟踪对照。原始目标变化率仍未全部通过；研究候选，完整原速人工复看及实机资格未完成。"},
 {id:"velocity_moe_continuity_v2_model800",title:"MoE学生连续性优化 · 第800轮",detail:"32秒原速：站立、前后行走、左右侧移、双向转向与停步。步态和指令跟踪对照。研究候选，完整原速人工复看及实机资格未完成。"},
 {id:"velocity_gait_moe_dataonly_v1_model1200",title:"MoE教师仅采样 · SuperMimic学生第1200轮",detail:"32秒原速：站立、前后行走、左右侧移、双向转向与停步。步态和指令跟踪对照。原始目标变化率仍未全部通过；研究候选，完整原速人工复看及实机资格未完成。"},
 {id:"velocity_gait_moe_teacher_v1_model1200",title:"MoE教师辅助 · SuperMimic学生第1200轮",detail:"32秒原速：站立、前后行走、左右侧移、双向转向与停步。步态和指令跟踪对照。原始目标变化率仍未全部通过；研究候选，完整原速人工复看及实机资格未完成。"},
 {id:"velocity_gait_v3_model800",title:"目标连续性优化V3 · 续训第800轮",detail:"32秒原速：站立、前后行走、左右侧移、双向转向与停步。步态和指令跟踪对照。原始目标变化率仍未全部通过；研究候选，完整原速人工复看及实机资格未完成。"},
 {id:"velocity_gait_v2_model2000",title:"步态优化 V2 · 续训第2,000轮",detail:"32秒原速：站立、前后行走、左右侧移、双向转向与停步。步态和指令跟踪对照。原始目标变化率仍未全部通过；研究候选，完整原速人工复看及实机资格未完成。"},
 {id:'velocity_model12000_hold',title:'最新速度行走 · 第12,000轮',detail:'32秒原速：站立、前后行走、左右侧移、双向转向和停步。使用控制目标初始化修复；独立评估仍有目标变化率超标，尚未完整验收。'} ,
 {id:'velocity_model500',title:'速度行走 · 第500轮',detail:'15秒原速记录：站立指令 → 前进0.5m/s → 停步指令。早期检查点，尚未通过行走验收。'}
];
function Demo({demo}:{demo:typeof demos[number]}){
 const [failed,setFailed]=useState(false);
 const base='/api/files/artifacts/supermimic/'+demo.id;
 return <article className="supermimic-demo"><h3>{demo.title}</h3>
 <video controls playsInline preload="metadata" poster={'mp4Only' in demo?undefined:base+'.jpg'} aria-label={demo.title+'仿真视频'} onError={()=>setFailed(true)}>
 {!('mp4Only' in demo)&&<source src={base+'.webm'} type="video/webm"/>}<source src={base+'.mp4'} type="video/mp4"/>
 当前浏览器无法播放，请使用下方下载链接。
 </video><p>{demo.detail}</p>
 {failed&&<p role="alert">播放器未能加载，请下载视频后观看，或刷新页面重试。</p>}
 <div className="supermimic-demo-links">{!('mp4Only' in demo)&&<a href={base+'.webm'} download>下载WebM</a>}<a href={base+'.mp4'} download>下载MP4</a></div>
 </article>;
}
export default function SuperMimicDemos(){return <section className="card supermimic-demos" aria-label="SuperMimic仿真演示">
 <div className="section-header"><div><span className="eyebrow">SUPERMIMIC · MUJOCO</span><h2>SuperMimic 动作演示</h2></div><span className="badge amber">研究演示 · 待验收</span></div>
 <a href="/supermimic.html" aria-label="更多SuperMimic行走演示视频">更多（全部 {demos.length} 个视频） ↗</a><p>点击视频播放，可拖动进度或全屏查看。这里是已录制的真实物理仿真，实时交互仿真接口尚未接入。</p>
 <div className="supermimic-demo-grid">{demos.slice(0,2).map(d=><Demo key={d.id} demo={d}/>)}</div>
 </section>;}
