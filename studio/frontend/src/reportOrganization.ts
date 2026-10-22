export type Report={id:string;name:string;path:string;aliases?:string[]};
export const reportCategories=[
 {id:'start',name:'入门与工作台',eyebrow:'GET STARTED',description:'先了解环境，再开始第一个实验。'},
 {id:'training',name:'训练与复现',eyebrow:'TRAIN & REPRODUCE',description:'训练方案、奖励设计与技能复现手册。'},
 {id:'simulation',name:'仿真与导航',eyebrow:'SIMULATE & EXPLORE',description:'运行技能、构建地形与验证导航任务。'},
 {id:'deployment',name:'部署与硬件',eyebrow:'DEPLOY & CONNECT',description:'机器人连接、传感器安装与技能热切换。'},
 {id:'validation',name:'验证与研究',eyebrow:'VERIFY & LEARN',description:'验收记录、失败诊断与源码审计。'},
 {id:'project',name:'项目与交接',eyebrow:'PROJECT NOTES',description:'项目进展、资源目录、发布与交接。'},
] as const;
const startPaths=new Set(['README.md','studio/README.md','docs/新手复现总览.md','docs/环境准备与路径约定.md','docs/术语与验收说明.md','docs/项目使用说明.md','docs/创建实验使用说明.md']);
// This is a presentation index only. Files, aliases, titles and contents are not changed.
export function reportCategory(report:Report){
 const p=report.path;
 if(startPaths.has(p))return 'start';
 if(/交接|发布|打包|数据目录|迁移|功能规划|交付|THIRD_PARTY|剩余任务/.test(p))return 'project';
 if(/记录|审计|审查|诊断|验收|研究目标|验证|研究/.test(p))return 'validation';
 if(/部署|标定|安装|热切换|SDK|状态机|合约|手掌识别/.test(p))return 'deployment';
 if(/训练|续训|实验|复现|步态指标|AME方法参考|步态初始化/.test(p))return 'training';
 if(/仿真|导航|地形|技能库|权重工作流/.test(p))return 'simulation';
 return 'project';
}
export const recommendedReports=[
 {path:'docs/新手复现总览.md',label:'第一次使用',title:'从这里开始',description:'环境、技能与复现路径，一次建立全局认识。'},
 {path:'docs/在线仿真使用说明.md',label:'动手体验',title:'运行一次仿真',description:'选择策略、观察动作，再查看任务遥测。'},
 {path:'项目交接记录.md',label:'项目进展',title:'了解当前进展',description:'查阅已完成工作、已知限制与下一步任务。'},
];
export function reportDisplayTitle(report:Report){
 const titles:Record<string,string>={'README.md':'Rob2Nav 项目总览','studio/README.md':'Studio 工作台使用指南','THIRD_PARTY_NOTICES.md':'第三方项目与许可','docs/project_audits/导航源码审计.md':'导航源码审计','docs/project_audits/技能源码审计.md':'技能源码审计','docs/project_audits/核心能力源码审计.md':'核心能力源码审计','docs/步态指标说明.md':'步态指标与首回合诊断','docs/工作台能力审计.md':'工作台能力审计','docs/统一能力与源码合约.md':'统一能力与源码合约','docs/部署框架与导航指南.md':'部署框架与导航指南','docs/部署框架验证记录.md':'部署框架验证记录','早期验证记录.md':'早期远程仿真验证记录','项目交接记录.md':'项目交接与进展记录'};
 return titles[report.path]||report.name;
}
export function reportKind(report:Report){return /手册|指南|说明/.test(report.path)?'操作指南':/审计|审查/.test(report.path)?'源码审计':/记录|验证|诊断/.test(report.path)?'研究记录':'项目文档'}
export function reportLink(report:Report){return '#reports?path='+encodeURIComponent(report.path)}
export function sortReports(reports:Report[]){return [...reports].sort((a,b)=>{
 const priority=[...recommendedReports.map(r=>r.path),'README.md','docs/环境准备与路径约定.md','docs/术语与验收说明.md','docs/实验训练方案说明.md','docs/感知强化学习训练说明.md','docs/MoE感知分支训练说明.md','docs/统一技能库使用说明.md','docs/实时技能仿真说明.md','skills/navigation/SCAN仿真手册.md','docs/统一部署框架与状态机.md','docs/D435i与MID360安装标定手册.md','docs/Go2Edu导航与机载连接指南.md','docs/技能模型验收清单.md','docs/空翻与直立研究目标和证据.md','docs/工作台能力审计.md','docs/能力复现交付状态.md','docs/版本发布记录.md','docs/模型与数据目录.md'];
 const rank=(r:Report)=>priority.indexOf(r.path);
 const ar=rank(a),br=rank(b);if(ar>=0||br>=0)return (ar<0?99:ar)-(br<0?99:br);
 return reportDisplayTitle(a).localeCompare(reportDisplayTitle(b),'zh-CN');
})}
