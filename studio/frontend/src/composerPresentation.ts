export const skillInfo:Record<string,{name:string;description:string}>={
 backflip60:{name:'后空翻',description:'完成一圈并站稳后再继续，可连续添加；当前仅限仿真。'},
 stand:{name:'站稳不动',description:'保持四脚站立，用于开始、结束或中途停留。'},
 walk_moe:{name:'日常行走（推荐）',description:'按设定速度前进或后退，适合与站立串联演示。'},
 walk_mlp:{name:'对照行走（实验）',description:'另一套行走模型，用于研究对比；切换组合尚未通过回归。'},
 stand_soft:{name:'柔和站立（实验）',description:'腿部支撑较柔和；与其他技能的切换组合尚未通过回归。'},
 stand_stiff:{name:'较硬站立（实验）',description:'腿部支撑较硬；与其他技能的切换组合尚未通过回归。'},
 sideflip69:{name:'侧空翻（研究）',description:'完成一次侧空翻后再继续；仍超关节额定速度，仅供仿真研究。'},
};
export function composerSkillName(id:string){return skillInfo[id]?.name||id}
export type TimedStep={seconds:number;trigger:boolean};
export function timingDisplay(p:{total_elapsed_s:number;dwell_elapsed_s:number}){
 const total=Math.round(p.total_elapsed_s*10)/10,dwell=Math.round(p.dwell_elapsed_s*10)/10;
 return {total:total.toFixed(1),dwell:dwell.toFixed(1),extra:(total-dwell).toFixed(1)};
}
export function planTiming(steps:TimedStep[],cycles:number){
 const rounds=Number.isInteger(cycles)&&cycles>=1&&cycles<=4?cycles:1;
 const expanded=Array.from({length:rounds},(_,i)=>i?steps.slice(1):steps).flat();
 return {count:expanded.length,dwell:expanded.reduce((n,s)=>n+(s.trigger?0:Number.isFinite(s.seconds)?s.seconds:0),0),atomic:expanded.filter(s=>s.trigger).length};
}
