import {useState} from 'react';
import './simulationTelemetry.css';

export type PhysicalFrame={
  time:number;
  imu?:{quaternion_wxyz:number[];gyro_rad_s:number[];linear_velocity_m_s:number[];projected_gravity:number[]};
  joints?:{name:string;position_rad:number;velocity_rad_s:number;actuator_effort_nm:number}[];
  sensors?:{name:string;unit:string;value:number[]}[];
};

export function physicalSignals(frame:PhysicalFrame){
  const values:Record<string,number>={};
  for(const [label,series,unit] of [
    ['陀螺仪',frame.imu?.gyro_rad_s,'rad/s'],
    ['机体系速度',frame.imu?.linear_velocity_m_s,'m/s'],
    ['投影重力',frame.imu?.projected_gravity,''],
  ] as const){
    series?.forEach((v,i)=>{values[[label,'XYZ'[i],unit].filter(Boolean).join(' ')]=v});
  }
  for(const joint of frame.joints??[]){
    values[joint.name+' 位置 rad']=joint.position_rad;
    values[joint.name+' 速度 rad/s']=joint.velocity_rad_s;
    values[joint.name+' 力矩 Nm']=joint.actuator_effort_nm;
  }
  for(const sensor of frame.sensors??[])sensor.value.forEach((v,i)=>{
    values[sensor.name+' '+i+' '+sensor.unit]=v;
  });
  return values;
}

export const metric=(n:number|undefined)=>Number.isFinite(n)?n!.toFixed(3):'—';

export function PhysicalReadout({frame}:{frame:PhysicalFrame}){
  const imu=frame.imu;
  return <div className="sim-physical">
    {imu?.quaternion_wxyz&&<div className="sim-signal-grid">
      <div><small>机身姿态 · 四元数 WXYZ</small><strong>{imu.quaternion_wxyz.map(metric).join(' / ')}</strong></div>
      <div><small>陀螺仪 · rad/s · XYZ</small><strong>{imu.gyro_rad_s.map(metric).join(' / ')}</strong></div>
      <div><small>机体系速度 · m/s · XYZ</small><strong>{imu.linear_velocity_m_s.map(metric).join(' / ')}</strong></div>
      <div><small>机体系投影重力 · XYZ</small><strong>{imu.projected_gravity.map(metric).join(' / ')}</strong></div>
    </div>}
    {!!frame.sensors?.length&&<div className="sim-signal-grid">{frame.sensors.map(s=><div key={s.name}>
      <small>{s.name} · {s.unit}</small><strong>{s.value.map(metric).join(' / ')}</strong>
    </div>)}</div>}
    {!!frame.joints?.length&&<details><summary>关节数据 · {frame.joints.length} 轴</summary>
      <div className="sim-joint-scroll"><table><thead><tr><th>关节</th><th>位置 rad</th><th>速度 rad/s</th><th>执行器力矩 Nm</th></tr></thead>
      <tbody>{frame.joints.map(j=><tr key={j.name}><th scope="row">{j.name}</th><td>{metric(j.position_rad)}</td><td>{metric(j.velocity_rad_s)}</td><td>{metric(j.actuator_effort_nm)}</td></tr>)}</tbody></table></div>
    </details>}
  </div>;
}

export default function SimulationTelemetry({frames}:{frames:PhysicalFrame[]}){
  const [selection,setSelection]=useState('');
  const latest=frames.at(-1);
  if(!latest)return null;
  const keys=Object.keys(physicalSignals(latest));
  const selected=keys.includes(selection)?selection:keys[0];
  const points=frames.flatMap(f=>{const value=physicalSignals(f)[selected];return Number.isFinite(value)?[[f.time,value]]:[]});
  const t0=points[0]?.[0]??0;
  const t1=points.at(-1)?.[0]??t0;
  const lo=Math.min(...points.map(p=>p[1])),hi=Math.max(...points.map(p=>p[1]));
  return <section className="sim-inspector">
    <header><h3>实时物理量</h3><span>{metric(latest.time)} s · MuJoCo</span></header>
    <PhysicalReadout frame={latest}/>
    {selected&&<div className="sim-live-chart">
      <label>监测量 <select aria-label="实时监测量" value={selected} onChange={e=>setSelection(e.target.value)}>{keys.map(k=><option key={k}>{k}</option>)}</select></label>
      <svg viewBox="0 0 600 115" role="img" aria-label={selected+' 实时曲线'}>
        <path d="M44 8V94H590" fill="none" stroke="#9ab0b8"/>
        <text x="0" y="14">{metric(hi)}</text><text x="0" y="96">{metric(lo)}</text>
        <text x="44" y="110">{t0.toFixed(2)} s</text><text x="590" y="110" textAnchor="end">{t1.toFixed(2)} s</text>
        {points.length>1&&<polyline fill="none" stroke="#167c80" strokeWidth="2" points={points.map(([t,v])=>[(t-t0)/(t1-t0||1)*546+44,90-(v-lo)/(hi-lo||1)*78].join(',')).join(' ')}/>}
      </svg>
    </div>}
  </section>;
}
