export default function VoiceControl(){
 return <section>
  <div className="page-heading"><div><div className="eyebrow">VOICE & NAVIGATION</div><h1>语音任务控制</h1><p>地图标注、导航和技能共用一个状态机。界面显示当前 demo / live 模式与实际运行资格。</p></div><a className="button" href="/voice/" target="_blank" rel="noreferrer">独立打开</a></div>
  <details className="card"><summary>连接与启动</summary><p>在 Rob2Nav 根目录启动：<code>python -m voice_agent --mode demo --speech-config configs/speech.local.json</code>。首次先按 docs/voice-control.md 安装并下载语音模型。实机需要 ROS 桥、地图配准和原有硬件资格；界面不会自动启动电机。</p></details>
  <iframe title="语音导航与技能控制" src="/voice/" allow="microphone; autoplay" style={{width:'100%',height:1100,border:0,borderRadius:16,background:'#f3f6fa'}}/>
 </section>
}
