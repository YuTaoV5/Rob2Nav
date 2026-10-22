type Obj=Record<string,any>;

const labels:Record<string,{label:string;help:string}>={
 sensors:{label:'感知输入',help:'与训练和评估的观测合约保持一致'},
 scene:{label:'训练场景',help:'随机障碍用于导航泛化训练'},
 steps:{label:'PPO环境步数',help:'实际总步数按完整rollout取整'},
 envs:{label:'并行环境数',help:'导航SB3环境数量'},
 device:{label:'计算设备',help:'当前专用Isaac训练器使用CUDA 0'},
 transfer_motors:{label:'显式电机模型迁移',help:'实验选项；需要重新评估关节速度和接触'},
};

export default function TrainingRecipeFields({recipe,config,update}:{recipe:Obj;config:Obj;update:(key:string,value:any)=>void}){
 const options=config.recipe_options||{};
 const setOption=(key:string,value:any)=>update('recipe_options',{...options,[key]:value});
 const keys=Object.keys(recipe.defaults?.recipe_options||{});
 return <div className="recipe-training-fields">
  <div className="section-header"><div><span className="eyebrow">REPRODUCTION WORKFLOW</span><h2>训练手册与任务步骤</h2></div><span className="badge cyan">专用训练器已接入</span></div>
  <p>选用配方后自动填充推荐值；以下参数仍可调整。执行模块与参数由服务端白名单校验，参考动作限项目内NPZ，训练完成不等于技能通过。</p>
  <ol className="recipe-workflow">{(recipe.workflow||[]).map((step:string)=><li key={step}>{step}</li>)}</ol>
  {recipe.manual_url&&<a className="button secondary" href={recipe.manual_url}>打开完整训练手册 ↗</a>}
  <div className="form-section-title"><h3>任务通用参数</h3></div>
  <div className="form-grid">
   <label className="field span2"><span>实验名称</span><input value={config.name||''} maxLength={80} onChange={e=>update('name',e.target.value)}/></label>
   <label className="field"><span>随机种子 <code>seed</code></span><input type="number" min="0" step="1" value={config.seed} onChange={e=>update('seed',Number(e.target.value))}/></label>
   {recipe.id!=='navigation_ppo'&&<><label className="field"><span>并行环境数 <code>num_envs</code></span><input type="number" min="8" max="4096" step="8" value={config.num_envs} onChange={e=>update('num_envs',Number(e.target.value))}/></label><label className="field"><span>PPO更新轮数 <code>iterations</code></span><input type="number" min="1" max="50000" step="1" value={config.iterations} onChange={e=>update('iterations',Number(e.target.value))}/></label></>}
   {<><label className="field"><span>初始化方式</span><select value={config.initialization} onChange={e=>{update('initialization',e.target.value);update('checkpoint','')}}><option value="base">从基座新训练</option><option value="resume">从检查点继续</option></select></label>{config.initialization==='resume'&&<label className="field span2"><span>项目内检查点路径 <code>checkpoint</code></span><input value={config.checkpoint||''} placeholder={recipe.id==='navigation_ppo'?'outputs/.../navigation.zip':'outputs/.../model_4999.pt'} onChange={e=>update('checkpoint',e.target.value)}/><small>仅允许本项目的 outputs / artifacts / skills 路径；保存前会核验文件存在及格式。</small></label>}</>}
  </div>
  <div className="form-section-title"><h3>技能专用参数</h3></div>
  <div className="form-grid">{keys.map(key=>{const schema=recipe.option_schema?.[key];const meta=schema||labels[key]||{label:key,help:''};const value=options[key]??recipe.defaults.recipe_options[key];return <label className="field" key={key}><span>{meta.label}<code>{key}</code></span>{typeof value==='boolean'?<input type="checkbox" checked={value} onChange={e=>setOption(key,e.target.checked)}/>:schema?.choices?<select value={value} onChange={e=>setOption(key,e.target.value)}>{schema.choices.map((v:string)=><option key={v} value={v}>{v==='front'?'前空翻':v==='back'?'后空翻':v}</option>)}</select>:schema?.type==='string'?<input value={value} onChange={e=>setOption(key,e.target.value)}/>:schema?.choices?<select value={value} onChange={e=>setOption(key,e.target.value)}>{schema.choices.map((v:string)=><option key={v} value={v}>{v==='front'?'前空翻':v==='back'?'后空翻':v}</option>)}</select>:schema?.type==='string'?<input value={value} onChange={e=>setOption(key,e.target.value)}/>:key==='sensors'?<select value={value} onChange={e=>setOption(key,e.target.value)}>{['lidar','depth','both'].map(v=><option key={v} value={v}>{v}</option>)}</select>:key==='scene'?<select value={value} onChange={e=>setOption(key,e.target.value)}>{['empty','random','slalom'].map(v=><option key={v} value={v}>{v}</option>)}</select>:key==='device'?<input value={value} disabled/>:<input type="number" min={schema?.min??1} max={schema?.max} step={schema?.step??1} value={value} onChange={e=>setOption(key,Number(e.target.value))}/>}<small>{meta.help}</small></label>})}</div>
  <details><summary>观测、动作与可扩展配置</summary><p><b>观测：</b>{recipe.observation}</p><p><b>动作：</b>{recipe.action}</p><p>通用任务字段与技能专用参数共同保存在实验配置中。新增技能时，在注册表声明默认值和流程，并在服务端参数白名单中接入训练器。</p></details>
 </div>
}
