import {useEffect, useState} from 'react';

const skills = {rolling: '滚动', stepping: '踏步', recovery: '起身'} as const;
type Skill = keyof typeof skills;
type Run = {id: string; recipe: Skill; smoke: boolean; entrypoint: string;
  status?: {state: string; error?: string}};
const states: Record<string, string> = {queued: '排队中', training: '训练中', trained: '训练结束', failed: '失败'};
const api = '/api/as2w';
const steps = [
  ['任务与权重', '选择技能和对应的起始检查点'],
  ['训练配置', '选择冒烟或正式训练'],
  ['核对与提交', '确认参数后提交任务'],
] as const;
const recipes: Record<Skill, {checkpoint: string; envs: number; seed: number}> = {
  rolling: {checkpoint: 'rolling-maneuvers-formal/model_1199.pt', envs: 384, seed: 303},
  stepping: {checkpoint: 'stepping-go2-reset-exploration-init/model.pt', envs: 256, seed: 305},
  recovery: {checkpoint: 'getup-symmetric-formal/model_1199.pt', envs: 384, seed: 301},
};

async function loadRuns(): Promise<Run[]> {
  const response = await fetch(api + '/runs');
  if (!response.ok) throw new Error('无法读取训练任务');
  const rows: Run[] = await response.json();
  return rows.filter(row => row.entrypoint === 'training/as2w-backup/train.sh');
}

export default function FormalAs2wTraining() {
  const [skill, setSkill] = useState<Skill>('rolling');
  const [smoke, setSmoke] = useState(true);
  const [step, setStep] = useState(0);
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [submitted, setSubmitted] = useState('');

  useEffect(() => {
    let active = true;
    const refresh = () => {
      loadRuns().then(rows => { if (active) setRuns(rows); })
        .catch(() => { if (active) setError('无法刷新训练任务列表'); });
    };
    refresh();
    const timer = window.setInterval(refresh, 5000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  async function submit() {
    setBusy(true);
    setError('');
    setSubmitted('');
    try {
      const response = await fetch(api + '/formal-runs', {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'X-Studio-Request': '1'},
        body: JSON.stringify({skill, smoke}),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : '提交训练失败');
      setSubmitted(result.id);
      setRuns(await loadRuns());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const recipe = recipes[skill];
  return <section className="as2w-section" aria-label="AS2-W 训练配方">
    <h2>AS2-W 训练配方</h2>
    <nav className="experiment-steps" aria-label="AS2-W 创建实验步骤">
      {steps.map(([label, description], index) => <button type="button" key={label}
        className="experiment-step-card" aria-current={step === index ? 'step' : undefined}
        disabled={busy} onClick={() => setStep(index)}>
        <span className="experiment-step-number" aria-hidden="true">{index + 1}</span>
        <span><strong>{label}</strong><small>{description}</small></span>
      </button>)}
    </nav>
    <div className="card">
      {step === 0 && <>
        <label className="field">技能 <select value={skill} onChange={event => setSkill(event.target.value as Skill)}>
          {Object.entries(skills).map(([id, name]) => <option key={id} value={id}>{name}</option>)}
        </select></label>
        <p>起始检查点 <code>{recipe.checkpoint}</code></p>
      </>}
      {step === 1 && <>
        <label className="field">运行方式 <select value={smoke ? 'smoke' : 'formal'}
          onChange={event => setSmoke(event.target.value === 'smoke')}>
          <option value="smoke">冒烟检查 · 3 轮</option>
          <option value="formal">正式训练 · 1200 轮</option>
        </select></label>
        <p>本次使用 {smoke ? 8 : recipe.envs} 个并行环境，随机种子 {recipe.seed}。奖励与课程沿用该技能的正式配方。</p>
      </>}
      {step === 2 && <>
        <h3>核对训练配置</h3>
        <dl>
          <dt>技能</dt><dd>{skills[skill]}</dd>
          <dt>起始检查点</dt><dd><code>{recipe.checkpoint}</code></dd>
          <dt>运行配置</dt><dd>{smoke ? '3 轮冒烟 · 8 个环境' : `1200 轮正式训练 · ${recipe.envs} 个环境`}</dd>
          <dt>随机种子</dt><dd>{recipe.seed}</dd>
        </dl>
      </>}
      <div className="experiment-next">
        {step > 0 && <button type="button" className="button secondary" disabled={busy}
          onClick={() => setStep(step - 1)}>上一步</button>}
        {step < 2 && <button type="button" className="button primary"
          onClick={() => setStep(step + 1)}>下一步</button>}
        {step === 2 && <button type="button" className="button primary" disabled={busy}
          onClick={submit}>{busy ? '正在提交' : '提交训练'}</button>}
      </div>
    </div>
    {submitted && <p role="status">已提交任务 {submitted}</p>}
    {error && <p role="alert">{error}</p>}
    {runs.length > 0 && <ul>{runs.map(run => <li key={run.id}>
      {skills[run.recipe]} · {run.smoke ? '冒烟' : '正式训练'} · {states[run.status?.state ?? ''] ?? '等待状态'}
      {' · '}<a href={api + '/formal-runs/' + run.id + '/log'} target="_blank" rel="noreferrer">查看日志</a>
      {run.status?.error && <div>{run.status.error}</div>}
    </li>)}</ul>}
  </section>;
}
