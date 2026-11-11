import {useEffect, useState} from 'react';

const skills = {rolling: '滚动', stepping: '踏步', recovery: '起身'} as const;
type Skill = keyof typeof skills;
type Run = {id: string; recipe: Skill; smoke: boolean; entrypoint: string;
  status?: {state: string; error?: string}};
const states: Record<string, string> = {queued: '排队中', training: '训练中', trained: '训练结束', failed: '失败'};
const api = '/api/as2w';

async function loadRuns(): Promise<Run[]> {
  const response = await fetch(api + '/runs');
  if (!response.ok) throw new Error('无法读取训练任务');
  const rows: Run[] = await response.json();
  return rows.filter(row => row.entrypoint === 'training/as2w-backup/train.sh');
}

export default function FormalAs2wTraining() {
  const [skill, setSkill] = useState<Skill>('rolling');
  const [smoke, setSmoke] = useState(true);
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

  return <section className="card as2w-section">
    <h2>AS2-W 训练配方</h2>
    <label>技能 <select value={skill} disabled={busy} onChange={event => setSkill(event.target.value as Skill)}>
      {Object.entries(skills).map(([id, name]) => <option key={id} value={id}>{name}</option>)}
    </select></label>
    <label>运行方式 <select value={smoke ? 'smoke' : 'formal'} disabled={busy}
      onChange={event => setSmoke(event.target.value === 'smoke')}>
      <option value="smoke">冒烟检查 · 3 轮</option>
      <option value="formal">正式训练 · 1200 轮</option>
    </select></label>
    <button type="button" disabled={busy} onClick={submit}>{busy ? '正在提交' : '提交训练'}</button>
    {submitted && <p role="status">已提交任务 {submitted}</p>}
    {error && <p role="alert">{error}</p>}
    {runs.length > 0 && <ul>{runs.map(run => <li key={run.id}>
      {skills[run.recipe]} · {run.smoke ? '冒烟' : '正式训练'} · {states[run.status?.state ?? ''] ?? '等待状态'}
      {' · '}<a href={api + '/formal-runs/' + run.id + '/log'} target="_blank" rel="noreferrer">查看日志</a>
      {run.status?.error && <div>{run.status.error}</div>}
    </li>)}</ul>}
  </section>;
}
