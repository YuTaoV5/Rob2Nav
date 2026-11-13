from pathlib import Path
import json,statistics
p=Path(__file__).parent
out={}
for k,v in json.loads((p/'front-trace-control/mujoco-stepping-diagnostic.json').read_text()).items():
 r=v['rows'][0]
 z=r['rear_trace']
 end=z[-1]['time_s']
 span=[a for a in z if a['time_s']>=end-.3]
 out[k]=dict(last=z[-1],peak_front_load_n=[max(a['load_n'][i] for a in span) for i in (0,1)],
              median_front_load_n=[statistics.median(a['load_n'][i] for a in span) for i in (0,1)],
              peak_front_error_rad=[max(abs(a['front_q_rad'][i]-a['front_target_rad'][i]) for a in span) for i in range(6)],
              first_contact=r['failure_detail']['contacts'][0])
(p/'front-trace-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
