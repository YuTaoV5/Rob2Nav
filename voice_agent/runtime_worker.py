"""Measured-state Rob2Nav worker; reuses original qualification/transport contracts.

No --allow-unqualified switch exists. It is never launched by demo mode.
"""
import argparse
import json
from pathlib import Path
import queue
import sys
import threading
import time


def main():
    p=argparse.ArgumentParser()
    for key in ('root','sdk-root','calibration','qualification','model'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--interface',required=True);p.add_argument('--enable-motors',action='store_true')
    p.add_argument('--output',type=Path,required=True,help='New directory for measured motor/control trace')
    args=p.parse_args()
    if not args.enable_motors:raise SystemExit('需要明确 --enable-motors 和原始运行时资格记录；未连接 DDS')
    sys.path.insert(0,str(args.root.resolve()))
    import numpy as np
    from rob2nav.deployment.service import preflight
    from rob2nav.deployment.calibration import Calibration
    from rob2nav.deployment.observer import KinematicObserver
    from rob2nav.deployment.unitree import Go2SdkTransport
    from rob2nav.deployment.runner import UnifiedRuntime
    root=args.root.resolve();calibration=Calibration.load(args.calibration)
    record=json.loads(args.qualification.read_text(encoding='utf-8'));observer=KinematicObserver.from_xml(args.model)
    transport=Go2SdkTransport(args.sdk_root,calibration,observer,wire_mode=record.get('wire_mode','hybrid_pd'))
    skill=preflight(root,'stand',record,calibration,observer.source_sha256,transport.bundle_digest())
    def approved(candidate):
        if not candidate.hardware_qualified or candidate.digest() not in record.get('approved_contracts',[]):
            raise ValueError('Skill contract not device-qualified: '+candidate.id)
    qualified=set(record.get('approved_contracts',[]));pairs=[]
    for pair in record.get('compact_switch_pairs',[]):
        if not isinstance(pair,list) or len(pair)!=2 or any(x not in qualified for x in pair):raise ValueError('Invalid compact transition qualification')
        pairs.append(tuple(pair))
    runtime=UnifiedRuntime(root,hardware=True,approval=approved,qualified_compact_pairs=pairs)
    commands=queue.Queue(maxsize=32);statuses=queue.Queue(maxsize=1);closed=threading.Event();heartbeat=[time.monotonic()]
    def read_commands():
        try:
            for line in sys.stdin:
                try:
                    command=json.loads(line)
                    if not isinstance(command,dict):continue
                    if command=={'heartbeat':True}:heartbeat[0]=time.monotonic();continue
                    commands.put(command,timeout=.1)
                except (ValueError,queue.Full):continue
        finally:closed.set()
    def write_status():
        while True:
            status=statuses.get()
            if status is None:return
            print(json.dumps(status,allow_nan=False),flush=True)
    threading.Thread(target=read_commands,daemon=True).start()
    threading.Thread(target=write_status,daemon=True).start()
    # Preserve the original service's nonblocking measured control evidence.
    args.output.mkdir(parents=True,exist_ok=False)
    trace=(args.output/'control.jsonl').open('w',encoding='utf-8',buffering=1)
    trace_queue=queue.Queue(maxsize=1000)
    def logger():
        while True:
            entry=trace_queue.get()
            if entry is None:return
            trace.write(json.dumps(entry,allow_nan=False)+'\n')
    trace_thread=threading.Thread(target=logger,daemon=True);trace_thread.start()
    claimed=False;fault_at=None;index=0
    try:
        transport.connect_readonly(args.interface);deadline=time.monotonic()+5
        while transport.latest is None and time.monotonic()<deadline:time.sleep(.01)
        snapshot=transport.read();runtime.stage_async('stand',snapshot);runtime.future.result(timeout=20)
        transport.claim_writer(record,skill);claimed=True
        runtime.arm(transport.read(),time.monotonic());last_status=0
        while True:
            now=time.monotonic();snapshot=transport.read(ignore_health=runtime.controller.phase=='FAULT')
            if closed.is_set() or now-heartbeat[0]>.5:runtime.controller.latch(now,'voice_bridge_heartbeat_lost')
            while not commands.empty():
                request=commands.get_nowait()
                try:
                    if request=={'stop':True}:runtime.controller.latch(now,'voice_emergency_stop')
                    elif set(request)=={'switch'}:runtime.stage_async(request['switch'],snapshot)
                    elif request=={'trigger':True}:runtime.controller.trigger()
                    elif set(request)=={'velocity'}:runtime.controller.set_velocity(request['velocity'],now)
                    else:raise ValueError('Unknown runtime command')
                except (ValueError,TypeError) as exc:runtime.controller.latch(now,'voice_command_rejected:'+str(exc))
            command=runtime.step(snapshot,now)
            context=runtime.controller.pending if runtime.controller.phase in ('WARMUP','BLEND') else runtime.controller.active
            if command.epoch and command.epoch not in transport.allowed_epochs and context is not None:transport.authorize_epoch(command.epoch,context)
            transport.write(command)
            index+=1
            if index%10==0:
                try:trace_queue.put_nowait(dict(time=now,phase=command.phase,epoch=command.epoch,
                    q=snapshot.q.tolist(),dq=snapshot.dq.tolist(),tau=command.tau.tolist(),
                    qref=command.qref.tolist(),kp=command.kp.tolist(),kd=command.kd.tolist(),fault=command.fault))
                except queue.Full:pass
            if now-last_status>=.05:
                velocity_known=snapshot.velocity is not None
                status=dict(phase=runtime.controller.phase,active=runtime.controller.active.id if runtime.controller.active else None,
                    pending=runtime.controller.pending.id if runtime.controller.pending else None,
                    loading=runtime.future is not None or runtime.queued is not None,
                    fault=runtime.controller.fault,stage_error=runtime.stage_error,
                    grounded=bool(np.all(snapshot.contacts())),
                    stationary=bool(velocity_known and snapshot.stable()),velocity_available=velocity_known)
                try:statuses.put_nowait(status)
                except queue.Full:pass
                last_status=now
            if runtime.controller.phase=='FAULT':
                if fault_at is None:fault_at=now
                if now-fault_at>.5:break
            time.sleep(max(0,.002-(time.monotonic()-now)))
    finally:
        try:
            if claimed:transport.emergency_damping()
        finally:
            transport.close();runtime.close()
            try:trace_queue.put_nowait(None)
            except queue.Full:pass
            trace_thread.join(timeout=3)
            if not trace_thread.is_alive():trace.close()


if __name__=='__main__':main()
