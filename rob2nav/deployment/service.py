"""Explicit opt-in device service. Shipped profiles/qualification examples cannot arm it."""
import json
from pathlib import Path
import queue
import sys
import threading
import time
from .types import Skill,SDK_COMMIT
from .runner import UnifiedRuntime
from .calibration import Calibration
from .observer import KinematicObserver
from .unitree import Go2SdkTransport


def preflight(root,profile,record,calibration,model_sha,bundle_sha):
    skill=Skill.load(Path(root)/'deployment/skills'/(profile+'.json'),root)
    if not skill.hardware_qualified:raise ValueError('Bundled policy is not device qualified: '+skill.blocker)
    if record.get('sdk_commit')!=SDK_COMMIT or record.get('runtime_bundle_sha256')!=bundle_sha:raise ValueError('SDK/runtime qualification differs')
    if not calibration.verified or record.get('calibration_sha256')!=calibration.digest():raise ValueError('Calibration not device qualified')
    if record.get('kinematics_sha256')!=model_sha or skill.digest() not in record.get('approved_contracts',[]):raise ValueError('Model/complete policy contract is not approved')
    if record.get('watchdog_verified') is not True or record.get('expires_unix',0)<=time.time():raise ValueError('Independent watchdog/qualification missing or expired')
    return skill


def run(root,profile,interface,sdk_root,calibration_path,qualification_path,model_path,output,duration=3600.,sequence_path=None):
    root=Path(root).resolve()
    if not profile or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for x in profile):raise ValueError('Invalid profile ID')
    # All static gates are evaluated before any ChannelFactoryInitialize call.
    calibration=Calibration.load(Path(calibration_path));record=json.loads(Path(qualification_path).read_text())
    observer=KinematicObserver.from_xml(model_path);transport=Go2SdkTransport(sdk_root,calibration,observer,wire_mode=record.get('wire_mode','hybrid_pd'))
    skill=preflight(root,profile,record,calibration,observer.source_sha256,transport.bundle_digest())
    def approved(s):
        if not s.hardware_qualified or s.digest() not in record['approved_contracts']:raise ValueError('Switch target not in device-qualified contract set')
    qualified_pairs=[]
    approved_contracts=set(record['approved_contracts'])
    for pair in record.get('compact_switch_pairs',[]):
        if (not isinstance(pair,list) or len(pair)!=2 or any(d not in approved_contracts for d in pair)):
            raise ValueError('Compact switch pair must contain two approved contract digests')
        qualified_pairs.append(tuple(pair))
    runtime=UnifiedRuntime(root,hardware=True,approval=approved,qualified_compact_pairs=qualified_pairs);commands=queue.Queue(maxsize=4)
    from .sequence import SkillSequence,load_steps
    choreography=None
    if sequence_path is not None:
        steps=load_steps(sequence_path)
        choreography=SkillSequence(runtime,steps)
        # Check all steps, including the appended final stand, before DDS opens.
        for step in choreography.steps:approved(runtime.profile(step.skill)[1])
    def input_loop():
        for line in sys.stdin:
            try:commands.put_nowait(json.loads(line))
            except (ValueError,queue.Full):pass
    log=None;claimed=False;log_queue=queue.Queue(maxsize=1000);log_thread=None
    try:
        output=Path(output);output.mkdir(parents=True,exist_ok=False);log=(output/'control.jsonl').open('w',buffering=1)
        def logger():
            while True:
                entry=log_queue.get()
                if entry is None:return
                log.write(json.dumps(entry)+'\n')
        log_thread=threading.Thread(target=logger,daemon=True);log_thread.start()
        transport.connect_readonly(interface);deadline=time.monotonic()+5
        while transport.latest is None and time.monotonic()<deadline:time.sleep(.01)
        snapshot=transport.read();runtime.stage_async(profile,snapshot)
        runtime.future.result(timeout=20)  # Preload happens before creating a LowCmd writer.
        transport.claim_writer(record,skill);claimed=True
        now=time.monotonic();runtime.arm(transport.read(),now)
        threading.Thread(target=input_loop,daemon=True).start();start=now;fault_at=None;index=0;last_status=now
        while time.monotonic()-start<duration:
            now=time.monotonic();snapshot=transport.read(ignore_health=runtime.controller.phase=='FAULT')
            if choreography is not None:choreography.tick(snapshot,now)
            while not commands.empty():
                request=commands.get_nowait()
                if not isinstance(request,dict):continue
                if request.get('stop') is True:runtime.controller.latch(now,'operator_stop')
                elif choreography is not None:
                    runtime.controller.event(now,'manual_request_rejected:sequence_owns_commands')
                elif request.get('trigger') is True:
                    try:runtime.controller.trigger()
                    except ValueError as exc:runtime.controller.event(now,'trigger_rejected:'+str(exc))
                elif set(request)=={'velocity'}:
                    try:runtime.controller.set_velocity(request['velocity'],now)
                    except (ValueError,TypeError) as exc:runtime.controller.event(now,'velocity_rejected:'+str(exc))
                elif set(request)=={'switch'}:
                    try:runtime.stage_async(request['switch'],snapshot)
                    except ValueError as exc:runtime.controller.event(now,'request_rejected:'+str(exc))
            command=runtime.step(snapshot,now)
            if now-last_status>=1.:
                print(json.dumps(dict(phase=runtime.controller.phase,active=runtime.controller.active.id if runtime.controller.active else None,
                    pending=runtime.controller.pending.id if runtime.controller.pending else None,
                    stage_error=runtime.stage_error,fault=runtime.controller.fault)),flush=True)
                last_status=now
            context=runtime.controller.pending if runtime.controller.phase in ('WARMUP','BLEND') else runtime.controller.active
            if command.epoch and command.epoch not in transport.allowed_epochs and context is not None:transport.authorize_epoch(command.epoch,context)
            transport.write(command);index+=1
            if index%10==0:
                try:log_queue.put_nowait(dict(time=now,phase=command.phase,epoch=command.epoch,q=snapshot.q.tolist(),dq=snapshot.dq.tolist(),
                    tau=command.tau.tolist(),qref=command.qref.tolist(),kp=command.kp.tolist(),kd=command.kd.tolist(),fault=command.fault))
                except queue.Full:pass  # Diagnostics must not block motor control.
            if runtime.controller.phase=='FAULT':
                if fault_at is None:fault_at=now
                if now-fault_at>.5:break
            delay=.002-(time.monotonic()-now)
            if delay>0:time.sleep(delay)
    finally:
        try:
            if claimed:transport.emergency_damping()
        finally:
            transport.close();runtime.close()
            if log_thread is not None:log_queue.put(None);log_thread.join(timeout=3)
            if log:log.close()
