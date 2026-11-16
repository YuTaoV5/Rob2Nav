#!/usr/bin/env python3
"""WIP / DO NOT USE FOR ACCEPTANCE -- per-frame trace recorder for the R4000 sanction evaluation.

STATUS (2026-11-17, still WIP): the import-hook design works (the evaluator is found and patched:
AS2W_TRACE_EVALUATOR_FOUND -> PATCHED), and the evaluator DOES call ``wrapped.step(actor(obs))`` at line 79 of
eval_as2w_isaac_speed_range.py, so a wrapper on that attribute should see every frame. Two attempts still
recorded 0 frames. The first failure was a Warp device-side assert caused by wrapping the articulation's
set_joint_velocity_target setter (that hook has been removed); the second attempt traced_evaluate entered and
exited early with trace_frames=[] and no AS2W_TRACE console line, so the remaining failure is before the first
recorded step and has not been isolated yet. Until then the metrics this file exists for - first-second /
3-5 s / last-3 s yaw split, longest continuous unload, worst per-axis leg target step, leg and wheel velocity
RMS - stay marked as pending, and the headline start/A/B comparison uses only ``scripts/stage2_compare.py`` on
the sanctioned evaluation JSON. A caller must pass --i-understand-this-is-wip or the script refuses to run.

Original design note follows.

Per-frame trace recorder for the R4000 sanction evaluation, without changing the sanctioned scorer.

Why it exists: the operator's protocol needs metrics the sanctioned evaluation only reports in aggregate -
signed yaw in the first second, at 3-5 s and in the last 3 s, the dispatched versus measured wheel
differential, four-wheel support, the longest CONTINUOUS unload, the worst per-axis leg target step and the
leg/wheel velocity RMS.  Re-implementing the scenarios would put a second control path next to the frozen
one, so instead this script:

  1. imports the R4000 run snapshot modules (the frozen entry, the frozen evaluator),
  2. wraps the evaluator with a recorder that watches ``wrapped.step`` and the articulation target setters,
  3. runs the frozen entry and the frozen evaluator UNCHANGED, so scenarios, initial states, seeds and the
     scoring are exactly the sanctioned ones,
  4. post-processes the recorded frames into the windowed metrics and prints them next to the aggregate
     numbers so a mismatch between the two would be visible immediately.

Frames are grouped into cases by the command actually in force (|vx|, |wz|) and by the episode boundary, so
no knowledge of the evaluator's internal case list is required.

Run (same arguments as the R4000 evaluation entry):
    AS2W_R4000_SNAPSHOT=<run>/source python scripts/stage2_trace.py \
        --usd ... --urdf ... --profile expanded --speed-cap .20 ... \
        --num-envs 24 --iterations 1 --out <dir> --checkpoint <ckpt> --evaluate --seed 7393
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

FAILS = []


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def classify(vx, wz, eps_v=0.02, eps_w=0.02):
    if abs(vx) < eps_v and abs(wz) < eps_w:
        return "hold", 0
    if abs(vx) < eps_v:
        return ("turn_left" if wz > 0 else "turn_right"), (1 if wz > 0 else -1)
    sign = 1 if wz >= 0 else -1
    if abs(wz) < eps_w:
        return ("straight_fwd" if vx > 0 else "straight_bwd"), (1 if vx > 0 else -1)
    return (("arc_left" if sign > 0 else "arc_right") if vx > 0 else
            ("arc_bwd_left" if sign > 0 else "arc_bwd_right")), sign


def analyse(frames, dt, tail_s=3.0):
    """Group frames into cases and reduce to the metrics the protocol asks for."""
    if not frames:
        return {}
    cases = {}
    cur_key, cur = None, []
    prev_t = None
    for f in frames:
        cls, sign = classify(f["cmd"][0], f["cmd"][2])
        new_case = (cls != cur_key) or (prev_t is not None and f["t"] <= prev_t) or (f.get("reset"))
        if new_case and cur:
            cases.setdefault(cur_key, []).append(cur)
            cur = []
        cur_key = cls
        cur.append(f)
        prev_t = f["t"]
    if cur:
        cases.setdefault(cur_key, []).append(cur)

    out = {}
    for cls, runs in cases.items():
        merged = []
        for r in runs:
            dur = r[-1]["t"]
            n1 = int(round(1.0 / dt))
            tail_start = max(0.0, dur - tail_s)
            win = {"first_1s": [f for f in r if f["t"] < 1.0],
                   "mid_3_5s": [f for f in r if 3.0 <= f["t"] <= 5.0],
                   "last_3s": [f for f in r if f["t"] >= tail_start]}
            rec = {"duration_s": round(dur, 2), "n_frames": len(r)}
            for wname, ws in win.items():
                if not ws:
                    rec[wname] = None
                    continue
                n = len(ws)
                yaw = sum(f["wz"] for f in ws) / n
                diff_d = [f["diff_dispatched"] for f in ws if f.get("diff_dispatched") is not None]
                diff_m = [f["diff_measured"] for f in ws if f.get("diff_measured") is not None]
                sup = sum(1 for f in ws if f["loaded"] >= 4) / n
                rec[wname] = {
                    "yaw_signed_mean": round(yaw, 5),
                    "yaw_abs_mean": round(sum(abs(f["wz"]) for f in ws) / n, 5),
                    "diff_dispatched_mean": round(sum(diff_d) / len(diff_d), 4) if diff_d else None,
                    "diff_measured_mean": round(sum(diff_m) / len(diff_m), 4) if diff_m else None,
                    "diff_delivery": (round(abs(sum(diff_m) / len(diff_m)) / abs(sum(diff_d) / len(diff_d)), 4)
                                      if diff_d and diff_m and abs(sum(diff_d) / len(diff_d)) > 1e-6 else None),
                    "support_frac": round(sup, 4),
                    "leg_target_step_max_rad": round(max(f["leg_step_max"] for f in ws), 6),
                }
            # continuous unload over the whole case, not just the tail
            best = run = 0
            for f in r:
                run = run + 1 if f["loaded"] < 4 else 0
                best = max(best, run)
            rec["longest_continuous_unload_s"] = round(best * dt, 3)
            rec["leg_dq_rms_rad_s"] = round((sum(f["leg_dq_sq"] for f in r) / len(r)) ** 0.5, 4)
            rec["wheel_dq_rms_rad_s"] = round((sum(f["wheel_dq_sq"] for f in r) / len(r)) ** 0.5, 4)
            rec["nonwheel_contact_frames"] = sum(1 for f in r if f["nonwheel_contact"])
            rec["terminated"] = any(f["term"] for f in r)
            merged.append(rec)
        out[cls] = merged
    return out


def main() -> int:
    if "--i-understand-this-is-wip" not in sys.argv:
        print("REFUSED: stage2_trace.py is WIP (verified attempt captured 0 frames because the sanctioned "
              "evaluator does not call wrapped.step). Pass --i-understand-this-is-wip only for debugging.",
              file=sys.stderr)
        return 3
    sys.argv = [a for a in sys.argv if a != "--i-understand-this-is-wip"]
    snapshot = os.environ.get("AS2W_R4000_SNAPSHOT", "")
    if not snapshot or not os.path.isdir(snapshot):
        print("ERR: AS2W_R4000_SNAPSHOT must point at the R4000 run snapshot source", file=sys.stderr)
        return 2
    for p in (snapshot, os.path.dirname(os.path.abspath(__file__))):
        if p not in sys.path:
            sys.path.insert(0, p)

    # recorder state, filled by the wrapper installed below
    state = {"frames": [], "case": None, "dispatched_wheel": None, "errors": []}

    # The evaluator must NOT be imported here: importing it (and the as2w modules it pulls) before Kit has
    # started fails with libanimationSchema.so / libusd_tf.so undefined-symbol errors.  The hook below patches
    # it right after the frozen entry imports it, i.e. after launch_simulation.
    state["orig_eval"] = None

    def traced_evaluate(env, wrapped, actor, args):
        import torch

        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        dt = env.step_dt
        wheel_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
        body_names = list(robot.body_names)
        wheel_body = [body_names.index(n) for n in wheel_names]
        sensor_names = [str(n) for n in getattr(sensor, "body_names", [])]
        sensor_wheel = [sensor_names.index(n) for n in wheel_names if n in sensor_names]
        sensor_nonwheel = [i for i, n in enumerate(sensor_names)
                           if not any(w in n for w in wheel_names) and "base_link" not in n]
        sensor_base = [i for i, n in enumerate(sensor_names) if "base_link" in n]
        joint_names = list(robot.joint_names)
        wheel_joints = [joint_names.index(n) for n in wheel_names]
        leg_joints = [i for i in range(len(joint_names)) if i not in wheel_joints]
        side = torch.tensor([1.0, -1.0, 1.0, -1.0], device=env.device)
        # NOTE: wrapping the articulation's target setters breaks the dispatch path (Warp device-side assert
        # before the first step), so the dispatched target is NOT intercepted here.  Instead the action the
        # evaluator hands to step is recorded, and the measured differential comes from joint velocities.  The
        # authoritative dispatched differential is the evaluator's own per-row value, which
        # scripts/stage2_compare.py already reports.
        side = torch.tensor([1.0, -1.0, 1.0, -1.0], device=env.device)

        prev_leg_target = {"v": None}
        original_step = wrapped.step

        def recorded_step(actions):
            obs, rew, don, info = original_step(actions)
            try:
                with torch.no_grad():
                    cmd = env.command_manager.get_command("base_velocity")[0].detach().cpu().numpy()
                    vxy = robot.data.root_lin_vel_b.torch[0].detach().cpu().numpy()
                    wz = float(robot.data.root_ang_vel_b.torch[0, 2])
                    dq = robot.data.joint_vel.torch[0].detach().cpu().numpy()
                    try:
                        tgt = robot.data.joint_pos_target.torch[0].detach().cpu().numpy()
                        leg_t = tgt[leg_joints]
                        step_max = (float(abs(leg_t - prev_leg_target["v"]).max())
                                    if prev_leg_target["v"] is not None else 0.0)
                        prev_leg_target["v"] = leg_t.copy()
                    except Exception as exc:
                        step_max = 0.0
                        if len(state["errors"]) < 5:
                            state["errors"].append("joint_pos_target: %s: %s" % (type(exc).__name__, exc))
                    cf = sensor.data.net_forces_w.torch[0]
                    loads = torch.linalg.norm(cf[sensor_wheel], dim=1) if sensor_wheel else None
                    loaded = int((loads > 5.0).sum()) if loads is not None else 0
                    nonwheel = False
                    if sensor_nonwheel or sensor_base:
                        idx = sensor_nonwheel + sensor_base
                        nonwheel = bool((torch.linalg.norm(cf[idx], dim=1) > 5.0).any())
                    wm = torch.as_tensor([dq[j] for j in wheel_joints], device=env.device)
                    diff_m = float(((wm * side).sum()) / 2.0)
                    diff_d = None
                    try:
                        aw = torch.as_tensor(actions[0][12:16], device=env.device, dtype=torch.float32)
                        # the R4000 wheel mapping is vx/(0.086*16) with an open-loop differential term, so the
                        # action differential times the scale is a lower bound of the dispatched differential
                        diff_d = float(((aw * side).sum()) / 2.0) * 16.0 * 0.086
                    except Exception:
                        diff_d = None
                    state["frames"].append({
                        "t": round(float(env.episode_length_buf[0]) * dt, 4),
                        "reset": bool(env.episode_length_buf[0] <= 2),
                        "cmd": [round(float(x), 4) for x in cmd[:3]],
                        "vx": round(float(vxy[0]), 4), "wz": round(wz, 5),
                        "loaded": loaded,
                        "leg_step_max": step_max,
                        "leg_dq_sq": float(sum(dq[i] ** 2 for i in leg_joints) / max(1, len(leg_joints))),
                        "wheel_dq_sq": float(sum(dq[i] ** 2 for i in wheel_joints) / max(1, len(wheel_joints))),
                        "diff_dispatched": diff_d,
                        "diff_measured": diff_m,
                        "nonwheel_contact": nonwheel,
                        "term": bool(don[0]),
                    })
            except Exception as exc:
                state["errors"].append("frame: %s: %s" % (type(exc).__name__, exc))
            return obs, rew, don, info

        wrapped.step = recorded_step
        try:
            return state["orig_eval"](env, wrapped, actor, args)
        finally:
            try:
                args_out = getattr(args, "out", None)
                if args_out:
                    payload = {"frames": state["frames"], "errors": state["errors"]}
                    (args_out / "trace_frames.json").write_text(json.dumps(payload))
                    summary = analyse(state["frames"], dt)
                    (args_out / "trace_summary.json").write_text(json.dumps(summary, indent=1))
                    print("AS2W_TRACE frames=%d errors=%d cases=%s"
                          % (len(state["frames"]), len(state["errors"]), sorted(summary)), flush=True)
                    for cls, runs in sorted(summary.items()):
                        for r in runs[:2]:
                            l3 = r.get("last_3s") or {}
                            print("  %-16s dur %.2fs yaw_last3=%s diff_d=%s diff_m=%s delivery=%s "
                                  "support=%s unload=%.2fs leg_step_max=%.6f term=%s"
                                  % (cls, r["duration_s"], l3.get("yaw_signed_mean"),
                                     l3.get("diff_dispatched_mean"), l3.get("diff_measured_mean"),
                                     l3.get("diff_delivery"), l3.get("support_frac"),
                                     r["longest_continuous_unload_s"],
                                     l3.get("leg_target_step_max_rad") or 0, r["terminated"]), flush=True)
            except Exception as exc:
                print("AS2W_TRACE_DUMP_ERROR %s: %s" % (type(exc).__name__, exc), flush=True)

    import builtins

    def _lookup():
        # during a circular/partial import the module object exists in sys.modules before its attributes do,
        # so never touch it without checking; an exception here would break the frozen entry's import chain.
        mod = sys.modules.get("eval_as2w_isaac_speed_range")
        if mod is None or state["orig_eval"] is not None:
            return
        fn = getattr(mod, "evaluate", None)
        if not callable(fn):
            return
        state["orig_eval"] = fn
        print("AS2W_TRACE_EVALUATOR_FOUND evaluator=%s" % getattr(fn, "__name__", "?"), flush=True)

    def _apply():
        if state["orig_eval"] is None:
            return
        mod = sys.modules.get("eval_as2w_isaac_speed_range")
        if mod is None or getattr(mod, "_as2w_traced", False):
            return
        if getattr(mod, "evaluate", None) is not state["orig_eval"]:
            return
        mod.evaluate = traced_evaluate
        mod._as2w_traced = True
        print("AS2W_TRACE_EVALUATOR_PATCHED", flush=True)

    original_import = builtins.__import__

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        result = original_import(name, globals, locals, fromlist, level)
        try:
            _lookup()
            _apply()
        except Exception as exc:
            if len(state["errors"]) < 5:
                state["errors"].append("hook: %s: %s" % (type(exc).__name__, exc))
        return result

    builtins.__import__ = hooked_import

    entry_path = os.path.join(snapshot, "train_as2w_isaac_robust.py")
    entry = load_module("r4000_entry_traced", entry_path)
    print("AS2W_TRACE_ENTRY snapshot=%s" % snapshot, flush=True)
    entry.main()
    print("AS2W_TRACE_DONE errors=%d" % len(state["errors"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
