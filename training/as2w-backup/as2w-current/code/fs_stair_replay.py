"""FS stair line: offline window replay and first-divergence localisation.

Two different questions, deliberately kept apart:

* the whole-window ``history_sha256`` can only localise the first **policy
  step** that differs from the recording;
* locating the **field** that differs needs that step's raw window, which is why
  the runner samples windows to ``.npz``.

``compare_windows`` answers the second question (field-level diff), ``scan_run``
answers the first (first step whose recorded hash no longer matches the stored
window), and ``reexport`` writes the offending window plus the diff.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import fs_stair_contract as contract_mod

FIELD_ORDER = contract_mod.FIELD_ORDER


def window_sha256(obs) -> str:
    array = np.ascontiguousarray(np.asarray(obs, dtype=np.float32))
    return hashlib.sha256(array.astype("<f4").tobytes()).hexdigest()


def _load(path: Path) -> dict:
    with np.load(str(path), allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def compare_windows(recorded_path, reference_path) -> dict:
    """Field-level diff between the recorded window and a reference window."""
    recorded = _load(Path(recorded_path))
    reference = _load(Path(reference_path))
    if recorded["obs"].shape != reference["obs"].shape:
        raise ValueError(f"window shape mismatch: {recorded['obs'].shape} vs "
                         f"{reference['obs'].shape}")
    obs_r = np.ascontiguousarray(np.asarray(recorded["obs"], dtype=np.float32))
    obs_f = np.ascontiguousarray(np.asarray(reference["obs"], dtype=np.float32))
    if obs_r.ndim == 1:
        obs_r = obs_r.reshape(-1, 60)
        obs_f = obs_f.reshape(-1, 60)
    field_order = [str(name) for name in reference["field_order"]]
    if field_order != list(FIELD_ORDER):
        raise ValueError("window field_order does not match the observation contract")
    frames, dim = obs_r.shape
    recorded_hash = window_sha256(obs_r)
    reference_hash = window_sha256(obs_f)
    flat = np.flatnonzero(obs_r.view(np.uint32) != obs_f.view(np.uint32))
    result = {
        "recorded_hash": recorded_hash,
        "reference_hash": reference_hash,
        "stored_hash": str(recorded["history_sha256"][0]) if "history_sha256" in recorded else None,
        "hash_match": recorded_hash == reference_hash,
        "frames": int(frames), "fields": int(dim),
        "n_mismatch_values": int(flat.size),
        "first_mismatch": None,
        "suspicion": None,
        "field_order": field_order,
    }
    if flat.size:
        position = int(flat[0])
        frame_slot, field_index = divmod(position, dim)
        result["first_mismatch"] = {
            "frame_slot": int(frame_slot),                      # 0 = oldest
            "age_frames": int(frames - 1 - frame_slot),          # 0 = newest
            "field_index": int(field_index),
            "field_name": field_order[field_index]
            if field_index < len(field_order) else f"index_{field_index}",
            "recorded_value": float(obs_r[frame_slot, field_index]),
            "reference_value": float(obs_f[frame_slot, field_index]),
            "abs_diff": float(abs(obs_r[frame_slot, field_index]
                                  - obs_f[frame_slot, field_index])),
        }
        if frames > 1 and np.array_equal(obs_r, obs_f[::-1]):
            result["suspicion"] = "reversed_frame_order"
        elif frames > 1 and np.all(obs_r == obs_r[-1]):
            result["suspicion"] = "repeated_fill_all_slots_equal_newest"
    return result


def scan_run(log_path, windows_dir) -> dict:
    """First policy step whose recorded hash no longer matches its stored window."""
    log_path = Path(log_path)
    windows_dir = Path(windows_dir)
    index_path = windows_dir / "index.jsonl"
    if not log_path.is_file() or not index_path.is_file():
        return {"status": "no_evidence", "log": str(log_path),
                "windows": str(windows_dir)}
    windows = {}
    for line in index_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        windows[(row.get("env_id", "w0"), int(row["step_id"]))] = row
    rows = [json.loads(line) for line in log_path.read_text().splitlines()
            if line.strip() and json.loads(line).get("kind") != "triple_tap_truncated"]
    checked = mismatches = 0
    for row in rows:
        key = (row.get("env_id", "w0"), int(row["step_id"]))
        window = windows.get(key)
        if window is None:
            continue
        payload = _load(windows_dir / window["path"])
        recomputed = window_sha256(payload["obs"])
        checked += 1
        if recomputed != row["history_sha256"]:
            mismatches += 1
            if mismatches == 1:
                return {"status": "mismatch", "env_id": key[0], "step_id": key[1],
                        "episode_id": row.get("episode_id"),
                        "policy_step": row.get("policy_step"),
                        "recorded_hash": row["history_sha256"],
                        "recomputed_hash": recomputed,
                        "window": str(windows_dir / window["path"]),
                        "checked_before": checked}
    return {"status": "ok" if checked else "no_windows_checked",
            "checked": checked, "mismatches": mismatches}


def reexport(window_path, out_path, diff: dict) -> dict:
    """Export the offending raw window and its field-level diff."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load(Path(window_path))
    np.savez_compressed(out_path, **payload)
    diff_path = out_path.with_suffix(".diff.json")
    diff_path.write_text(json.dumps(diff, indent=2) + "\n")
    return {"window": str(out_path), "diff": str(diff_path)}
