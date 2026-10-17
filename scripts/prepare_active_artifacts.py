"""Collect complete raw trials and exact source snapshots for publication."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
base = root/"artifacts/active"
base.mkdir(parents=True, exist_ok=True)
mapping = dict(course="raw_guard_course", stairs="final_stairs", blind="final_blind",
    slalom="final_slalom", nju_gap="publish_nju_gap", wide_stop="final_wide_stop",
    near_goal="final_near_goal", depth="final_depth", lidar="final_lidar",
    no_prepare="publish_no_prepare", changed_goal="raw_guard_goal")
for name, source in mapping.items():
    src = root/"outputs"/source
    if not (src/"summary.json").exists():
        raise FileNotFoundError(src/"summary.json")
    shutil.copytree(src, base/name, dirs_exist_ok=True)
shutil.copy2(root/"active_tests_final.log", base/"tests.log")
explore = base/"exploration"
explore.mkdir(exist_ok=True)
for name in ["bbc3_calibration", "bbc2_calibration", "bbc3_canter", "bbc3_limits",
             "course_calibration", "course_contact_calibration", "nju_course_search", "nju_course_fast"]:
    trial = root/"outputs"/name/"trials.json"
    if trial.exists():
        shutil.copy2(trial, explore/(name+".json"))
(explore/"README.md").write_text("These are exploratory frozen-policy timing searches, not RL training. "
    "Earlier searches used evolving controller/scene prototypes, including a course with descending stairs. "
    "They are retained as development evidence and are not release benchmarks. "
    "Release comparisons are the sibling directories with summary.json and archived source hashes.\n")

sources = base/"sources"
sources.mkdir(exist_ok=True)


def archive(payload):
    digest = hashlib.sha256(payload).hexdigest()
    (sources/(digest+".py")).write_bytes(payload)


for path in (root/"go2_nav").glob("*.py"):
    archive(path.read_bytes())
# Preserve the earlier release's source for unaffected non-BBC ablations.
# Its only differences are BBC release checks and a comment, not their physics.
env = (root/"go2_nav/active_env.py").read_text()
comment = ("        # BBC's previous-action feature follows upstream's raw network output\n"
           "        # convention. It must not be overwritten with joint-clipped targets.\n")
guard = ("                front_support = any(f in contacts for f in self.foot_ids[:2])\n"
         "                rear_support = any(f in contacts for f in self.foot_ids[2:])\n"
         "                if front_support and rear_support and self.rotation[2, 2] > .9 and self.data.qpos[2]-self.support_z > .23:\n")
old_guard = "                if sum(f in contacts for f in self.foot_ids) >= 2 and self.rotation[2, 2] > .8:\n"
for without_comment in (False, True):
    for old in (False, True):
        variant = env.replace(comment, "") if without_comment else env
        if old:
            variant = variant.replace(guard, old_guard)
        archive(variant.encode())
print("Collected", len(mapping), "trial groups")
