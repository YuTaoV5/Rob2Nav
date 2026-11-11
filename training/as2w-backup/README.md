# AS2-W training reproduction

This snapshot preserves the AS2-W training workspace from the backup server. Read as2w-current/AGENTS.md for the evaluation and safety requirements.

On Linux with a compatible NVIDIA GPU and the Isaac Sim / Isaac Lab environment, run these commands from this directory. Output paths must be new and their parents must already exist.

```bash
bash train.sh rolling /tmp/as2w-rolling
bash train.sh stepping /tmp/as2w-stepping
bash train.sh recovery /tmp/as2w-recovery
```

The recipes use the archived upstream checkpoints, seeds 303/305/301 and 1200 iterations. They recreate the knee-margin rolling, stance-margin-v2 stepping and mixed-side recovery jobs. For an environment smoke check, prefix any command with AS2W_ITERATIONS=3 AS2W_NUM_ENVS=8. The original recipes, evaluation and ONNX export scripts are in as2w-current/runs and as2w-current/code. Historical shell scripts may refer to the original server paths, job IDs and intermediate checkpoints.

The training host reported Isaac Sim 6.2.0-rc.14 and Isaac Lab 28.0.0. The pip freeze files are environment inventories with editable paths, not portable pip install files. Supply the matching NVIDIA runtime separately. Set ISAAC_SIM_PYTHON, ISAACLAB_SOURCE and ISAACLAB_SITE_PACKAGES when your installation uses different paths. The Unitree description is from revision 4005a1b0c3f464f515f380be4145cc617fbca725. The snapshot includes the converted USD, robot description, seed checkpoints, teacher trajectories, contracts and historical evaluation summaries.

The historical weights remain simulation candidates. Rolling training uses a 1 m/s cap in this recipe. The 3 m/s evaluation target remains separate from that training cap. Reports in the archive record the original results and limitations.

The three [skill recipes](../../skills/as2w/README.md) are maintained separately. Each new run records source-revision.txt, input-sha256.txt and command.sh in its output directory.
