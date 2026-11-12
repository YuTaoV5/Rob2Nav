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

## Filtered low-speed rolling

From the repository root, run the opt-in Isaac Lab recipe:

```bash
bash training/as2w-backup/train.sh rolling-filtered /tmp/as2w-rolling-filtered
```

This defaults to 24 environments and 20 iterations. It initializes from the archived
R1199 rolling knee-margin checkpoint, uses seed 311 and a 0.20 m/s cap, and retains
the expanded domain randomization, safety constraints and existing rolling rewards.
The short run checks the training loop and reward activation. For a later approved
long run, set `AS2W_NUM_ENVS=384 AS2W_ITERATIONS=1200` on the same command.

The new runtime option, `--rolling-leg-filter-hz 5`, applies a causal first-order
low-pass to the 12 clipped leg residuals before the rolling wrapper. At 50 Hz its
coefficient is about 0.467. State resets to zero on each environment reset.
Wheel commands bypass the filter so wheel braking is not delayed by it.
The wrapped leg slew limit, motion gating, and safety protections still apply.
The option also enables a gradual low-yaw penalty and a post-release phase reward
with distinct braking and parked position anchors. The parked phase penalizes
residual motion, drift and joint deviation from fixed stand. These costs do not
modify the controller during inference or replace any acceptance gates.

Use the same `--rolling-leg-filter-hz 5` when running the archived
`eval_as2w_mujoco_rolling.py` and the Isaac evaluator
`train_as2w_isaac_robust.py --evaluate`, with the same fixed-exit and park options
as the recipe. For Isaac also pass `--rolling-low-speed-smooth`. Leave out the
training-only `--rolling-maneuvers` and `--rolling-knee-margin` in Isaac evaluation.
Keep the ONNX actor interface at 65 inputs and 16 outputs. The filter has runtime
state outside ONNX and requires equivalent deployment code and reset parity before
any hardware use. A smoke pass does not qualify a checkpoint for deployment or
unlock a higher speed tier.

Server smoke evidence: Isaac Lab completed the 24-environment, 20-iteration
recipe and logged a nonzero post-release penalty. MuJoCo ran all six
0.20 m/s scenarios with the R1199 seed and the same filter enabled. Its
safety gate passed, and none of the six trials met the full behavior gate.
