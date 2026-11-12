# AS2-W R1199 rolling reanchor preview

R1199 checkpoint and ONNX with a revised rolling handoff. **Simulation preview, full acceptance pending.** The model is unchanged from R1199. The revised behavior is implemented in the simulation wrapper.

| MuJoCo at 0.05 m/s | Full trial pass |
| --- | --- |
| Measured-posture proxy, no pushes | 8/8 |
| Independent seeds with pushes | 8/12 |

In the pushed suite, 12/12 entered and returned to fixed stand. Four cases failed the 10 s hold speed limit, including one reverse case that also exceeded the stopping-time limit. The full gate remains false. Matched Isaac Lab, left/right turns, the speed ladder through 3 m/s, and sustained hardware jitter remain unverified for this revision.

## Contents

- `weights/` contains the checkpoint and ONNX. Check both against `SHA256SUMS`.
- `runtime/` contains the action wrapper, simulator entry points, mapping reference and deployment plan.
- `evidence/` contains the existing ONNX parity report and the new MuJoCo summaries. The existing parity report checks raw policy actions. It predates the reanchor wrapper.

The preview contains no hardware command publisher. See [deployment plan](runtime/DEPLOYMENT.md) for the evaluation command and integration contract.
