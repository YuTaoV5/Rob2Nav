# AS2-W rolling leg-filter parity

Offline comparison for the 0.20 m/s filtered run, terminal checkpoint SHA256
7a27a6c9cb36332c87f7bd6676b16e67b00b85716795a5c7a4398f388e918dde.

The C++ component expects the 12 leg outputs from the 65×16 actor, physical vx
and wz, and the normalized stance offset supplied by the surrounding controller.
It reproduces ±6 actor clipping, the 5 Hz low-pass at a 20 ms control step,
motion gating, ±1 residual clipping, stance addition and 0.9 rad/s target
slew in that order. Filter and slew state are zeroed at reset. Outputs are
normalized position targets. Multiply by 0.35 rad and add the active joint
reference to obtain leg position targets before final safety checks.

The replay uses 1,034 observations from the recorded session replay and five
synthetic frames. The stance offsets are deterministic test inputs. Wheel
control, stance-offset computation, motor mapping, command transport, FSM,
safety limits, and ARM64 behavior need their own verification. The program
reads stdin and writes stdout. It does not publish motor commands.

Build with a C++14 compiler:

```bash
c++ -std=c++14 -O2 -Wall -Wextra -Werror replay_filter.cpp -o replay_filter
```

Run with the field evidence archive, original R1199 checkpoint, and candidate
checkpoint. The audit script is beside the rolling replay report. The parity
script requires Python, NumPy and PyTorch:

```bash
python verify_filter.py ./replay_filter ../R1199-motion-jitter-audit/compare_actor_replay.py \
  FIELD_ARCHIVE.zip R1199.pt CANDIDATE.pt actor-audit.json parity.json
```
