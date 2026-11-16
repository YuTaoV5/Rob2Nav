#!/usr/bin/env python3
"""B-group training entry: the frozen R4000 entry, unmodified, with exactly one class rebound.

Why this shape
--------------
The R4000 run's entry (``source/train_as2w_isaac_robust.py``) builds its command term through
``configure_speed``, which resolves the module global ``as2w_isaac_speed_range.RollingSpeedCommands`` at
call time.  Rebinding that module attribute before the entry runs therefore swaps the turn curriculum
without editing a single byte of the R4000 source snapshot.  The offline self-check uses that property:
it hashes the snapshot before and after the B launch plan and requires them to be identical.

Everything else - entry point, flags, observation, network, action wrapper, control contract, rewards,
PPO, initialisation, seed, environment count, iteration count - is exactly the A command line.
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    snapshot = os.environ.get("AS2W_R4000_SNAPSHOT", "")
    if not snapshot or not os.path.isdir(snapshot):
        print("ERR: AS2W_R4000_SNAPSHOT must point at the R4000 run snapshot source directory, got %r"
              % snapshot, file=sys.stderr)
        return 2
    entry = os.path.join(snapshot, "train_as2w_isaac_robust.py")
    if not os.path.isfile(entry):
        print("ERR: missing R4000 entry %s" % entry, file=sys.stderr)
        return 2
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (snapshot, here):
        if path not in sys.path:
            sys.path.insert(0, path)

    import as2w_isaac_speed_range as sr
    from b_adapter import make_b_class

    base = sr.RollingSpeedCommands
    sr.RollingSpeedCommands = make_b_class(base)
    os.environ["AS2W_TURN_CURRICULUM_B"] = "1"
    print("AS2W_TURN_B_ENTRY snapshot=%s base_class=%s name=%s"
          % (snapshot, base.__name__, sr.RollingSpeedCommands.__name__), flush=True)

    # Run the frozen entry as if it were invoked directly, with the same argv.
    runpy.run_path(entry, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
