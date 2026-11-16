#!/usr/bin/env python3
"""Single entry for BOTH continuation arms: the frozen R4000 entry plus two documented injections.

Why one entry for both arms
---------------------------
The operator fixed the continuation budget at 16000 envs x 128 steps x 3000 iterations, but the frozen R4000
entry has no switch for ``num_steps_per_env`` (it constructs ``UnitreeGo2FlatPPORunnerCfg()`` and never
touches that field, so it would silently use 24).  Both arms must get the same value, so both are launched
through this one entry and the injection is identical; the only difference between the arms remains the
curriculum.

Why the injections happen lazily through an import hook
-------------------------------------------------------
Importing ``as2w_isaac_flat_env_all`` / ``as2w_isaac_speed_range`` before ``launch_simulation`` has already
broken this project once (``libusd_tf.so: undefined symbol pxrInternal_...``).  The frozen entry imports them
INSIDE ``run()``, i.e. after the simulator is up, so this entry never imports them itself: it installs an
``__import__`` hook that patches the module attribute right after the frozen entry imports it.  That is also
what keeps the snapshot byte-identical - nothing here writes into the R4000 run directory.

Injections
    both arms   runner_cfg.num_steps_per_env = AS2W_AB_NUM_STEPS   (default 128)
    arm B only  as2w_isaac_speed_range.RollingSpeedCommands = turn-curriculum-migration adapter

Environment variables
    AS2W_R4000_SNAPSHOT   path of the R4000 run source snapshot (required)
    AS2W_AB_ARM           A or B (default A)
    AS2W_AB_NUM_STEPS     num_steps_per_env to force (default 128)
    AS2W_TURN_B_SEED      seed for the B curriculum machine (default 311)
"""
from __future__ import annotations

import builtins
import os
import runpy
import sys


def _install_hook(arm: str, num_steps: int) -> dict:
    state = {"runner_patched": False, "b_patched": False, "errors": []}
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    def _runner_patch(mod):
        base = mod.UnitreeGo2FlatPPORunnerCfg
        if getattr(base, "_ab_num_steps_patched", False):
            return
        original_init = base.__init__

        def patched_init(self, *a, **k):
            original_init(self, *a, **k)
            self.num_steps_per_env = int(num_steps)

        base.__init__ = patched_init
        base._ab_num_steps_patched = True
        state["runner_patched"] = True
        print("AS2W_AB_RUNNER_OVERRIDE num_steps_per_env=%d class=%s" % (num_steps, base.__name__), flush=True)

    def _b_patch(mod):
        from b_adapter import make_b_class

        base = mod.RollingSpeedCommands
        mod.RollingSpeedCommands = make_b_class(base)
        state["b_patched"] = True
        print("AS2W_AB_CURRICULUM_OVERRIDE base=%s installed=%s"
              % (base.__name__, mod.RollingSpeedCommands.__name__), flush=True)

    def _apply():
        try:
            mod = sys.modules.get("as2w_isaac_flat_env_all")
            if mod is not None and not state["runner_patched"]:
                _runner_patch(mod)
            if arm == "B" and not state["b_patched"]:
                mod2 = sys.modules.get("as2w_isaac_speed_range")
                if mod2 is not None:
                    _b_patch(mod2)
        except Exception as exc:  # never hide a failed injection: it must fail the run loudly
            state["errors"].append("%s: %s" % (type(exc).__name__, exc))
            print("AS2W_AB_INJECTION_ERROR %s: %s" % (type(exc).__name__, exc), flush=True)

    original_import = builtins.__import__

    def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
        result = original_import(name, globals, locals, fromlist, level)
        _apply()
        return result

    builtins.__import__ = hooked_import
    return state


def main() -> int:
    snapshot = os.environ.get("AS2W_R4000_SNAPSHOT", "")
    if not snapshot or not os.path.isdir(snapshot):
        print("ERR: AS2W_R4000_SNAPSHOT must point at the R4000 run snapshot source directory, got %r"
              % snapshot, file=sys.stderr)
        return 2
    entry = os.path.join(snapshot, "train_as2w_isaac_robust.py")
    if not os.path.isfile(entry):
        print("ERR: missing frozen R4000 entry %s" % entry, file=sys.stderr)
        return 2
    arm = os.environ.get("AS2W_AB_ARM", "A").upper()
    if arm not in ("A", "B"):
        print("ERR: AS2W_AB_ARM must be A or B, got %r" % arm, file=sys.stderr)
        return 2
    num_steps = int(os.environ.get("AS2W_AB_NUM_STEPS", "128"))

    for path in (snapshot, os.path.dirname(os.path.abspath(__file__))):
        if path not in sys.path:
            sys.path.insert(0, path)

    state = _install_hook(arm, num_steps)
    print("AS2W_AB_ENTRY arm=%s snapshot=%s num_steps=%d" % (arm, snapshot, num_steps), flush=True)
    runpy.run_path(entry, run_name="__main__")
    print("AS2W_AB_ENTRY_DONE arm=%s runner_patched=%s b_patched=%s errors=%s"
          % (arm, state["runner_patched"], state["b_patched"], state["errors"]), flush=True)
    return 0 if not state["errors"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
