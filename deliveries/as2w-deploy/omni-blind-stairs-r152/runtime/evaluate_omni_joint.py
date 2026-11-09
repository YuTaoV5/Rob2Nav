"""Run the fixed and independent joint locomotion benchmarks."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from straight_roll import StraightRoll
from yaw_harmonic import YawHarmonic
from centered_conditioned_gait import CenteredConditionedGait
from conditioned_gait import ConditionedGait
from reverse_conditioned_gait import ReverseConditionedGait
from stop_stance import StopStance
from stall_response import StallResponse
from flat_trim import FlatTrim
from flat_recovery import FlatRecovery
from signed_conditioned_gait import SignedConditionedGait
from train_omni_lateral_history import screen
from train_omni_stop_transition import load


def make_controller(policy, contract, controller, route, adjuster=None):
    if policy.get('stair_stall_response'):
        route = StallResponse(route, policy['stair_stall_response'])
    actor = controller(route, tuple(policy.get("steering_release", (0., False))))
    geometry = actor.teacher.geometry
    if "yaw_harmonic" in policy:
        actor.teacher.yaw_harmonic = YawHarmonic(policy["yaw_harmonic"], contract["control_dt"])
    actor.teacher.swing_profile = policy.get("flat_swing_profile") or None
    if policy.get("flat_trim"):
        actor.teacher.move = FlatTrim(actor, contract["control_dt"], policy["flat_trim"])
    if "center_gait_conditioning" in policy:
        actor = CenteredConditionedGait(actor, policy["reverse_gait_conditioning"],
                                        policy["forward_gait_conditioning"], policy["center_gait_conditioning"],
                                        policy["forward_yaw_window"])
    elif "forward_gait_conditioning" in policy:
        actor = SignedConditionedGait(actor, policy.get("reverse_gait_conditioning", (0.,) * 8),
                                     policy["forward_gait_conditioning"],
                                     policy.get("forward_yaw_window", (0., .01)))
    elif "reverse_gait_conditioning" in policy:
        actor = ReverseConditionedGait(actor, policy["reverse_gait_conditioning"])
    elif "gait_conditioning" in policy:
        actor = ConditionedGait(actor, policy["gait_conditioning"])
    if adjuster is not None:
        actor = adjuster(actor, geometry)
    elif "flat_recovery" in policy:
        actor = FlatRecovery(actor, geometry, policy["flat_recovery"])
    if "stop_stance" in policy:
        actor = StopStance(actor, contract["control_dt"], policy["stop_stance"])
    if "straight_roll" in policy:
        actor = StraightRoll(actor,contract["control_dt"],geometry,policy["straight_roll"])
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--audit-inputs", action="store_true",
                        help="Require identical actions with simulator handles removed")
    args = parser.parse_args()
    policy, contract, limits, base, controller = load(args.results.resolve(), args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)

    audited_ticks = 0

    def create(route):
        live = make_controller(policy, contract, controller, route)
        if not args.audit_inputs:
            return live
        detached = make_controller(policy, contract, controller, base())

        class AuditedController:
            def plan(self, model, data, obs, command, tick, advance):
                nonlocal audited_ticks
                np.testing.assert_allclose(obs[:3], command - (.4, 0., 0.), rtol=0., atol=1e-7)
                action = live.plan(model, data, obs, command, tick, advance)
                detached_action = detached.plan(None, None, obs.copy(), command.copy(), tick, None)
                np.testing.assert_array_equal(action, detached_action)
                audited_ticks += 1
                return action

        return AuditedController()

    for confirmation in (False, True):
        result = screen(base, policy["anchor"], None, np.asarray(policy["parameters"]),
                        contract, limits, create, confirmation)
        name = "confirmation" if confirmation else "fixed"
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2, default=lambda x: x.item()))
        flat, stairs = result["flat"], result["stairs"]
        print(name, {"flat": [sum(x["passed"] for x in flat), len(flat), sum(x["fallen"] for x in flat)],
                     "stairs": dict(Counter(x["stair_rise_m"] for x in stairs if x["passed"])),
                     "stair_total": len(stairs)}, flush=True)

    if args.audit_inputs:
        print("Proprioception-only equivalent actions", audited_ticks, flush=True)


if __name__ == "__main__":
    main()
