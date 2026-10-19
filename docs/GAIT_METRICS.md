# First-episode gait diagnostics

2026-10-20. `moe_perceptive.evaluate --gait-metrics` records diagnostics from the pre-reset state captured by the existing termination hook. Accumulation includes the terminal step and stops for each environment after its first episode. It does not modify actions, rewards or success rules.

* Stance criterion: vertical net foot-link contact force >5N. Integrate horizontal world foot-link speed over all stance samples, divide by summed stance-foot time. This is a link-velocity slip proxy, not contact-point slip or obstacle collision count.
* Absolute mechanical joint work: integral of sum of `abs(applied_torque * joint_velocity)` over time. It is not battery energy, signed net work or cost of transport.
* Record duration, planar speed integral, mean/minimum upright and sample counts. Compare success and failure groups separately and account for unequal episode duration.
* Command tracking error, shin collisions, foot-clearance margins, gait aesthetics and battery power remain unmeasured. Do not fabricate these from the current fields.

Validation:35 unit/API tests passed; production frontend build passed. Native Isaac smoke ran8 environments for1 second with the retained platform checkpoint, seed900, without interrupting the500-iteration mixed training. All8 first episodes included metrics. Environment0:50 samples,2.54 stance-foot-seconds,0.14149m/s stance-link speed proxy,33.278J absolute joint work and0.99716 mean upright. These are short startup measurements, not skill-quality or success estimates. The summary is explicitly unscored/diagnostic.

`moe_perceptive.suite` now samples gait diagnostics by default; `--no-gait-metrics` retains the legacy measurement protocol. Independent `--seed` is supported. The current active training process is unchanged; its later evaluation subprocess will use the new suite. The next gates are completed500-iteration evaluation, matched retained-model comparisons at independent seeds, then a justified continuation or redesign. A high training terrain level alone is not a reason to promote weights.

Studio can display these records even without a video and separates success/failure groups. Missing metrics remain missing; short-horizon diagnostic scores do not enter the task benchmark.
