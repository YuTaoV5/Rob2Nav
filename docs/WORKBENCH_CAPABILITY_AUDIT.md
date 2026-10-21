# Workbench capability audit (2026-10-21)

The active workbench now retains seven independent skills with per-skill folders and three command manuals. Training completion is not acceptance; failing named stunt tasks remain visible with their failures. Hardware work is plans only.

| Retained skill | Folder | State |
|---|---|---|
| stairs | skills/stairs | 已有历史仿真 |
| platform | skills/platform | 已有历史仿真 |
| multiterrain | skills/multiterrain | 已有失败与保持对照 |
| backflip | skills/backflip | 动作通过；关节约束未通过 |
| sideflip | skills/sideflip | MuJoCo动作8/8；速度约束未通过 |
| navigation | skills/navigation | PPO训练完成；8回合与网页点击通过 |
| upright_walk | skills/upright_walk | 原生起立前进；MuJoCo未通过 |

The following cards and their unused training recipes are archived in studio/skill_registry.json and excluded from executable selectors. Historical artifacts/reports are preserved. This follows the request to remove hard-to-complete skills after investigating them; it does not claim they were solved.

| Archived skill | Reason |
|---|---|
| locomotion | Duplicate of the mixed-terrain MoE retention recipe; that recipe did not list locomotion as an executable capability. |
| narrow_stakes | Repeated native training and strict original-dimension evaluations failed; widened/shallow precursor scores do not establish this ability. |
| water_avoid | Historical helper prototype, no independent trained sensor-policy and SDK chain. |
| gap_jump | Historical skill composition with external jump dependency; no complete native Go2 train/sim/SDK chain. |
| recovery | External reference not integrated into the installed native training runtime. |
| gait | External IsaacGym policy not integrated into the installed native training runtime. |
| visual_parkour | Source inference only; native training and strict independent Go2 evaluations not established. |
| world_model | External IsaacGym world-model project not integrated; no runnable native recipe. |
| lidar_nav | FAST-LIO/local-planner references do not supply the requested native Go2 training chain; goal navigation is a separate retained skill. |
| deployment | A deployment contract is a cross-cutting plan, not a separately trainable motor skill. |
| motion_jump | Source weights/inference only; no verified native retraining and independent evaluation chain. |

The SDK plans describe missing robot-side adapters and do not invent a hardware launch command. The native MoE terrain skills share implementation in moe_perceptive but have separate configuration/manual folders. Navigation uses a live MuJoCo engine; terrain training/evaluation uses native IsaacLab/PhysX. Handshake remains a separate live MuJoCo demo, with its existing contact controller and deployment plan under go2_nav/paw_contact.
