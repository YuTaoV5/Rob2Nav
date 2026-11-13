# AS2-W skills

## 训练与交付手册

- [AS2-W 训练守则](../../training/as2w-backup/as2w-current/AGENTS.md)
- [AS2-W 训练到部署交付清单 v3](../../training/as2w-backup/as2w-current/AS2-W-training-delivery-checklist-v3.md)
- [AS2-W 后双轮直立训练与部署手册](rear-two-wheel/README.md)

| Skill | Training recipe | Scope |
|---|---|---|
| Rolling | [formal.sh](rolling/recipes/formal.sh) | Straight rolling, turning, braking and fixed stand |
| Rolling, filtered low speed | [filtered.sh](rolling/recipes/filtered.sh) | Opt-in 0.20 m/s controller and reward smoke test |
| Stepping | [formal.sh](stepping/recipes/formal.sh) | Omnidirectional stepping and stairs |
| Recovery | [formal.sh](recovery/recipes/formal.sh) | Rising from prone and side tilt |

Run the recipes through the [training entry point](../../training/as2w-backup/README.md). Each output directory records the Git revision, recipe and input hashes, and the effective command. The historical simulator and evaluation scripts remain in the [training snapshot](../../training/as2w-backup/as2w-current) for reproduction.

## Rolling evaluation

Compare reports from the same checkpoint and speed cap with the archived R1199 example below. The output retains each backend's safety and behavior gates, scenario counts, and Isaac Lab turn rows. Inspect the original contracts before comparing pass counts across different protocols.

```bash
D=training/as2w-backup/as2w-current/runs/r290-three-skills
python skills/as2w/rolling/compare_reports.py \
  "$D/rolling-knee-margin-formal/model_1199.pt" \
  "$D/rolling-knee-margin-fixed-exit-trace-1199" \
  "$D/rolling-knee-margin-fixed-exit-trace-isaac-1199"
```
