# 双口径报告工具（mean / min）+ 一处我自己的判定错误

操作员要求评测「**同时报 mean 与 min 两个口径**」。验收脚本的 JSON 里已经有 `yaw_track_mean`、`support_min`、`upright_min` 等字段，但控制台只打印通过数，没有给出两个口径各自的判决。本工具从 JSON 直接产出双口径表。

## 1. 两个口径的定义

| 口径 | 通过条件 | 用途 |
| --- | --- | --- |
| **min 口径** | **每个种子都通过**才算通过 | **决定性口径** —— goal 按最差工况选候选 |
| **mean 口径** | 参与种子的平均跟踪在带内、最差支撑/倾角仍达标、无终止 | 用于识别「接近」而非「无望」的用例 |

**pass(mean) 但 fail(min) 的用例 = 种子间有散布** —— goal 按最差工况判定，因此这类用例**不得晋级**，但值得记录，因为它说明还有改善空间。

## 2. 🔴 工具首次运行暴露我自己的判定错误

第一次跑在真实的 `ff_arm.json` 上，输出 **`brake_fwd` / `brake_bwd` min=PASS 但 mean=FAIL** —— 明显矛盾，因为验收脚本自身的判决是 PASS。

**原因**：我在 mean 口径里对**所有**用例都套用了 vx 跟踪带 `[0.70, 1.30]`。但**刹车用例的末段 vx 本就接近 0**，`vx_track ≈ 0.01` 自然落在带外 → 每个刹车都被误判失败。

**修正**：跟踪带**只施加于真正要求跟踪的用例**：
- yaw 带 → 仅 `turn*` 与 `arc*`- vx 带 → 仅 `straight*` 与 `arc*`

修正后结果：**min 6/11、mean 6/11、散布 0**（单种子，两口径应当一致，符合预期）。

## 3. 验证输出（A 组 `model_5500`，单种子 7393）

| 用例 | min | mean | yaw_mean | sup_min | up_min |
| --- | --- | --- | --- | --- | --- |
| hold | PASS | PASS | - | 1.000 | 0.999 |
| turn_left_0.05 | FAIL | FAIL | +0.04 | 1.000 | 0.997 |
| turn_right_0.05 | FAIL | FAIL | +0.12 | 1.000 | 0.997 |
| turn_left_0.10 | FAIL | FAIL | +0.02 | 1.000 | 0.998 |
| turn_right_0.10 | FAIL | FAIL | +0.25 | 0.993 | 0.991 |
| straight_fwd_0.25 | PASS | PASS | - | 1.000 | 0.994 |
| straight_bwd_0.25 | PASS | PASS | - | 1.000 | 0.999 |
| arc_left_0.25 | FAIL | FAIL | +0.33 | 1.000 | 0.998 |
| arc_right_0.25 | PASS | PASS | +1.27 | 0.960 | 0.986 |
| brake_fwd_0.25 | PASS | PASS | - | 1.000 | 0.996 |
| brake_bwd_0.25 | PASS | PASS | - | 1.000 | 0.999 |

**→ min 6/11、mean 6/11、散布 0。**

## 4. 用途

B 组终点评测（看护作业）产出的 `ff_arm.json` 可直接喂给本工具，得到双口径表。

```bash
python runtime/r11_dual_basis_report.py --json <eval>/ff_arm.json --out <eval>/dual_basis.json
```

## 5. 边界

1. **mean 口径是本工具重新实现的**，不等同于验收脚本的原生判决；原生 `pass_all` 才是 min 口径的权威2. 首次运行即发现一处我自己的逻辑错误（刹车被 vx 带误判），已修正；说明**新工具必须在真实数据上先自检**3. 未对「散布」用例做统计显著性处理 —— 两种子下的散布只是提示，不构成结论
