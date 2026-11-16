# 长驻留实现 + Isaac 冒烟实测证明（操作员第 3 项）

## 1. 冒烟结论：**长驻留已真实生效，平均 9.68 秒落在计划 8–12 秒内**

真实 Isaac 环境、1024 环境、40 次迭代、完整跑完（`R11_FANZIQI_LAB_COMPLETE`），`R11_GUARD_COUNTS {'crossed': 0, 'rear_overextended': 0, 'low_knee': 0, 'checks': 960}`。

**汇总行（训练日志原文）**：

```
LONG_TURN_SUMMARY steps=972800 started=544 completed=327 truncated=9 \
  mean_actual_dwell=9.68s planned=[8.0,12.0] turning_step_share=0.850
```

| 量 | 值 | 判定 |
| --- | --- | --- |
| 计划驻留范围 | [8.0, 12.0] s | 配置 |
| **实测平均驻留** | **9.68 s** | ✅ **落在计划区间内** |
| 启动的长驻留 | **544** | — |
| **按计划完成** | **327** | ✅ 证明驻留真的发生了 |
| 被新采样打断 | 9 | 如实单列，不静默丢弃 |

**逐环境 END 事件样本（真实时长）**：

```
LONG_TURN_END env=379 actual=8.54s planned=8.56s completed=True  reason=resampled
LONG_TURN_END env=401 actual=7.12s planned=11.18s completed=False reason=resampled
LONG_TURN_END env=183 actual=7.16s planned=10.63s completed=False reason=resampled
LONG_TURN_END env=494 actual=0.00s planned=9.70s completed=False reason=resampled
```

→ `actual=8.54 planned=8.56 completed=True` 就是「斜坡完成后指令真的保持了约 8.5 秒」的直接证据。

事件日志全文：`evidence/dwell/long_dwell_events.log`。

## 2. ⚠️ 一个我自己标的错名，必须先声明

上面 `turning_step_share=0.850` 这个字段**名不副实**：它的定义是
`(|wz| > 1e-3) | (|vx| > 1e-3)`，**把直行也算成了「转向」**。按采样配比，真正的转向类（纯转 40% + 弧线 25%）= **65%**，而 0.850 是**非零指令步占比**。
**→ 该字段的正确读法是「实际非零指令步占比 0.850」，不是「实际转向步占比」。**我没有改这个字段名（会改动已跑作业的日志口径），而是在此**明确标注**，下一版会把它拆成两个字段。

## 3. 实现路径与踩到的三个真 bug

新类 `training/as2w_long_dwell_curriculum.py :: LongDwellCommands`，在父类行为之上：

| 能力 | 实现 |
| --- | --- |
| 8–12 s 随机驻留 | 长样本的 `time_left` 设为 U(8,12) |
| 按类别分比例 | 纯转 37.5%（全体 15 pp）、弧线 40%（全体 10 pp） |
| 事件日志 | `LONG_TURN_START` / `END` / `SUMMARY` |

**三个真 bug，全部由自检或冒烟抓出，全部已修：**

| # | bug | 怎么被抓到 | 后果（若未修） |
| --- | --- | --- | --- |
| 1 | 短样本沿用上一轮的长 `time_left` | **CPU 自检第 3 项** | **短指令被错误保持 8–12 秒** |
| 2 | 辅助方法命名为 `_resample`，**覆盖了 `CommandTerm._resample`** | Isaac 冒烟 `TypeError: multinomial num_samples must be int, not slice` | 课程初始化直接崩 |
| 3 | `self.step_dt` 不存在（`step_dt` 在 env 上） | Isaac 冒烟 `AttributeError` | 累计时长无法推进 |

**→ 第 1 个是最危险的一个：它不会报错，只会让短指令悄悄变成长驻留，直接污染整个对照实验。**

## 4. 自检（6/6，`LONG_DWELL_SELFTEST_PASS`）

`training/as2w_long_dwell_curriculum_selftest.py`

| # | 检查 | 实测 |
| --- | --- | --- |
| 1 | 长样本子比例 | 纯转 0.370 / 弧线 0.394 ✅ |
| 2 | 计划驻留范围 | 长 8.000–11.998、短 0.000 ✅ |
| 3 | `time_left` 真的收到计划值 | 长 8.000–11.998、**短 1.8–3.0（已重置）** ✅ |
| 4 | 长驻留按计划完成 | 994/994 ✅ |
| 5 | 被打断记为 TRUNCATED | 69 个 ✅ |
| 6 | 汇总算术 | mean 9.95 s ✅ |

## 5. 下一步：**只有驻留时长不同的短对照**

操作员要求「只有驻留时长不同」的对照。现在具备条件，三组同配比、同起点、同种子：

| 组 | 驻留 | 开关 |
| --- | --- | --- |
| D0 | 10 s 固定（现状） | `AS2W_AB_LONG_DWELL_CURRICULUM=0` |
| D1 | 8–12 s 随机 | `=1`（本轮已证明生效） |
| D2 | 2.5 s 短驻留 | `resampling_time_range=(2.5,2.5)` |

## 6. 边界

1. 冒烟是 **40 次迭代、1024 环境**的短作业，其 `success_rate`/奖励数值**不作为验收结果**
2. `turning_step_share` 字段名不副实（见第 2 节）
3. 未做 D0/D1/D2 三组对照，仅验证了 D1 的机制生效
4. 冒烟使用 1024 环境而正式训练用 4096，采样统计会有差异
