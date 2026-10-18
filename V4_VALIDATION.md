# V4：真实深坑课程与失效诊断

2026-10-18。用户已批准停止平台期方案并执行新方案。**本页没有新的窄桩成功结论；V4仍在训练完整地图教师。**

## 已执行的停止与保全

旧 `nav_teacher_v3` 控制器及其训练进程组已通过 SIGINT 停止，不再自动启动后续块。最后一个完整原子检查点 `model_3500.pt` 经ZIP完整性检查后，复制到 `/root/gpufree-data/MoE-runs/nav_teacher_v3_stopped/`，保留源码、配置和SHA256。检查点后的未保存更新没有声称已保存。

[停止记录](artifacts/v4_redesign/v3_stop_for_redesign.json)、[保留权重](artifacts/v4_redesign/nav_teacher_v3_stopped/model_3500.pt)。第三轮前3000轮的三个完整gate均保留在 [v3_gates](artifacts/v4_redesign/v3_gates)。全部六组原桩始终为0/32；3000轮时平地/楼梯/高台各32/32。第四块未完成，不标记成完成4000轮。

## 先诊断，而非把猜测写成原因

固定第三轮 `model_2999.pt`，同一seed801、32环境、双列布局；本次新增执行前航向、足力和地图记录：

| 测试 | 完整通过 | 失败终止 | robot0平台边缘前航向偏差 | robot0低于平台0.1m的接触样本 |
|---|---:|---:|---:|---:|
| 原深坑，difficulty0.5 | 0/32 | 32/32 | 约63.1° | 0 |
| 浅坑诊断，difficulty0.5 | 2/32 | 30/32 | 约27.1° | 0 |
| 原深坑，difficulty1 | 0/32 | 32/32 | 约40.1° | 0 |

航向采样取robot0首次机身x>0.75m的诊断帧，不能作为所有32环境的统计。接触计数同样只来自robot0，不证明其他回合没有坑底支撑。**现有结果不支持简单断言“所有失败就是依赖坑底”**；浅坑本身也多数失败。可以确认旧策略存在较明显的航向与横向偏移，且深浅地形输入会改变行为。

![同一权重的足端首次接触与机身路径](artifacts/v4_redesign/v4_failure_analysis/foot_contacts.png)

[完整诊断统计](artifacts/v4_redesign/v4_failure_analysis/summary.json)、[原始回合/动作前地图](artifacts/v4_redesign/v4_diagnosis)。浅坑2/32不能计为原场景成功。

原生地图检查将robot0初始姿态的高度扫描XY与预期33×21网格逐点比较，再与解析原AME高度场比较；排除距碰撞盒边界2mm以内的格。两组各632个高度样本全部正确，最大XY误差约3.8微米，最大高度误差约2.6–3.8微米。**此次初始姿态检查未发现转置或偏移错误**，不代表已覆盖所有运动姿态和传感器误差。

[中等难度对齐记录](artifacts/v4_redesign/v4_diagnosis/deep05/map_alignment.json)、[最终难度对齐记录](artifacts/v4_redesign/v4_diagnosis/deep1/map_alignment.json)。

## 新课程的实际实现

- `moe_perceptive/sparse_course.py`新增 `wide_deep`：所有等级坑底始终−2m，支撑面从1.0m逐步缩到0.2m，起始间隙约5cm，保留5cm网格量化。
- 中间布局明确是训练前置任务。difficulty=1直接调用原AME生成器；测试已验证高度场逐格相同、碰撞盒列表相同。
- 原严格suite默认仍是original，不带wide_deep，绝不将宽桩成绩混为20cm原桩成绩。
- 每个环境在当前等级重新累计成功EMA与尝试数，完整到达后才晋级；晋级要求至少5次尝试且成功EMA>0.75，切换等级后清零，避免沿用易级成绩快速跳级。
- 针对诊断的偏航与横移：增加 `1-cos(yaw)` 惩罚（权重−2），通道中心惩罚从−0.5改为−2。没有在线修改关节动作或加入IK。
- 从第一轮MoE感知权重初始化，使用新的优化器与课程；不沿用第三轮已偏航/失效的策略。仍保留冻结的原MoE参考和明确物理限制。

四轮32环境原生短训练已完成，验证配置、PPO、课程更新与检查点可运行；两个几何测试已通过，确认每级真实坑及最终原几何一致。这些是接口和几何检查，不是越障技能证据。

[原生短训练完成记录](artifacts/v4_redesign/wide_deep_v4_smoke/completed.json)。

## 当前运行与自动评估

控制器 `moe_perceptive/deep_trial.py`；输出 `outputs/moe_perceptive/wide_deep_v4`；训练前缀 `/root/gpufree-data/MoE-runs/wide_deep_v4_chunkN`。

512环境，500轮一块，初始最多5块。每块结束：

1. 对原始九项场景完整评估，录制20cm双列/交错桩视频；仍是完整地图教师成绩。
2. 另测宽桩起始课程的双列/交错完整通过率，并做原生地图对齐检查，明确标为precursor。
3. 记录严格通过数、课程等级和precursor成绩；若连续两个gate均无实质改善，即使停在较高级也转入诊断，不再只判断是否卡在level0。

[启动记录](artifacts/v4_redesign/launch_wide_deep_v4.json)、[启动时源码/参数](artifacts/v4_redesign/wide_deep_v4_chunk1/manifest.json)。当前公开目录仅包含V4启动快照，不是最终权重或最终评估。

```bash
python3 -m moe_perceptive.deep_trial \
  --moe vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt \
  --initializer /root/gpufree-data/MoE-runs/visible_moe_v1_seed42/model_999.pt \
  --smoke /root/gpufree-data/MoE-runs/wide_deep_v4_smoke \
  --output outputs/moe_perceptive/wide_deep_v4 \
  --run-prefix /root/gpufree-data/MoE-runs/wide_deep_v4 \
  --chunks 5 --chunk-iterations 500 --num-envs 512
```

以上是已启动命令，接手后先检查进程，不重复运行。所有输出必须使用未存在的新目录。

## 后续仍未完成

教师真实深坑通过后，才继续可见D435i/Mid360学生蒸馏和PPO、安装/记忆误差验证、独立种子、MuJoCo以及目标点导航整合。当前阶段没有新增实机或学生成功结论。

归档包含旧三次完整gate、停止权重、匹配诊断、V4原生短训练及启动元数据。见 [文件校验清单](artifacts/v4_redesign/archive_manifest.json)。

## 2026-10-18 17:57 更新：V4 已触发平台期停训

控制器实际状态为 `needs_diagnosis`，完成四块共2000轮，保留 `wide_deep_v4_chunk4/model_1999.pt`，没有启动第五块。四次原场景严格评估的六组桩均为0/32。

| 块 | 平地 | 楼梯 | 高台 | 宽双列前置 | 宽交错前置 |
|---|---:|---:|---:|---:|---:|
| 1 | 32/32 | 32/32 | 27/32 | 3/32 | 17/32 |
| 2 | 32/32 | 32/32 | 31/32 | 7/32 | 18/32 |
| 3 | 32/32 | 29/32 | 28/32 | 5/32 | 22/32 |
| 4 | 32/32 | 32/32 | 25/32 | 4/32 | 6/32 |

第四块的独立视频/诊断重放为宽双列2/32、宽交错6/32；双列与普通评估有差异，不应假称严格确定性重放。两次初始地图对齐各632个非边界样本全部正确。robot0在平台边缘前偏航约−16.5°；双列最终机身位置约(1.84,−0.10,0.09)m，交错约(3.61,0.57,−0.13)m。仅这两个robot0轨迹均无足端低于平台0.1m的接触样本，不足以解释全部环境失败。

下一步先分析支撑几何、脚位、奖励与课程晋级。现有checkpoint只保存地形等级，未保存本级EMA/尝试次数，分块恢复会重新积累晋级资格；这是需要处理的恢复语义问题，但尚不能断言是失败根因。当前没有继续训练，也没有进入学生阶段。

服务器证据：`outputs/moe_perceptive/wide_deep_v4/`（四次gate与前置评估），`outputs/moe_perceptive/v4_plateau_diagnostic/`（视频、动作前地图、足力、analysis.json与contact_analysis.json）。这些新增完整V4结果尚未发布；公开的v4_redesign仍主要是V3停止、诊断和V4启动记录。

## V4完整评估归档（2026-10-18 18:12）

四次严格评估、八组宽桩前置评估、十段完整robot0视频及第二/第四块权重已收集至 [artifacts/v4_completed](artifacts/v4_completed)。[归档清单](artifacts/v4_completed/archive_manifest.json)记录1569个文件的SHA256与大小；服务器同时核对1296条源码引用及十段视频帧数，本地再次校验全部1569个文件。它们是失败与退化证据，不是成功演示。

复核命令：`python -m moe_perceptive.verify --root artifacts/v4_completed --manifest-name archive_manifest.json`（需numpy/torch及imageio/ffmpeg环境）。完整地图教师仍未通过原窄桩。

## 2026-10-18 18:32：arrival_v5_probe 有界探针

非零课程恢复原生验证通过：64环境，EMA注入0.37、次数7，恢复后两轮所有值保持一致；普通两轮+恢复两轮也完成。证据服务器 outputs/moe_perceptive/v4_resume_smoke/sentinel_check.json。

新增可选 --arrival-aware（旧V4默认奖励不变）：目标速度在距目标0.8m内逐渐降低，最大0.6m/s；进度改为朝目标投影速度并在近处衰减。原成功标准未放宽。旧奖励固定支付全局前进，存在接近目标仍获推进奖励的冲突；robot0交错诊断在x>3.3m的33帧平均水平速度约0.76m/s，不能外推全部失败。

已启动 moe_perceptive.arrival_trial，控制输出 outputs/moe_perceptive/arrival_v5_probe，训练 /root/gpufree-data/MoE-runs/arrival_v5_probe，512环境仅200轮，从V4 model1999初始化、全新优化器与课程。这是探索探针，并非严格隔离变量消融。完成后自动执行原九项严格suite、两类宽桩视频/落脚诊断，然后停止，不自动加轮数。仍为特权教师，不能计为原窄桩成功。

## 2026-10-18 18:42：探针失败，进入数值诊断

arrival_v5_probe在第54轮报 `normal expects all elements of std >= 0.0`，只有model0，未完成200轮。启动器异常返回0使控制器误进入缺失model199的评估；已停止进程组并写invalid_evaluation.json，所有这些下游尝试无效，不计通过率。arrival_trial新增完成标记、失败标记、指定权重和评估输出验证。不能把零退出码视为成功。

新增 --numeric-debug：观察/rollout有限值检查与autograd异常追踪，不做NaN替换或强行续训。已从相同初始权重启动512环境最多60轮的数值复现，输出 /root/gpufree-data/MoE-runs/arrival_numeric_debug，日志 outputs/moe_perceptive/arrival_numeric_debug/driver.log。这只是故障定位，不是新一轮成功训练。当前没有有效V5通过率。

## 2026-10-18 18:47：数值复现未复现原异常

numeric-debug完整跑完60轮，有限值检查与autograd异常检测均未触发；这不证明原异常已修复。已从model59保存的优化器、课程、随机状态恢复，追加最多140轮至总200轮；活动路径 /root/gpufree-data/MoE-runs/arrival_debug_cont140，控制日志 outputs/moe_perceptive/arrival_debug_cont140/driver.log。不再使用失败arrival_v5_probe的评估。下一步仅在completed标记存在、failed不存在、model199确实存在后运行严格suite与宽桩诊断。

## 2026-10-18 19:14：arrival诊断200轮完整评估

outputs/moe_perceptive/arrival_debug_eval 已完成九项原场景与两类宽桩诊断。原桩六组仍全部0/32；平地32/32、楼梯32/32、高台25/32；宽双列2/32、宽交错26/32。对比初始化V4末轮宽双列4/32、宽交错6/32，交错改善但双列未改善；本轮包含fresh optimizer/课程重启及数值debug，不能将变化严格归因于单一奖励。

robot0交错在x>3.3m区间85帧平均水平速度约0.371m/s（旧诊断0.760），最终位置(3.78,−0.156,0.118)m；双列仍在约(2.45,−0.084,0.116)m失败，未进入终点区间。当前基座动作与残差修正平均绝对差双列0.431、交错0.383，最大1.04/1.09（normalized action，乘0.25为关节rad）；并非零残差，也没有证据表明需要单纯扩大动作上限。原始失败数值原因仍未复现、未修复确认。下一步检查双列实际摆脚落点、支撑边缘和接触奖励，不追加同配置训练。

## 2026-10-18 19:24：双列robot0落脚线索

arrival_debug_eval/footfall_analysis.json记录实际足力触发样本。双列robot0在2.12s左前脚约(2.461,−0.028,0.023)m处出现119N竖直力，但其足部link中心对应高度格为深坑；该点靠双列中缝，球形足端可能接触边缘，不能称为坑底支撑。此后机身高度由约0.35m降至0.21m并最终触地。实际反馈指向支撑边缘附近的脚位与稳定性，非单纯停稳问题。

当前supported_contact仅在首次接触时奖励3×3、6cm足下支撑比例；没有对将要落入空区的摆动脚提前提供直接训练信号。下一步可做独立、可选的下降低脚安全落脚奖励探针，仍由RL输出关节动作，不加入IK纠正；需先验证观测/奖励时间对齐和地形边缘容差，不能直接宣称根因或效果。

## 2026-10-18 19:34：可选下降脚风险奖励短验证

新增 foothold.py 与 --foothold-aware，限显式teacher-map实验：仅足部link高度在桩面相对[−0.05,0.16]m、竖直速度<−0.05m/s时，按足下6cm九点扫描缺失支撑比例施加−2权重成本。高摆动/向上/已深落脚不触发；没有IK或落点动作纠正。它仍是特权训练奖励，尚无有效性证据。

64环境两轮原生smoke输出 outputs/moe_perceptive/foothold_smoke/run，日志同父目录driver.log；从arrival_debug_cont140/model199初始化。完成后先检查非有限值和奖励分布，再决定有界训练，不重启V4。

## 2026-10-18 19:44：foothold_v6_probe 启动

64环境两轮原生短验证完成且无failed；奖励项已注册，但两轮Episode统计为0，尚不能证明风险信号覆盖失败落脚，正式探针须检查其分布与实际触发。已启动foothold_trial，512环境200轮，从arrival_debug_cont140/model199初始化、全新优化器/课程，保留arrival-aware并开启numeric-debug。训练 /root/gpufree-data/MoE-runs/foothold_v6_probe；控制 outputs/moe_perceptive/foothold_v6_probe，PID625095。结束后自动九项严格suite与两类宽桩诊断；所有阶段要求输出标记存在，失败即停。只是一轮有界探索，不自动增加训练块。

## 2026-10-18 20:14：foothold_v6完整结果，不继续原配置

200轮正常完成，原六组窄桩仍0/32；平地/楼梯/高台各32/32。宽双列1/32、宽交错18/32，对照初始化arrival结果2/32、26/32没有改善。robot0在双列恰好完成，不能用其单视频代表总体。新增风险奖励确实触发，训练末段Episode_Reward约−0.005至−0.009，但触发不代表学到稳定落脚。控制器evaluations_completed，没有活动训练。

不追加本配置轮数。下一步需验证地图注意力是否对落脚产生有效影响：相同重置种子的正常地图/地图遮蔽或置换消融，并统计动作差与原尺寸任务结果，区分感知分支无效、策略利用错误和足端控制能力不足。消融必须明确是诊断，不把缺地图测试作为部署结果；继续保留最好dense checkpoint与已有宽交错checkpoint。

## 2026-10-18 20:24：地图依赖诊断启动

输出 outputs/moe_perceptive/map_dependence_v6，固定v6 model199、seed801与32环境；双列宽前置difficulty0及原difficulty1，各做正常地图和全未知地图两组。新增evaluate --map-probe记录每个尚未结束的首回合状态下正常/未知地图的配对动作差；--map-ablation unknown仅遮蔽actor的height/mask/age，保留本体历史与目标。默认评估路径不变。遮蔽回合明确diagnostic_only，不是部署成绩。动作敏感不等于有效利用；完整回合比较也受分布外输入和物理随机性影响。

## 2026-10-18 20:34：地图消融seed801完成

| 双列地形 | 正常地图 | 全未知地图 |
|---|---:|---:|
| 宽前置difficulty0 | 1/32 | 5/32 |
| 原difficulty1 | 0/32 | 0/32 |

正常地图访问状态下，遮蔽地图带来normalized action平均绝对差宽桩0.229、原桩0.307（乘0.25约0.057/0.077rad）。所以地形分支确实影响动作，但目前没有成功率收益证据；不能仅因未知地图5>1就断言感知有害，未知观测是分布外且仅单种子。已启动seed802/803、同四条件的八组重复评估，输出 outputs/moe_perceptive/map_dependence_v6_repeat，避免从单次结果做结构结论。

## 2026-10-18 20:44：地图消融三种子汇总

重置seed801/802/803，每组32环境，固定同一地形布局与v6权重。宽双列正常地图分别1/1/2，合计4/96；遮蔽分别5/1/5，合计11/96。原尺寸双列正常与遮蔽均0/96。三种子不是独立训练种子或地形布局泛化。结果未展示地图带来成功率收益；正常与未知配对动作差非零，说明感知分支在改变动作，但不证明学到了正确的落脚关系。遮蔽分布外等限制仍在，不能据此证明传感器无用。

下一步不重复调同类奖励。优先检查地图到每条腿的空间对应与注意力分布、坐标运动对齐、控制动作和实际脚位的关系；现模型四个注意力头为通用proprio查询，没有显式每腿落脚目标。保留MoE并评估加入明确足端空间约束/中间落脚表示的方案，实施前先建立可复核的诊断，不把未验证结构写成既成能力。

## 2026-10-18 20:54：运动地图对齐复核

新增 audit_motion_map.py，从v6两类宽桩robot0动作前记录逐帧重建yaw对齐33×21网格，与解析地形比较，遵循高度编码clip，排除高度边缘一格范围。双列31411/31411、交错24352/24352样本误差均小于2mm。该检查覆盖运动和偏航，不再仅限初始姿态；但只覆盖这两条robot0轨迹、特权地图和非边缘区域，未验证机载传感器。当前证据未发现这些轨迹的地图整体偏移/转置，注意力与脚位空间关联仍待诊断。

## 2026-10-18 21:04：注意力原生诊断

attention_diagnostic.py只读重建当前TerrainAttention权重，逐帧assert重建输出与实际模块一致，记录四个通用头的XY质心、近脚区域质量、空token质量、支持高度区域质量与熵。evaluate --diagnostic现增加attention字段；没有动作修正。已启动 outputs/moe_perceptive/attention_v6/wide_stakes，32环境、双列宽前置、v6同权重seed801。头并非每腿指定，质心不是落脚指令；等待实际分布后再决定空间约束。

## 2026-10-18 21:14：注意力分布证据

attention_v6/wide_stakes诊断完成，重建attention输出与原模块逐帧assert一致，robot0共68帧。四头平均熵5.99/6.46/6.46/6.43，694token均匀分布熵约6.54；近脚矩形质量约25–27%，空token约0.1%，多数头关注分散。初始四头XY质心约x0.24–0.28、y接近0，尚无明确左右前后分工；质心与熵不能单独证明注意力错误，但结合地图遮蔽未损害成功率，现有通用汇聚缺少足端空间对应是有证据支持的结构诊断方向。

下一步拟新增可选足端空间先验attention实验：四头对四腿名义落脚邻域施加软距离偏置，保留全局token和原MoE，不做IK；旧模型默认行为必须保持不变，初始化、保存、恢复、TorchScript和原生短训练均须验证。固定名义邻域不是实际足端预测，需清楚标注；训练后仍以原窄桩与dense保留评估判断，不以注意力变尖锐作为成功。

## 2026-10-18 21:24：四腿空间先验实现与兼容验证

TerrainAttention新增默认关闭的spatial_prior；四头名义中心FL(.25,.15)、FR(.25,−.15)、RL(−.2,.15)、RR(−.2,−.15)m，sigma0.22m的软距离偏置，仍保留所有地图token与null。不是实时实际脚位、不是落脚规划或IK。--spatial-attention写入训练来源并禁止恢复时改变；evaluate从checkpoint来源自动恢复该开关。空间偏置是可重建非持久buffer，不改变旧权重state键。

服务器接口验证通过：旧v6 checkpoint严格加载、非有限值检查、attention重建与模块一致、TorchScript保存加载输出一致；合成输入动作最大变化0.049 normalized。输出 outputs/moe_perceptive/spatial_v7_smoke/compatibility.json。64环境两轮原生smoke已启动在同目录run，尚无有效过桩结论。

## 2026-10-18 21:34：spatial_v7_probe启动

原生64环境两轮完成，无failed。空间先验200轮探针已启动，512环境、初始化v6model199、fresh optimizer/课程，保留arrival/foothold奖励与numeric-debug；run /root/gpufree-data/MoE-runs/spatial_v7_probe，控制 outputs/moe_perceptive/spatial_v7_probe，PID673202。自动原九项suite及两类宽桩视频诊断后停止。仍无新原桩通过证据。

## 2026-10-18 21:56：v5/v6及消融完整归档

完整评估、八段视频、两份选定权重、三种子地图消融、运动地图与注意力诊断见 [perception_diagnostics_v6](artifacts/perception_diagnostics_v6)。1314文件在服务器与本地SHA/大小一致，服务器另验证1125源码引用及八段视频帧数。复核 `python -m moe_perceptive.verify --root artifacts/perception_diagnostics_v6 --manifest-name archive_manifest.json`。它们均不是原窄桩成功证据。

spatial_v7已完成200轮且无failed；九项strict已完成，原桩六组仍0/32，平地32/32、楼梯30/32、高台26/32；宽双列0/32，宽交错诊断仍在进行。暂无继续同配置训练的依据。

补记：v7宽交错最终15/32，控制器evaluations_completed；宽双列0/32，相比v6的1/32、18/32未改善，dense也下降。停止追加该结构配置。v5/v6完整诊断和当前源码已发布5f5fc7f，v7完整归档尚待补充。

## 2026-10-18 22:06：复查原论文，修正结构归因与训练预算判断

复读已有deepxiv提取的AME方法原文：其query长度也是1，由本体embedding查询地图；h=16、d=64，并未要求每头对应一条腿。因此先前“没有明确每腿分工”只能作诊断现象，不能认定违反AME结构或为失败根因；v7空间先验不是论文原方法，未显示收益，应停止。

原论文ANYmal训练为4096环境、stage1 18000轮＋stage2 3600轮（约6天A100）；这不是Go2保证，也不能直接搬其耗时。当前200轮×512环境探针仅约stage1环境轮积的1/720；即使V4 2000×512也约1/72（未在此比较每轮实际步数，不能直接称同精度sample ratio）。因此短探针失败不充分证明模型结构不可行，更不应每200轮更换结构导致反复fresh optimizer/课程。

服务器AME-criticaldamped/go2/runner_cfg.py的Go2RunnerCfg是此前适配配置（3000轮），不是作者验证的Go2配方。后续需区分原论文、原仓库已有任务及本地Go2改造；检查阶段1基础速度训练是否足够、原AME两阶段与地形课程、actor/critic共享encoder和探索强度差异，形成固定配置的预算与验收门槛后再启动，不盲目延长失败v7。

## 用户暂停（2026-10-18）

用户要求暂存任务，停止继续探索。自动化go2核实PAUSED，服务器当前无moe_perceptive训练/评估/trial进程。不再启动实验，待用户明确恢复。当前权重、代码、原始输出均保留；v7完整归档上传仍待办，具体路径见handover.md顶部。
