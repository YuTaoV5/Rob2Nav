# 最新交接：2026-10-21（本节优先于下方历史记录）

## SCAN动态多房间与指定人物跟随（2026-10-21）

导航页默认SCAN，新增相连六房间、三个移动碰撞人物、家具跟随、低阶梯、刚性沙地近似及综合场景；可选择人物01/02/03、跟随距离和难度，重置后开始，支持旋转/平移/缩放。使用已知静态地图A*、实际MuJoCo点云、固定社区SCAN及MoE关节控制；动态场景加入明确的`rob2nav_dynamic_map_v1`占用老化补丁（2秒、静态场景默认关闭）、可见仿真ID跟踪与近人预测保护。慢/常规领路人会等待，非任意人群单方避障证明。

最终v8快速多房间seed201到达，移动8.861m；综合人物01跟随seed200移动9.491m，低阶梯和沙地近似均进入/退出，基座最高z=0.466m，平均跟随距离误差0.345m，无碰撞/跌倒。旧版停滞、v7地形出口不完整均保留。指定人物不可见时零速度、不换身份；网页选人和拖动视角通过；77项Studio测试、最终3项任务测试及前端构建通过；补丁后原静态两场景均到达。详见[任务说明](docs/SCAN_DYNAMIC_FOLLOW_TASKS.md)和[原始证据](artifacts/scan_tasks_20261021/README.md)。

验证限于固定场景：人物不是视觉ReID；阶梯最高0.10m，沙地为刚性粗糙低摩擦近似；尚无标准楼梯/真实软土或长期鲁棒性验收。实机不能照搬占用老化配置，需按手册完成真实观测、地形运动策略和独立安全链。未连接实机、未新增导航训练权重。网页与本机8766隧道已恢复，默认多房间READY。

## 标准站姿首尾专项训练（进行中，2026-10-21）

续训补充：侧翻v3中途权重初段站姿改善但只翻半周；单纯推理时降低桥接残差仍失败。训练与推理共用的新桥接残差合约已实现，v4从仍能全周旋转的v2检查点排队训练，完成后四种子独立验收。后翻v2仍排队、直立v2仍训练，13项测试通过。全部新候选保持隔离，未更新网站默认权重。

最新后翻v2：新增1秒站姿准备、同步延迟的60维相位/奖励时钟和训练推理一致的有界输出，短程训练/CPU导出/5秒MuJoCo执行完成但仍未过能力验收。正式1024环境1000更新通过`scripts/train_backflip_fixstand_v2.sh`排队等待当前侧翻v3退出后启动；完成后9300–9303评估。侧翻v3与直立v2的真实训练进程已复查存活。12项单元测试通过，网站仍未替换未达标权重。

后续更新：第一轮三项500更新均完成，各4个种子全部不通过。后翻末段站姿已有单种子通过（最大误差0.0851rad），但初段、身体触地和轴纯度失败。侧翻v3正在续训500更新，边界残差可配置为0.20rad；直立v2新增平滑阶段关节目标、真实迈步奖励与严格双后足承重周期验收，先32环境2更新再1024环境1500更新，脚本`scripts/train_upright_fixstand_v2.sh`。11项测试通过；无新权重获准更新网站，详细状态以服务器任务和[专项手册](skills/FIXSTAND_ROUNDTRIP.md)为准。

新增[训练目标、状态与操作](skills/FIXSTAND_ROUNDTRIP.md)：侧翻69维新参考首尾各1秒站立、禁止随机相位起步；后翻60维加强首0.35秒及末段站姿、延长到4秒并取消随机episode时钟；直立从48扩为52维阶段输入，100%标准四足起步，学习起立后回四足。32环境2更新三项均产生模型并完成CPU/MuJoCo接口评估，但新的首尾联合验收全部未通过，不能覆盖部署权重。

服务器 `outputs/fixstand_finetune_20261021` 三项各500更新/1024环境串行训练，独立CPU监控在每项结束后按6100–6103评估。首项侧翻5699已完成但0/4，直立继续训练、后翻排队。另启动有界参考残差的侧翻v2（`outputs/fixstand_sideflip_residual_v2_20261021`，500更新/1024环境，完成后7100–7103评估）；v2短程恢复完整旋转但有身体触地和约0.16rad端点误差，仍未通过。当前所有新候选均未获实机资格，旧默认权重保留。运行中的任务不应被写作最终交付通过。

## 紧凑热切换与导航服务恢复（2026-10-21）

仿真中stand/MoE/sideflip组合支持低速稳定后直接进入目标入口，按姿态差调整入口时长，在ENTRY提前更新新策略历史，并增加实际Kp/Kd收敛检查；不满足条件保留中立路径，入口超时只恢复一次。硬件禁用紧凑路径。普通配对序列从约24.6秒降至18.7秒；侧翻组合从44.616秒降至35.406秒，最终v3为35.650秒，交接力矩步进≤0.300001Nm。侧翻总关节运动量没有下降、速度仍超额定，不获实机资格。MLP及低刚度混合回归新旧路径均失败，记录保留，网页默认隐藏到实验性技能入口。详见[完整说明](docs/COMPACT_HOT_SWITCH.md)和[原始对照摘要](artifacts/compact_switch_20261021/README.md)。

导航卡住原因为supervisor退出后仅恢复了Studio，8768/8769物理worker未启动，随后本地8766隧道也退出。现已恢复全部五项supervisor进程和隧道，网页看到约1×实时PPO导航及目标到达；导航/SCAN页增加明确重连提示并在恢复后清除连接错误。新增仓库启动脚本会恢复全部服务，不只启动网页与托爪。容器重启后仍需由宿主调用启动脚本。

## 工作台视频与创建实验训练配方（2026-10-21）

工作台技能视频限制高度、两栏等宽；手册、权重和验证记录可展开查看，避免长内容拉高左列。创建实验现可选择地形MoE、导航PPO、侧空翻、站立行走和后空翻；专用配方预填手册预算和种子，展示复现步骤，并开放通用任务参数、技能参数及支持的断点路径。导航使用MuJoCo/SB3训练器，三个动作技能使用固定Isaac训练器；服务端只接受登记的选项及项目内检查点，提交前检查上游源码提交、参考动作和解释器。训练产物完成与能力验收仍分开，专用技能不会自动宣称独立仿真通过。详见[实验配方说明](docs/STUDIO_EXPERIMENT_RECIPES.md)。

服务器Studio已重建前端并重启API，健康检查通过；工作台两栏截图与侧空翻、导航配方在网页中核对，二者预检通过。服务器32项Studio相关测试通过，前端生产构建通过。本次没有提交训练任务，也没有连接实机。

## D435i / MID360安装标定与真实托爪方案（2026-10-21）

新增[传感器安装与标定总手册](docs/GO2_D435I_MID360_INSTALL_CALIBRATION.md)及[真实相机托爪流程](go2_nav/paw_contact/REAL_CAMERA_DEPLOYMENT.md)，覆盖供电/安装、ROS驱动、坐标/时间、固定相机外参、60°视野、掌面RGB-D定位、LIO/SCAN、真实支撑与柔顺接触、导航/托爪控制权交接及现场验收。明确当前托爪尚无真实相机识别闭环，MID360并非原地托爪必需，60°不替代外参标定。另附未标定模板、仅合成60°示例和离线几何/holdout检查工具；不采集传感器、不连接机器人、不会授予硬件资格。


## 网页编排与服务器恢复（最新）

研究服务器Studio已恢复在 `127.0.0.1:8765`，本机SSH隧道 `http://127.0.0.1:8766` 健康检查通过。部署页新增可拖拽的统一技能时间线、最多4轮循环、显式侧翻研究开关、实际MuJoCo实时画面与相机控制、序列JSON导出；前端生产构建和服务器路由测试8/8通过。网页没有实机电机控制接口。

从网页新路由实跑三组：短序列 `stand→walk_moe→stand` 3次提交通过；两轮组合 `stand→MoE→sideflip→MoE→stand→MoE→sideflip→MoE→stand` 9次提交、两次侧翻、92.821秒实时帧通过，累计滚转-12.7132rad、交接力矩步进≤0.300001Nm，但最大关节速度/额定=1.0850，仍**不获实机资格**。第三组在打包精确Go2模型后从仓库内XML/16网格重新编译，短序列再次通过。侧翻必需参考NPZ、模型、MIT许可与选择的策略权重均纳入仓库；原始summary及序列见[网页编排证据](artifacts/composer_20261021/README.md)；逐周期NPZ/视频可按[完整操作与实机指南](docs/UNIFIED_GO2_DEPLOYMENT.md)重算。实机指南新增机载电脑连接、接口核对、键盘速度意图和全状态机说明。实际设备未连接、未发送LowCmd。

## 最终缓存回归补充：v14（本节优先）

新增引用句柄与预加载记录的有界缓存，避免长时间反复翻转后耗尽文件描述符。v14共用序列完成stand→MoE→sideflip→MoE→sideflip→MoE→stand共7次激活；两个显式触发、两个倒置后回正区间均核对，累计滚转-12.7138977rad、交接命令力矩步长≤0.300001Nm、保留记录3。18项核心测试通过，当前全部部署源码与冻结v14 AST一致（包括CLI）。新的重复序列文件和操作命令已入库。最高速度比1.08440仍超额定包络，未放行实机；长时设备可靠性、硬实时及广泛扰动鲁棒性仍需另验。


## 统一部署后续验证：v13（本节优先，2026-10-21）

已补齐共用`SkillSequence`设备服务接入、原子动作显式trigger、加载期间转向后的入口yaw对齐、发送失败检查、原始固件version/导入SDK来源检查、完整编译模型指纹。按技能契约设置承重模型正则：普通入口0、侧翻入口2，避免低Kp稳态误差与翻转入口卸载问题互相回归。最新17项核心测试、62项工作台测试通过；SDK真实IDL/CRC及固件变化拒绝、模型质量变化指纹检查均通过，没有打开机器人DDS。

v12/v13五配置soft→stiff→MLP→MoE→stand两次通过；共用序列stand→MoE→sideflip5200→MoE→stand在0°、+30°、-30°三个固定初始朝向完成，约一周翻转，交接命令力矩步长≤0.300001Nm、Kp步长≤0.030001。最高速度比分别约1.0842/1.0851/1.0993，**动作流程通过但电机速度安全不通过**，全部hardware_qualified仍false。不是随机地形/载荷或实机鲁棒性证明。

逐周期NPZ、视频、成功/失败摘要、v10–v13确切源码及测试记录补入[统一部署归档](artifacts/unified_deployment_20261021/README.md)，见[逐项验收数据](artifacts/unified_deployment_20261021/release_acceptance.json)。当前控制源码与v13 AST一致；CLI仅计划展示另增入口姿态差与ARMED说明。v10/11失败仍保留，不沿用旧结果覆盖新状态。[完整部署方案](docs/UNIFIED_GO2_DEPLOYMENT.md)已更新序列运行命令与设备阶段准入。未连接机器人，也未重新部署网页服务器。


## 发布补充：资源完整性与统一部署验证（2026-10-21，本节优先）

本次将剩余修改、选定权重索引和完整文档数据一并提交。`artifacts/release_20261021/`保留597个原始文件、265,722,077字节；更新的README/模型索引重新生成发布校验值，原始manifest和原始文件不变。这里的“选定/最佳候选”只代表记录中的选择依据，不代表全局最优或实机合格。后空翻4999、侧翻DC5200及候选/失败模型、地形199、导航PPO、日志、配置、评估轨迹与视频均有路径和SHA索引。

新证据见 [统一部署归档](artifacts/unified_deployment_20261021/README.md)：冻结v9在源机器人模型上完成一次stand→MoE→侧翻5200→MoE→stand连续自由基座MuJoCo循环（seed9304，5次提交，积分滚转-6.3597rad）。受控交接最大力矩步长0.3001Nm；原子动作本身最大力矩步长44.6787Nm，最高关节速度比1.08188，**仍不通过额定速度安全验收**。同版本RoboGauge模型seed9209在stand_soft入口超时、0提交；旧seed9205虽完成5配置切换，但参考滤波抑制步态，只是历史集成验证。成功、失败、逐周期NPZ及视频全部保留，不宣称多种子鲁棒通过。

归档另含确切v9源码/配置与哈希清单、真实SDK IDL/CRC离线结果；当前工作源码有之后的推理节拍、速度命令过期、缓存保护及序列编排增量，不自动继承v9物理验证。最新本机核心测试9/9、工作台62/62、前端生产构建通过。新增网页统一部署页仅生成计划，无电机执行接口；本次不声称该新增页面已经重新部署至服务器。

所有hardware_qualified仍为false，未连接机器人、未发送电机命令。后续需修复stand_soft入口、补多种子/模型匹配切换、原子动作速度包络、当前源码完整物理回归，再进行独立看门狗/标定/实际设备准入。既有训练中断状态及SCAN/托爪等能力限制以下文各节为准。


## 工作台与模型工作流更新（2026-10-21）

工作台技能点击改为左侧原位视频和详情，新增手册/权重详情与独立能力矩阵定位入口；在线仿真可选择任务队列的最佳候选和其他保存点。新增独立纯推理 API，保留训练关闭状态、共用GPU锁，并使用匹配的原生Isaac运行时。模型清单补齐ONNX/ZIP/基础MoE，默认区分14项核心文件与历史中间权重，并展示配套配置和匹配哈希的评估。既有阶梯199检查点隔离运行6秒、151帧、输入哈希一致；不创建训练，不冒充新技能验收。完整操作与验证见 [工作流手册](docs/STUDIO_MODEL_WORKFLOW.md)。

## SCAN 导航手册补充（2026-10-21）

新增 [SCAN仿真操作与复现](skills/navigation/SCAN_SIMULATION.md)和[Go2完整导航实机部署方案](skills/navigation/SCAN_DEPLOYMENT.md)，导航页和工作台手册入口已接入。明确区分运动学模拟、MuJoCo真接触仿真与实机；覆盖传感器/LIO、坐标时间契约、SCAN启动、全局路线、Sport SDK速度安全门、分阶段验收、运维与网页硬件接入。当前实机传感器适配、LIO和速度执行桥仍待实现/联调；没有执行机器人通信或运动，不把此方案标为已通过实机部署。

## 本次发布与网站

用户要求启动网站、更新handover，并将现有代码、成果、特技选定权重、文档和数据上传到 https://github.com/YuTaoV5/Rob2Nav 。发布仓库为本目录；上层X:/Items/data_fusion不是该Git仓库。所有验证与权重按各自哈希记录，不把完成训练等同于技能或实机通过。

网站访问： http://127.0.0.1:8766/#navigation 。本次发现服务器容器已重启，原网站、隧道及训练进程均不在；已通过部署脚本恢复5个Supervisor服务，并重建本地8766 SSH隧道。检查返回网站health=200/ok、导航READY、托爪WAIT。隧道凭据只在进程内存，不写入本仓库。

```bash
# 服务器恢复网站，不启动训练
bash /root/gpufree-data/deployment/start-studio.sh
supervisorctl -c /root/gpufree-data/deployment/rob2nav-supervisord.conf status
# 本机交互式建立访问隧道（根据提示输入凭据）
ssh -N -L 8766:127.0.0.1:8765 -p 30752 root@120.209.70.195
```

## 当前真实完成程度

| 能力/功能 | 已执行结果 | 不能宣称的部分 |
|---|---|---|
| 网页技能部署 | 在线仿真下拉选择保留技能，实际MuJoCo实时渲染；6项启动和旋转/平移/缩放、完成后检查相机通过 | 启动成功不等于每个技能动作通过；地形入口用RoboGauge预训练MoE，不能冒称Isaac微调权重迁移 |
| SCAN网页 | 固定ROS2 community提交原生规划/跟踪→真实MuJoCo点云、里程计与MoE；空旷/绕障2/2到达，无碰撞/跌倒；修复前1/2保留 | 两个固定布局不是广泛鲁棒性或实机验证；不是原生运动学模拟器冒充接触动力学 |
| 左右爪托手 | 拖虚拟手，左右爪接触、撤手、停止、推扰；74项检查；新增相机交互 | 没有真实相机/手掌/机器人测试 |
| 后空翻 | 自训5000轮，源MuJoCo完成翻转落稳；提供4999检查点及ONNX/PT | 严格落地与电机边界失败，没有完整验收合格的实机权重 |
| 侧空翻 | 自训5000轮后DC适配，固定5200在独立种子3100–3107动作8/8 | 最高额定速度比1.08505，速度门限0/8；5998仅未验证候选 |
| 站立行走 | 5000轮结束；PhysX起立前移，但后足无真实换步并伴滑移；MuJoCo失败 | 没有可称为最佳合格行走的模型；不能用位移代替迈步 |
| 地形手册复跑 | stairs32/32；platform18/32；multiterrain flat31/32、stairs32/32、platform32/32 | 固定seed801与场景，不是全地形或全局最优 |
| 统一Go2部署框架 | 契约、后台模型加载、切换状态机、统一PD/力矩输出、SDK编解码及门禁原型；8项核心检查、官方SDK消息/CRC离线通过 | 首轮MuJoCo中立获取超时，提交次数0；不宣称MLP/MoE热切换或实机通过；全部hardware_qualified=false |

网页与手册：[实时技能](docs/LIVE_SKILL_SIMULATION.md)、[SCAN](docs/SCAN_WEB_SIMULATION.md)、[统一部署验证中方案](docs/UNIFIED_GO2_DEPLOYMENT.md)、[能力状态](docs/CAPABILITY_REPRODUCTION_STATUS.md)。研究报告页收录docs、skills及托爪手册；7能力各3份手册21个链接已实测。

## 权重、日志、视频和数据

新增完整归档：`artifacts/release_20261021/`。原始服务器快照597文件、265,722,077字节，缺失0、排除0；附加验证与索引均在该目录。不要把下载用的145MB ZIP重复提交；仓库保存已解开的普通Git资源，每个文件均小于95MiB，不依赖LFS。

- 总索引：[README](artifacts/release_20261021/README.md)、[模型选择及SHA](artifacts/release_20261021/model_index.json)、[原始文件清单](artifacts/release_20261021/manifest.json)。
- 后空翻：`weights/backflip/selected/model_4999.pt`、`exported/policy.onnx`及`policy.pt`。
- 侧空翻选定：`weights/sideflip/selected/model_5200.pt`和`exported/policy.pt/.json`，策略SHA `7616e53e2c95434f824c2728621f8052e2d41a4e033fd00a25f316de211e07e1`。原5000与DC5998候选分目录。
- 直立行走：`weights/upright_walk/representative_failed_baseline/`；DC5998候选与新增迈步奖励5700中断快照均单独保留，均未放行。
- `weights/{stairs,platform,multiterrain}/manual_reproduction/model_199.pt`含各自配置、源清单、迭代日志。
- `evaluations/`包含本次特技成功/失败诊断、8个侧翻holdout完整trace、导航、SCAN、地形、托爪和浏览器记录；`logs/`含5000轮与适配训练日志；`validation/`含SDK检查、核心检查和切换失败轨迹/视频。
- 既有导航PPO在`skills/navigation/policy/navigation.zip`；RoboGauge本体与旧留存权重仍在原vendor/artifacts位置，模型索引含其哈希。
- 外部无根许可声明的CSV/源码缓存、虚拟环境、node_modules和凭据不打包；参考动作按训练手册固定URL/SHA另行准备。

```bash
python scripts/verify_release_resources.py
# 部署离线检查，不初始化DDS
python -m rob2nav.deployment profiles
python -m rob2nav.deployment plan --from walk_moe --to backflip60
```

## 容器重启后的训练状态

原backflip/sideflip/upright各5000轮均已完整结束。侧翻DC1000轮和直立DC1000轮适配均结束，最终5998检查点存在。新增`upright_stepping_full_20261021`原预算1000轮，最后保存`model_5700.pt`，没有`completion.json`，没有终态exit文件，当前无训练进程：**容器重启中断，不是完成**。日志最后约37,453,824 timesteps。此次只恢复网站，没有重复开训/自动续训。

## 统一部署框架下一步（不能跳过）

当前原型位于`rob2nav/deployment/`，配置在`deployment/skills/`。新的目标是统一Go2 Python SDK、安全权重热切换，尤其避免PD、默认姿态、关节顺序、相位和历史差异造成冲击。协议检查使用独立`go2-sdk-check`环境、SDK814556d与CycloneDDS0.10.2；仅消息构造/CRC，未初始化DDS或创建发布器。

真实MuJoCo首轮seed9201失败：中立获取超时、3501步、0提交，最大相邻力矩变化约0.0694N·m、kp变化0.03、倾斜9.67°、最低高度0.301m。先分析四足支撑、姿态/重力补偿和退出过程，不仅放宽门槛。原报告backend文字是测试计划，实际未执行到MLP/MoE提交。

后续需补多种子实际权重热切换、异步加载失败回滚、推理停顿与低层恢复、符号/零位/IMU标定及设备健康闭环、48维线速度来源、69维参考yaw对齐，以及真实实时性/独立看门狗与硬件资格。现有原型含部分标定/健康门禁代码及native_pd、运动学观测、--atomic-cycle研究仿真分支；这些开发增量尚未完成完整物理验证。不可因文件存在或8项逻辑测试通过而放行实机。仍遵守原范围：实机仅方案与离线检查，不向真实机器人发送电机指令。

---

<details>
<summary>2026-10-20及更早历史交接（旧状态可能已过期；以上最新节优先）</summary>

# 最新工作台整合（2026-10-20）

持续迭代新增（2026-10-20早间）：混合任务72f9bb2d17384900检查时仍训练中，未重复启动。新增步态测量 `--gait-metrics`，原生8环境1秒烟测已通过，全部8回合有指标，输出 `outputs/gait_smoke_native`，PID260772为已结束烟测，不是训练。短时结果显式不计任务成功率。suite默认采集，可用 `--no-gait-metrics`做旧测量对照；当前训练进程不变，后续评估子进程使用新suite。网页新增按成功/失败分组的步态诊断，并允许无视频的指标记录。定义及未测量边界见docs/GAIT_METRICS.md。35项测试通过。下一步等500轮完整评估，再对留存基线与新权重做802/803独立种子匹配，不能仅靠训练地形等级升高续训或宣称步态更好。

最新上线：独立 `#deployment` 和 `#navigation` 页面、核心 `rob2nav` CLI/契约/源策略后端已部署。导航页可提交现有MoE规则导航、查看结果视频，以及运行参考SE(2)几何规划/固定α CBF测试；SCAN与SEA原生适配仍未接通。服务器Ubuntu24.04/ROS Jazzy，无Docker命令；不能直接按Noetic步骤覆盖系统。审计发现SEA 240°射线与CBF默认180°不一致，且未带导航权重，后续需修正并测试。详见 `docs/FRAMEWORK_NAVIGATION.md`。

高台任务 `f1478e77469949e1` 已完成，dense三项各32/32、原桩六项各0/32。新任务 `72f9bb2d17384900` 是从该权重初始化的混合技能500轮、冻结本体、原观测/rig、seed74，已提交；先检查真实进程/完成标志/评估，不要重复启动。导航网页验收任务 `09f6f49cf6114151` 已执行完成。新核心CLI后空翻2秒烟测 `outputs/framework_smoke/backflip_v2` 已通过，不等于完整翻转验收。

自动化 `go2` 已由PAUSED更新为ACTIVE，每20分钟继续本聊天任务。初始每设计5000轮审查、通常每500轮评估；独立匹配评估连续3块无进步或保持能力退化时诊断换方案，不靠单一训练奖励宣称最优。`moe_perceptive.suite` 新增可显式选择 `--seed`（默认仍801），后续使用802/803等独立重置种子及布局，补齐步态/碰撞/能耗指标。当前尚未证明步态最优或所有任务完成。

最终更新：`be9053facb374533` 九场景评估完成，平地/阶梯/高台各32/32，六组原桩各0/32；已归档选定权重和完整记录。当前新任务 `f1478e77469949e1` 正在受控高台训练，不要重复启动。LSD首次三个任务因MuJoCo解析API不兼容失败；已修正，并通过独立CPU执行完成三个6秒记录，视频/轨迹归档 `artifacts/capability_integration/lsd_*`。后空翻单次结束直立，侧空翻倒置落地失败，均不代表多种子技能验收。下文“正在九场景评估/源动作排队”是本轮较早快照。

用户新授权将多个开源项目统一到网站，并开展对应能力训练；旧窄桩盲目探索仍暂停，不能重启V3/V4/V7。新增注册表、能力矩阵、项目审计、机器人/策略合约和训练方案见 `docs/UNIFIED_CAPABILITIES.md`。外部无根许可证的源码/权重只保存在服务器相邻 `Rob2Nav-upstreams` 缓存，禁止打包到公开Git。

PIE源ONNX已完成10秒真实深度MuJoCo楼梯推理：前进4.448m，最低直立度0.9951，500帧；是未独立评分的推理记录，不能宣称通过率。证据归档 `artifacts/capability_integration/pie_source_inference`。第一轮新阶梯微调退化，9场景证据归档 `rejected_stairs_v1`，不得推广权重。

新对照任务 `be9053facb374533`：保留V2观测、原传感器rig、冻结本体，200轮已训练完，当前在九场景评估。先检查真实任务状态再行动。随后排队的源动作推理：jump `f1a3bd8e7576472c`、backflip `7a5498a1f6514464`、sideflip `b9ea43a73b5b40e0`。不要重复启动。所有记录经Studio任务队列保存。

原生MoE训练可执行；PIE源推理可执行；LSD源推理待本轮验证；其他原生训练器仍是明确标注的待接入方案。PIE原生训练CLI可加载但尚未训练验证。部署、版本和后续验收门槛见上述整合文档，以下旧快照不代表当前无任务。

---

# 用户暂停状态（2026-10-18，历史快照）

> 新增工作台交付：用户随后要求实现Rob2Nav Studio，网页开发已进行；原机器人探索暂停仍有效。服务器Studio仅监听127.0.0.1:8765，本机SSH隧道访问127.0.0.1:8766。网页支持显式任务提交，但没有自动恢复训练。只运行了2秒MuJoCo网页链路验收，不进行新策略训练。使用见studio/README.md，验收边界见docs/STUDIO_VALIDATION.md。不要把网站交付当成原窄桩能力达成。

用户明确要求暂存并停止继续探索。自动化go2已核实为PAUSED；服务器未发现moe_perceptive训练、评估或trial进程。除非用户明确恢复，不启动任何实验。

最新v7已结束：六组原桩0/32，dense平地/楼梯/高台32/30/26，宽双列0/32、交错15/32。权重/root/gpufree-data/MoE-runs/spatial_v7_probe/model_199.pt，完整原始输出/root/gpufree-data/Rob2Nav/outputs/moe_perceptive/spatial_v7_probe。该轮完整视频/权重尚未归档上传；不要删除服务器输出。此前v5/v6、消融及源码已发布5f5fc7f。

恢复时先读V4_VALIDATION.md末尾的方法复核：原AME不要求注意力每头对应一腿，200轮短探针不充分证明结构不可行；后续计划尚未定稿，不应直接重启v7。核心原窄桩、机载学生、MuJoCo导航均未完成。

---

# Rob2Nav 项目交接文档

> 最新状态（2026-10-18 21:56）：V3/V4、arrival和foothold训练均结束。空间注意力spatial_v7已完成200轮，strict六组原桩仍0/32，dense32/30/26，宽双列0/32，交错评估尚在进行。v5/v6完整归档见artifacts/perception_diagnostics_v6。先读V4_VALIDATION.md末尾，不重启旧任务。


> 更新时间：2026-10-18 13:17（Asia/Shanghai）；服务器核查时间为 05:17 UTC。
> 这是交接时的状态快照，不是实时监控。接手后先查询服务器，再决定操作。
> **任务尚未完成：原 AME 窄梅花桩仍未通过。不要把训练完成、浅坑课程升级或教师输入当作最终感知越障能力。**

## 最新覆盖更新：已切换V4（2026-10-18）

**下文V3“正在运行”段落保留为历史快照，现已过期。** 用户随后批准新方案，V3控制器和第四块训练已停止，保留完整`model_3500.pt`。目前活动任务为 `moe_perceptive.deep_trial`，控制目录 `outputs/moe_perceptive/wide_deep_v4`，训练前缀 `/root/gpufree-data/MoE-runs/wide_deep_v4_chunkN`，512环境、每块500轮、初始最多5块。

V4所有课程等级保持真实2m坑，支撑面1m→20cm、起始小间隙，增加偏航约束和更严格的本级成功晋级。每块进行原九项严格评估，另列宽桩precursor，新增任意等级平台期检测。教师/学生界限不变；仍无原窄桩成功结论。

同一旧权重的浅坑/深坑诊断已完成：中等难度深坑0/32、浅坑2/32；robot0有明显偏航，未观察到其低于平台0.1m的足部接触。初始地图对齐632个非边界样本全部正确，不能再将“坑底依赖”或“地图转置”写成已证实原因。V4四轮原生短训练已通过且正式训练已启动。

接手先读 [V4执行与诊断报告](V4_VALIDATION.md)，再读本文历史背景。不得重启旧V3控制器；自动化`go2`也已改为执行新方案。V3完整三个gate及V4诊断/启动已归档在 `artifacts/v4_redesign/`。

## 1. 先读这一页：现状和下一步

1. 当前主线是：**复用 RoboGauge Go2 预训练 MoE → 加入地形注意力 → 先训练完整地图教师 → 再训练可见相机/雷达学生 → MuJoCo 验证 → 接回目标点导航**。
2. 前两轮可见感知 MoE 实验已结束，均未学会原尺寸窄桩。第三轮改成完整地图教师，并增加目标输入、目标奖励和浅坑起步课程。
3. 第三轮累计已完成 2000 轮和两次完整严格评估；两次全部六组桩测试均为 **0/32**。当前第三个训练块正在运行，核查时全局迭代号 **2384**，约 4059 环境步/秒。
4. 控制器仍会自动完成当前块、评估，然后按条件继续，最多累计 5000 轮。**接手时不要另起同名任务，不要重复占用 GPU。**
5. 重要疑点：浅坑课程推进到等级 5–6，但原深坑仍全部失败。需要实际分析落脚、支撑接触、地图坐标和奖励，不应只增加相同配置的轮数。坑底依赖目前只是待验证假设。
6. 机载感知学生蒸馏尚未实现；新主线尚未做 MuJoCo 迁移或实机验证。旧项目中有其他学生与 MuJoCo 成绩，不能挪用为本轮成果。
7. 本聊天有自动化 `go2`，每 **5 分钟**检查一次，用户要求重要变化主动通知。换 AI 时先确认自动化归属，避免两个 AI 同时改训练或启动进程。

## 2. 用户目标、偏好和已授权事项

### 最终目标

- 平台为 Unitree Go2；计划使用 D435i 和 Mid360。
- 给定目标点，无需提前建图，在线利用感知主动绕障、提前抬腿上阶梯、爬高台、跨坑，并实现原 AME 那种窄梅花桩精确落脚。
- 当前范围以**仿真实现**为主；本体控制要参考/复用 RoboGauge，感知与稀疏落脚参考 AME 等开源工作。
- 需要可复现代码、选取权重、完整仿真视频、README、使用说明及包含失败结果的评估报告。
- 用户明确允许持续迭代至完成；已要求先用 arXiv 技能调研其他工作及 GitHub 实现。
- 用户实机将按仿真结果安装传感器，因此需要给出有动态证据的安装建议。目前仅有候选，不能宣布最优安装已定。

### 已授权操作

- 在服务器上安装必要依赖、修改代码、训练、评估、导出及整理结果。
- 将项目和上述产物上传到 `https://github.com/YuTaoV5/Rob2Nav.git`。
- 持续检查任务并主动通知重要变化，无需反复询问是否继续。

### 沟通要求

- 用中文，先给实际结果；直接说明失败和能力边界。
- 重要事件包括：训练块完成、完整评估完成、首次原窄桩成功、明显退化、崩溃或停滞、设计变更、进入学生或 MuJoCo 阶段。
- 普通轮数增长不应每次通知；同一事件不要重复通知。
- 早前给出的 1–2 小时是单轮运行时间估算，不是技能收敛或项目交付保证。文献训练规模大得多。

## 3. 仓库、路径与计算环境

| 项目 | 位置/信息 |
|---|---|
| GitHub | `https://github.com/YuTaoV5/Rob2Nav.git` |
| 真正的本地项目 | `X:\Items\data_fusion\.publish\Rob2Nav` |
| 默认工作区 | `X:\Items\data_fusion`，不是本项目根目录，不要误提交其其他项目 |
| 本地分支 | `main`；本文之前最新已发布提交 `d2d0392f97168c2859b1dd02f71a0cb4e918deb1` |
| SSH | `ssh -p 30249 root@120.209.70.195` |
| 服务器工作目录 | `/root/gpufree-data/Rob2Nav` |
| 独立验证克隆 | `/root/gpufree-data/Rob2Nav-perceptive-63f5bcf`；版本可能落后，需先检查 |
| 原始第三方 AME | `/root/gpufree-data/AME-criticaldamped`，固定版本 `15d02c891273d3f4c9b55c9afe3073bcd2d9152f` |
| MoE 上游研究副本 | `/root/gpufree-data/MoE-upstream`，版本 `30e74dc507bec7a642a8c98be26081f2c6f0822d` |
| 原生训练入口 | `/root/isaacsim/python.sh` |
| 原生环境 | Isaac Sim 5.1 / Isaac Lab 2.3.2 / PyTorch 2.7+cu128 / RSL-RL 3.1.2 |
| 轻量分析环境 | `/root/gpufree-data/envs/go2-mapless/bin/python`，包含 CPU Torch、imageio/ffmpeg、matplotlib 等 |
| GPU/资源 | NVIDIA L40，约 46 GB 显存；CPU 配额 14 核，内存约 120 GB |

**密码不写入本文、脚本、Git、日志或自动化提示词。** 用户已在原聊天提供；无该上下文的 AI 应通过安全方式取得凭据，不能猜测或从其他文件搜集秘密。

服务器工作树曾通过 SFTP 单独上传源文件，不能假定其 Git 状态等同于 GitHub HEAD。训练启动时的 `source/` 与源文件哈希比当前工作树更可靠。不要在正在训练的目录盲目 `git pull`、重置或覆盖代码。

### SSH/SFTP 辅助工具

本地可运行：

```powershell
cd X:\Items\data_fusion\.publish\Rob2Nav
python scripts\remote_session.py 120.209.70.195 --port 30249
```

工具通过 `getpass` 提示密码，不保存密码；要求主机已在本地 OpenSSH `known_hosts` 中，并拒绝未知主机。建立连接后输入 JSON 行：

```json
{"op":"run","command":"nvidia-smi"}
{"op":"get","remote":"/root/gpufree-data/Rob2Nav/outputs/example.json","local":"X:/Items/data_fusion/.publish/Rob2Nav/outputs/example.json"}
{"op":"put","local":"X:/Items/data_fusion/.publish/Rob2Nav/handover.md","remote":"/root/gpufree-data/Rob2Nav/handover.md"}
{"op":"exit"}
```

此前终端工具的 session ID 是临时会话，不是可移交凭据或稳定接口。会话失效时重新连接即可。控制器使用独立 POSIX 会话运行，不依赖 SSH 终端存活。

## 4. 接手后第一组检查（只读）

在服务器执行：

```bash
cd /root/gpufree-data/Rob2Nav
cat outputs/moe_perceptive/nav_teacher_v3/status.json
ps -eo pid,etime,args | grep 'moe_perceptive\.' | grep -v grep
nvidia-smi
python3 - <<'PY'
from pathlib import Path
import json
for p in sorted(Path('/root/gpufree-data/MoE-runs').glob('nav_teacher_v3_chunk*/status.json')):
    d=json.loads(p.read_text())
    print(p.parent.name, d['iteration'], d['steps_per_s'])
    print('sparse histograms', d['terrain_histograms'][4:])
    print('completed', (p.parent/'completed.json').exists(),
          'failed', (p.parent/'failed.json').exists())
root=Path('outputs/moe_perceptive/nav_teacher_v3')
for p in sorted(root.glob('gate_chunk*/*/summary.json')):
    d=json.loads(p.read_text())
    print(p.parent.parent.name, p.parent.name, d['successes'], d['total'],
          sum(e['failure'] for e in d['episodes']))
PY
```

如果状态 JSON 与进程不一致，检查日志尾部、文件更新时间、`failed.json` 和返回码；不能仅凭旧 `state=training` 宣称还在训练。不要杀其他进程；确需停止当前方案时，先保存检查点并确认控制器及子进程关系，避免控制器自动启动下一块。

## 5. 当前正在运行的任务：第三轮完整地图教师

### 控制器与记录

- 控制程序：`moe_perceptive/teacher_trial.py`。
- 控制目录：`/root/gpufree-data/Rob2Nav/outputs/moe_perceptive/nav_teacher_v3`。
- 控制日志：`outputs/moe_perceptive/nav_teacher_v3_controller.log`。
- 启动记录：`outputs/moe_perceptive/launch_nav_teacher_v3.json`。
- 训练记录：`/root/gpufree-data/MoE-runs/nav_teacher_v3_chunkN`。
- 严格评估：`outputs/moe_perceptive/nav_teacher_v3/gate_chunkN`。
- 核查时控制器 PID `314810`；当前 chunk3 的 Python PID `414525`，其 shell PID `414520`。PID 仅供快照对照，操作前重新验证。

### 训练配置

- 1024 环境，seed 42；每块增加 1000 轮，最多 5 块。
- chunk1 从第一轮 `visible_moe_v1_seed42/model_999.pt` 初始化 policy 与 critic；不使用停滞更严重的第二轮作为起点。
- 各阶段边界均为 0，直接 joint；新参数学习率 `2e-4`、原 actor `2e-5`、原历史编码器 `1e-5`。
- 新增 4 维目标输入：截断到 2 m 的相对 XY，以及航向误差 sin/cos；零初始化投影到感知融合路径。
- 教师直接读取理想高程/有效性/年龄形式的地形，不运行相机和雷达。**这不是机载感知学生。**
- 训练中浅坑深度为 `0.02 + 1.98*difficulty**2` m；由 `gentle_pit=True` 显式启用，仅改变训练坑底，不更改目标桩面布局。
- 新奖励包含朝目标运动、前进、终点接近、完成奖励、支撑接触等；去掉原来的固定速度和零转速跟踪密集奖励，增加停滞终止。
- 桩课程使用成功率 EMA 推进，避免每次失败都回到最低级。
- 严格评估 `training=False`，不启用浅坑，回到原 AME 2 m 坑；本体物理、关节映射和控制缩放保持明确记录。

### 当前训练进展

核查时 chunk3 全局迭代号 2384；chunk1、chunk2 分别在 999、1999 完成。第三块尚未完成，无其最终严格评估。

课程直方图索引是 0–9。双列桩主要停在 4–5，交错桩主要在 6，尚未到最高级。直方图只描述当前训练地形分配，不是严格通过率。

### 两次已完成的严格评估

每项固定布局、seed 801、32 个初始扰动首回合。

| 地形 | difficulty | 1000 轮通过 | 2000 轮通过 |
|---|---:|---:|---:|
| 平地 | 0 | 32/32 | 32/32 |
| 楼梯 | 0.5 | 32/32 | 31/32 |
| 高台 | 0.5 | 27/32 | 30/32 |
| 双列桩 | 0、0.5、1，各自 | 全部 0/32 | 全部 0/32 |
| 交错桩 | 0、0.5、1，各自 | 全部 0/32 | 全部 0/32 |

两次六组桩测试均每组 32/32 失败终止，不是“站着完成”。1000 轮时终态平均前进约 1.3 m。2000 轮的具体距离、接触、轨迹和视频请读实际文件，不补造数值。

**已知自动停止逻辑的局限：** `teacher_trial.py` 所谓连续两块无进展，只检查所有稀疏地形是否完全留在 level0；它不检测 level5/6 平台期。因此现在即使严格评估连续为零也会继续跑。接手 AI 应诊断真实平台期，不要把控制器继续运行当作“算法仍在有效学习”的证据。

### 控制器后续行为

每块结束后运行完整九项 suite 并录制最窄桩视频；若所有九项通过率均≥90%，输出 `teacher_gate_passed` 后结束教师实验。否则继续下一块，最多五块。过教师门槛后还需要学生、独立种子和 MuJoCo，不能标记整体完成。

## 6. 历史实验：不要混淆成绩

### A. 原 MoE 与第一轮感知改造

- 实际初始文件：`vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt`。
- SHA256：`2206659a6a23886446dcfc05d86888d6cdc9b8a4e31f44cbbc4e2d17ba3d6ead`。
- 原文件中 actor 和 student history encoder 可完整恢复；没有原 critic、训练优化器和随机数状态。不要称为恢复完整原训练会话。
- 当前上游 MoE 架构与该 137k 权重有差别；本项目按实际部署文件严格重建，不能直接用新上游类加载旧参数。
- 第一轮 run：`/root/gpufree-data/MoE-runs/visible_moe_v1_seed42`，512 环境、1000 轮，1 秒可见观测记忆；依次 value/fusion/actor/joint。
- 第一轮最终权重 `model_999.pt`；公开归档 `artifacts/moe_perceptive/`。
- 原 MoE 平地/楼梯 32/32、高台31/32；新策略三者均32/32。所有六组桩均0/32。
- 最窄桩第一轮失败数增加；权重未替换稳定导航控制器。
- 早期双列桩基线视频有多环境可视模型叠加，仅作失败诊断；后续视频仅显示 robot0，统计仍为32环境。

### B. 第二轮 4 秒记忆适配

- run：`/root/gpufree-data/MoE-runs/visible_moe_memory4_v2_seed42`。
- 从第一轮 policy/critic 初始化，新优化器与新课程；512 环境额外1000轮，全部 joint。
- 控制目录：`outputs/moe_perceptive/memory4_v2`；已完成。
- 公开结果：`artifacts/moe_memory_v2_end/`，选取 `model_999.pt`。
- 平地/楼梯/高台各32/32；六组桩仍各0/32。最窄桩平均只前进约0.43m，主要停在平台超时，跌倒减少不能算技能提高。
- `artifacts/moe_memory_v2_start/` 是历史启动快照，不能据其 training 状态判断现在仍在训练。

### C. 第三方 criticaldamped AME Go2 从零复现（历史对照）

- 包：`critical_ame/`；原始上游保持固定版本。
- 第一阶段累计3000有效更新，最终 `/root/gpufree-data/AME-runs/stage1_1000_3000/model_2999.pt`。
- 分类 gate：stairs_down2/32、stairs_up27/32、slope_down19/32、slope_up32/32、boxes12/32、rough28/32、steppingstones0/32、gaps0/32。
- 第一阶段 gate 未过，第二阶段没有正式训练。直接测试原双列/交错桩仍0/32。
- 另有外部 MLP 教师步态初始化、BC、PPO、critic warmup 等对照，均需按独立实验报告；没有产生窄桩成功权重。
- `gait` 后续 warm-PPO 曾中断，最新保存模型100、未完成计划轮数；旧状态文件可能残留 running，不能视为活动任务。
- 公开归档：`artifacts/critical_ame/`；详见三份 critical AME 报告。

### D. 更早的其他分支

- `go2_nav/`：MuJoCo 反应式无地图导航、主动台阶修正、技能调度及 NJU 跳坑集成。固定 course曾3/3；不同目标有失败，非任意目标可靠导航。
- `perceptive_rl/`：早期理想地形教师、学生蒸馏、AME 增量实验。早期宽 stones 有成功成绩，但**不是原20cm窄桩**。
- 旧学生 MuJoCo 楼梯4/5、高台5/5、宽 stones2/5；不得作为当前 MoE 或原窄桩结果。
- 旧 NJU 学习跳跃、手写 IK/感知动作修正与新 RL 权重是不同机制，不要称为“原 MoE 某个跳跃专家自动完成”。

## 7. 代码地图和关键接口

| 文件/目录 | 作用 |
|---|---|
| `moe_perceptive/model.py` | 实际137k MoE重建、地形CNN/注意力、残差融合、可选目标投影、分阶段冻结 |
| `moe_perceptive/mdp.py` | 原MoE观测顺序、可见地形、理想地形、奖励/终止、关节限位 |
| `moe_perceptive/memory.py` | 5cm的33×21高度/有效性/年龄地图，仅保存已见返回，按理想位姿重投影 |
| `moe_perceptive/task.py` | Isaac Lab场景、传感器、物理、地形比例及基础奖励 |
| `moe_perceptive/navigation.py` | 第三轮目标输入、教师地图、目标奖励、停滞、课程、浅坑开关 |
| `moe_perceptive/rig.py` | 候选安装、深度最小Z、雷达近距与保守机身遮挡过滤 |
| `moe_perceptive/train.py` | PPO、初始化/续训、参数检查、优化器、原子检查点及来源快照 |
| `moe_perceptive/evaluate.py` | 首回合评估、执行前地图诊断、原生视频、actor导出 |
| `moe_perceptive/suite.py` | 九项固定测试；支持显式教师/目标/安装配置 |
| `moe_perceptive/teacher_trial.py` | 当前分块教师控制器 |
| `moe_perceptive/pipeline.py`、`memory_trial.py` | 第一、二轮历史控制器，不是当前活动控制器 |
| `moe_perceptive/mount_audit.py` | 17个安装候选的几何可见性扫描，不是动态过桩 |
| `moe_perceptive/analyze_diagnostic.py` | robot0足边地图覆盖统计与绘图 |
| `moe_perceptive/provenance.py` | 启动时源码/依赖快照、参数、版本、初始权重SHA |
| `moe_perceptive/archive.py`、`verify.py` | 第一二轮风格归档、哈希/源码/回合/视频检查 |
| `perceptive_rl/ame_terrain.py` | 已验证的原AME双列/交错桩几何与网格量化 |
| `perceptive_rl/terrain.py` | Isaac地形适配；本轮增加默认关闭的gentle_pit |
| `perceptive_rl/geometry.py` | 与MuJoCo共用几何入口 |

### 本体控制与网络

- 关节顺序：FL、FR、RL、RR，每条腿 hip/thigh/calf。
- 默认关节角：`[.1,.8,-1.5, -.1,.8,-1.5, .1,1.,-1.5, -.1,1.,-1.5]`。
- PD：20/0.5；动作缩放0.25；控制周期0.02s。上部关节力矩23.7Nm，小腿45.43Nm，限位与部署XML明确匹配。
- 原MoE输入：5×45=225；历史编码为8专家混合的32维归一化隐变量；actor输入77，输出12。
- 地形：3×693，高度/掩码/年龄；地图范围x[-.55,1.05]、y[-.5,.5]m，5cm格。
- v1/v2 actor观测2304，critic3693；v3加4维目标后分别2308/3697。
- 新目标投影零初始化；旧critic输入列复制，新增4列置零；原权重与接口不被静默替换。
- 冻结 `anchor` 始终保存原MoE参考。最终 joint 会更新实际base actor和历史编码器，并非永远只加一个小残差。

## 8. 传感器研究：结论和未完成事项

完整资料见 [SENSOR_RESEARCH.md](SENSOR_RESEARCH.md)。

- 原安装：相机相对base `(0.25,0,0.06)m`、向下20°；Mid360 `(0,0,0.16)m`、水平正装。
- 正装Mid360垂直视场−7°～52°；高0.5m时最近平地射线约4.07m，远超当前局部图。因此不能说原桩场景中雷达提供了稠密脚下信息。
- 当前优先候选 `near_down60`：D435i相对base `(0.30,0,0.06)m`、向下60°，镜头伸出机身前沿。base高0.34m时镜头高约0.40m。
- 倒装备选 `near_down60_front_lidar`：Mid360前伸 `(0.30,0,0.12)m`、roll180°。倒装改善下视角，但机身会遮挡部分方向，不能只改旋转就称为最佳安装。
- 原生候选探针仍0/32；它是旧权重零样本改变安装/记忆/过滤的消融，不是新训练成功策略。
- robot0足边已知比例从旧约34/34/17/15%提高到新约49/46/42/45%，但时间样本不同，不是总体统计或信息完全充分的证明。
- 简化深度64×32射线、雷达96×12射线、10Hz，均不是完整真实传感器扫描模型。
- 使用理想里程计；未知格保持未知。当前无真实外参、时间同步、里程计漂移、完整腿/支架遮挡、传感器载荷动力学和实机验证。
- 非original候选包含0.195m深度光轴最小Z、0.2m雷达可靠近距、保守机身包围盒过滤；original保留旧实验合同，因此新旧安装消融不只是单独改变俯角。
- 不提前建图允许在线局部地图与时间记忆，不意味着不需要状态估计。

## 9. 论文、开源依据与技能

已使用本机 `deepxiv-cli` 技能，位于 `C:\Users\YuTao\.codex\skills\deepxiv-cli\SKILL.md`。本机命令在 `X:\SoftWare\Conda\Scripts\deepxiv.exe`。

已实际检索/阅读：

- AME：arXiv `2506.09588`，`--section Method`。
- AME-2：`2601.08485`，`--section "IV Locomotion Control"`。
- Learning Agile Locomotion on Risky Terrains：`2311.10484`，`--section Method`。
- Extreme Parkour：`2309.14341`，`--head`。
- Robot Parkour Learning：`2309.05665`，`--brief`。
- Egocentric Vision：`2211.07638`，`--brief`。

开源核对：

- [官方 AME-2](https://github.com/leggedrobotics/ame2_minimal)，本地 `.research/ame2_minimal`，版本`8beb9a688d67882828019583f7d5a78e3517f233`。
- 该版本公开G1教师，神经地图模型/完整学生未发布；不存在本次已验证的现成Go2窄桩权重。
- [AME作者调参指南](https://github.com/zita-ch/techblogs/blob/main/2026-03-28-AME%20Tuning%20Guide.md)。
- [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour)、[Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour)。
- AME-2官方库GPL-3.0；当前只在忽略目录核对方法，没有复制其GPL实现进本仓库。未来若直接复用要处理许可。

原始研究文本在忽略目录 `.research/perceptive_v3/`，没有上传论文全文。Windows调用deepxiv时设置 `$env:PYTHONIOENCODING='utf-8'`，否则部分符号会触发GBK编码错误。受限网络失败时使用经批准的网络方式，不反复重试同一失败调用；不要输出CLI token。

## 10. 接下来要做的事情（按优先级）

### P0：解释严格评估连续失败

1. 获取chunk1和chunk2六组桩的完整首回合视频、`trace.json`、失败终止项。先定位第一次错误落脚，区分踏空、滑边、跨距不足、偏航、俯仰或小腿接触；不要凭课程直方图猜结论。
2. 补录真实足底接触、四足位置/速度、相位、所选地图位置、理想高程及策略动作。验证是否在训练中依赖浅坑底，记录坑底接触次数与高度。
3. **核实地图坐标与展平顺序。** 当前33×21使用y行/x列；理想height_scanner带x=0.25m偏置，配置`ordering='xy'`。应在已知单桩几何下检查每个坐标与射线真值，不能因训练可运行就当成对齐正确。
4. 比较同一checkpoint在浅坑与原深坑上的动作/足端/接触分布；固定起始状态、只改坑底，隔离课程迁移问题。浅坑通过率单独报告。
5. 检查奖励实际量级和终止时序：完成奖励、末段目标接近、first-contact支撑项、anchor、非足接触和停滞。避免仍奖励错误支撑、错误完成或保守停滞。
6. 检查控制权限、动作边界、关节可达范围与首个目标桩位置；先确认Go2的目标步幅/横向跨度可行，再调网络。

这些是排查计划，不是已经证实的bug。若需要调整，建立新版本目录和原生短验证，不覆盖旧运行或伪装成同配置续训。

### P1：根据证据改训练

- 如果浅坑导致坑底依赖，修改课程，逐步训练安全支撑而非靠坑底前进；可考虑更细的首步难度、较易真实稀疏通用地形、再迁移原布局。
- 如果是感知/坐标问题先修接口，不用奖励掩盖；教师失败时不要先花大算力做学生。
- 如果明显平台期，应停当前配置供诊断；控制器自身的level0停止条件不够。
- 若引入新探索、对称性、落脚目标或模型辅助，明确作为新实验，和原端到端/IK方案区分。
- 保留原MoE平地、楼梯、高台的对照，避免过桩未成而本体能力先退化。

### P2：教师通过后进入感知学生

- 实现动作/地形表示蒸馏与学生PPO，目前`moe_perceptive`主线没有完整现成学生蒸馏程序。
- 可参考旧`perceptive_rl`蒸馏流程，但网络、观测、目标输入、关节顺序必须逐项对齐，不能直接混用检查点。
- 学生仅用实际可见返回和可部署本体状态，不能静默读取理想地图、坑底真值或特权接触。
- 加入遮挡、时延、丢点、近距、姿态/外参扰动、记忆和里程计误差；评估安装候选的任务级效果。
- teacher-map切换到student不是精确`--resume`；用明确初始化/蒸馏协议并保存来源。

### P3：MuJoCo与导航闭环

- 新主线的2308维目标观测、历史、地图、PD、关节映射和物理约束尚未完整接到MuJoCo。
- 先验证同状态动作等价，再做同几何动力学迁移；两者不能混为一谈。
- 通过单项后接回目标点导航、障碍绕行、台阶、高台、深坑技能。`navigation.goal()`当前固定目标约`(3.6,0)`，基础route监督器在x>3.5置零速度，**尚非通用任意目标导航**。
- 选择完整任务视频而非只展示成功帧，保留失败、重试及回退机制。

## 11. 验收与评估规范

- 最终必须为原尺寸20cm方桩、原排列和2m深坑；双列与交错均需完整通过。
- 名义间距与5cm量化后的碰撞几何不同，详见`NARROW_VALIDATION.md`；不要重新将宽stones当成目标场景。
- 当前完成：首回合沿+x超过3.5m，随后≥2足接触>5N、直立>0.9、速度<0.25m/s，持续≥0.3s，无失败终止。
- 32环境目前是同一固定布局的不同初始扰动，不是32种独立地形。默认评估seed801，XY±3cm、yaw±0.04rad。
- 后续至少加入多个独立重置种子（如801/802/803），再区分不同地形布局泛化。地形内部也有固定seed，不能只改reset seed就声称新布局。
- ≥90%是当前内部推进门槛，不是已经满足的用户验收。教师达标仍不是学生达标，Isaac达标仍不是MuJoCo/实机达标。
- 回合成功、实际跌倒、停滞超时和离开路线必须分开统计。优先报告完整通过率与原始记录，不用平均奖励代替能力。

## 12. 运行、续训与归档注意事项

### 读取与手动评估

```bash
cd /root/gpufree-data/Rob2Nav
MOE=vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt
/root/isaacsim/python.sh -m moe_perceptive.evaluate --headless \
  --moe "$MOE" \
  --checkpoint /root/gpufree-data/MoE-runs/nav_teacher_v3_chunk2/model_1999.pt \
  --navigation --teacher-map --kind ame_stakes --difficulty 1 \
  --num-envs 32 --seed 802 --video --output outputs/new_unique_teacher_eval
```

这是教师评估示例，不是学生部署命令。输出目录必须未存在；不要与正在运行的suite重复评估/竞争资源。

### 初始化和续训区别

- `--initialize-from`：只迁移policy/critic，新Adam、新RNG、新课程、新回合；v1/v2→v3只允许明确新增的目标投影和critic列迁移。
- `--resume`：恢复权重、优化器、RNG、课程、下一轮编号；seed、环境数、kind、阶段边界、memory、navigation、teacher_map和rig需一致；不恢复物理瞬时状态。
- 检查点`iter`为下一轮编号；resume后`--iterations 1000`是增加1000轮。例如chunk2从1000训练到1999。
- RSL `ActorCritic.load_state_dict`返回bool，不能解包missing/unexpected；迁移检查使用`torch.nn.Module.load_state_dict`。该问题已在原生smoke中修复。
- 模型训练前做原生短验证即可，不在大规模训练中反复跑相同检查。修改有新风险时再补对应检查。

### 已完成的关键验证

- 4路64步、局部reset下原MoE重建最大动作差约1e-5；零融合时逐值相同。
- 1秒记忆的未知、重投影、过期、reset、重复返回更新时间测试通过。
- 原生四阶段短训练验证冻结/解冻张量变化，anchor不变。
- v3目标适配器初始零等价和TorchScript导出通过；原生四轮教师训练通过。
- 安装扫描的最近交点、反向未命中和空隙真值检查通过；新安装原生32环境诊断已完成但未通过桩。

### 归档陷阱

- `archive.py`只识别pipeline下`baseline/`、`trained/`结构，**当前teacher_trial使用`gate_chunkN/`，不能直接调用旧归档器就宣称完整归档。** 下个AI需写适配器或显式复制每个完整gate。
- 只复制返回码成功且MP4 writer关闭的评估，不能归档正在写的视频。
- 选取必要检查点，保留精确源码、参数、版本、哈希、日志、回合记录和完整视频，避免把所有大权重都塞进Git。
- `.gitattributes`对归档目录设置`-text -whitespace`，防止Windows换行改变来源哈希；新增归档目录也要处理。
- `python -m moe_perceptive.verify --root artifacts/moe_memory_v2_end`可检查旧标准归档的文件SHA、源快照、计数、视频帧数。
- 第三轮chunk1/2的最终权重和gate目前主要在服务器；GitHub只有启动快照及研究材料，**完整第三轮结果归档/上传仍是待办**。

## 13. 自动检查与通知交接

- 自动化ID：`go2`；名称：`Go2 感知越障持续迭代`；ACTIVE，5分钟一次。
- 绑定本聊天thread：`01a1479b-d9f6-7a00-93e0-f6b8949936fa`。
- 本机配置：`C:\Users\YuTao\.codex\automations\go2\automation.toml`；通过Codex automation工具查看/修改，不直接编辑配置文件。
- 英文prompt用于避免PowerShell输出编码损坏，要求中文通知。曾发生旧中文提示经shell读取后变成乱码，已整段重写修复；以后读JSON要保证UTF-8。
- 提示中要求使用忽略文件`outputs/notification_state.json`记录已通知事件；接手时检查是否存在。此前状态查询大多未写该文件，不要假定去重记录完整。
- 用户已经得知chunk1、chunk2完整评估均窄桩0/32，以及chunk3已开始；不要把这两次旧结果再次当新突破通知。
- 换到新聊天/新AI时需协调现有自动化，避免旧聊天与新AI同时改文件、停进程、启动任务。可以保留单一监控方，明确谁拥有执行权。
- 用户未要求停止后台训练或自动化。本文交接不意味着暂停训练；任务真正完成后才关闭自动化。

## 14. 文档与产物索引

建议接手顺序：

1. 本文 → [当前实验报告](MOE_PERCEPTIVE_VALIDATION.md) → [论文、安装与第三轮研究](SENSOR_RESEARCH.md)。
2. [MoE主线结构/命令](docs/MOE_PERCEPTIVE.md)与`moe_perceptive`源码。
3. [原AME尺寸说明](NARROW_VALIDATION.md)。
4. [第三方AME结果](CRITICAL_AME_VALIDATION.md)、[复现说明](docs/CRITICAL_AME_REPRODUCTION.md)、[外部教师对照](docs/CRITICAL_AME_GAIT_INITIALIZATION.md)。
5. [旧主动导航](docs/ACTIVE_NAVIGATION.md)、[旧主动导航评估](ACTIVE_VALIDATION.md)。
6. [更早学生实验](PERCEPTIVE_VALIDATION.md)、[更早AME增量实验](AME_VALIDATION.md)，只作历史对照。

主要公开产物目录：

- `artifacts/moe_perceptive/`：第一轮最终结果、原MoE基线、原始视频与model999。
- `artifacts/moe_memory_v2_end/`：第二轮最终权重、九项结果、窄桩失败/停滞视频。
- `artifacts/moe_memory_diagnostic/`：1秒/4秒零样本记忆消融。
- `artifacts/sensor_research_v3/`：17安装候选扫描、新安装原生探针、失败/修复后的smoke与第三轮启动源码快照。
- `artifacts/critical_ame/`：历史从零AME与初始化对照。
- `outputs/`、`.research/`、tar.gz：默认忽略，服务器/本地存在不代表GitHub已上传。

### 上传操作

本地先检查只提交本项目授权文件。Windows Git推送环境曾需要：

```powershell
cd X:\Items\data_fusion\.publish\Rob2Nav
$env:PATH = 'C:\Program Files\Git\usr\bin;C:\Program Files\Git\mingw64\bin;' + $env:PATH
git status --short
git diff --check
# add、commit后：
git push origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Git写入/联网在当前Codex沙箱可能需要工具层提权，用户已授权上传。先完成可审阅产物，不反复询问同一上传权限。不要输出凭据。

## 15. 最重要的事实边界

- 已有可运行的模拟与训练链路，**还没有成功复现原窄桩**。
- 完整地图教师仍失败，说明不能把所有问题推给D435i/Mid360安装。
- 新安装有覆盖改善证据，但不是最终最佳支架方案或实机可用证明。
- 浅坑课程升级不是原2m深坑成功；教师不是学生；宽stones不是原20cm桩。
- 仍需深入诊断，而不是自动等待最多5000轮后把“完成运行”写成“完成任务”。

## 状态覆盖（2026-10-18 17:57）

V4已完成2000轮并自动停在 `needs_diagnosis`，第五块没有启动。四次原桩六组始终0/32；宽交错前置从第三块22/32退化至第四块6/32。接手先读 V4_VALIDATION.md 最后更新及服务器 v4_plateau_diagnostic，不得按旧“活动控制器”描述重启V4或V3。下一步诊断与重设计；学生蒸馏、独立种子和MuJoCo仍未完成。





### Studio导航与可视化更新（2026-10-18）
Studio主入口已重组，研究基准/模型/报告集中到左下图标对应的研究中心；任务队列移至实验页签。新增只读Go2运动学三维查看、本地URDF+OBJ/STL导入与真实训练生成器地形预览（含20cm原尺寸桩终点）。详细使用与实现边界见studio/README.md。模型查看、预览不启动训练；原自主探索保持暂停。本次改动不恢复V3/V4/V7，也不代表原AME越桩已完成。


### 2026-10-19 网站部署资料提交范围调整
用户允许不上传中间训练产物，当前交付改为Git主仓库所需源码/依赖/部署文档及既有精选模型、曲线、评估和视频。先前准备的full-archive-2026-10-19没有发布，Release草稿没有附件，服务器准备好的分包不删除、不继续上传，也不作为网站依赖。README、DEPLOYMENT和DATA_CATALOG已纠正相关说明；docs/release-summary.json明确标记server_archive_only_not_published。

Git资料独立目录检查通过：18条训练记录、272组评估、93视频、25模型入口、28报告；页面/模型/地形接口200、视频Range206、真实曲线可读、空任务队列、只读执行模式。核对记录见docs/site-deployment-check.json与site-data-summary.json。前端dist需新环境npm ci/npm run build重建。训练探索继续暂停。本次不改写Git历史、不删除已提交精选资料。

## 2026-10-20 实验设计页改造
新增 docs/EXPERIMENT_DESIGNER.md：完整奖励解释、模式有效权重、方法路线及接入边界、阶段预设，以及可选固定机身/摆腿高度训练目标。已通过38项测试和Isaac 8环境2迭代执行短测；这是功能验证而非能力提升。不要把固定目标当成已有权重运行时指令。旧worker 250737已退出，原72f9bb2d17384900完成499轮并已有评估目录；本轮未重启该训练。后续研究按真实评估继续。

## 2026-10-20 08:06 保持能力评估
72f9bb2d17384900的500轮及九项严格评估已完成：seed801平地/阶梯/高台各32/32，原窄桩六组各0/32；不能声称提升。核实无活动训练后启动匹配独立seed802复评：平台model199与混合model499依次各9项，启用gait metrics，记忆4秒、original安装不变。服务器outputs/moe_perceptive/retention_independent_802，driver PID278704已核实正在原生evaluate。下次先查driver日志、各summary及completed，勿重复启动；比较滑移/功率等代理与通过率后决定是否继续训练。本次不增加训练轮数。

## 2026-10-20 仿真/队列信息架构调整
按用户要求将在线仿真配置收进运行新仿真弹窗，来源限制为队列训练任务检查点；默认在线页签，与历史回放分开。候选权重跳转打开预选弹窗，不自动启动。实验/曲线与任务队列合并，保留历史日志展开入口，能力与方案移到底部图标。仅前端静态部署，不重启GPU任务。38项测试及前端构建通过。

## 2026-10-20 08:27 独立种子802完成
18项匹配评估完成。平台对照平地/阶梯/高台32/32各；混合500平地31/32、阶梯高台32/32各，双方六窄桩均0/32。共同成功回合支撑足链速度代理混合增16–22%、机械功率略低；尚不推荐替代对照。报告docs/RETENTION_SEED802.md，原始artifacts/retention_seed802。已启动seed803仅密集三场景×两权重共6项复核，outputs/moe_perceptive/retention_dense_803，PID290987。先查日志和各summary及controller_completed，勿重复；无新训练。

## 2026-10-20 08:48 支撑脚运动实验
seed803六项密集复评双方均32/32，但混合足链运动代理增19–26%，方向与802一致。docs/RETENTION_SEED802.md追加复核、artifacts/retention_seed803归档。新stance_foot_motion显式可选奖励默认不启用，公式mean(I(Fz>5N)*vxy²)，候选权重-0.5；39项测试通过，原生8env2iter完成且model1存在/无failed。新Studio job4549d2b29e744c95已启动并核实iteration4/fusion，512env500iter、seed75、initialize平台model199、学习率1e-5、冻结本体、原安装/记忆4，结束自动九项严格评估。勿重复启动，勿将训练奖励或单滑移代理当晋级标准；后续需独立802/803复核，训练种子变化不支持单因素因果结论。未恢复旧V3/V4/V7。

## 2026-10-20 09:08 SEA射线接口独立回归
stance任务4549d2b29e744c95仍正常运行，核实iteration258/fusion无数值异常；勿重复启动。CPU侧新增rob2nav/sea_contract.py，仅加载SHA核验外部SEA原生CBF层并显式41束240度；加载state_dict后需再次validate_geometry，测试证实旧buffer可覆盖角度。scripts/validate_sea_cbf.py原生CPU16样本前向/梯度/yaw透传通过，180度和恢复buffer错配被拒绝。原生actor没有被修改，导航checkpoint/训练和动态Go2仍未接通。41测试通过，外部无许可证源文件只在私有缓存，不能上传。证据artifacts/framework_navigation/sea_cbf_contract/validation.json，说明docs/FRAMEWORK_NAVIGATION.md。

## 2026-10-20 09:28 stance训练结束、严格评估中
4549d2b29e744c95已完成500轮，model499 SHA448b60e8836343dd66cf984430c1bc4344bd0742e7653ed353e532d029d35aa91，自动严格suite正在执行；已核实前三密集场景32/32各，首个窄桩0/32，不能提前称九项完成。seed801足链运动代理候选平地0.1463/阶梯0.1094/高台0.1305，高于旧混合0.1228/0.0990/0.1185，未见目标改善；末轮训练奖励stance项-0.01029、速度跟踪0.6337，仅幅值诊断，非贡献因果解释。暂不追加训练。已安排同候选seed802/803三密集场景共6次原生复核，目录outputs/moe_perceptive/stance_independent_802_803，等待控制器PID317368，先等九项completed+九份32回合summary+model499+无trainfailed才执行；勿重复启动。下次比较已存baseline802/803，不把单一代理或更低机械功当成功。

## 2026-10-20 09:48 stance完整评估、平方指标诊断
4549d2b29e744c95 strict九项完成，密集各32/32，原桩六项0/32；独立802/803密集6项全32/32，但足链均值高于平台对照。artifacts/stance_v4_completed和docs/STANCE_V4_VALIDATION.md归档，不晋级不追加训练。诊断发现训练平方速度与旧均值指标非同一量，新增平方积分、按脚数和总时长归一化的奖励原始量、支撑峰值，旧记录网站显示未采集；41测试和构建通过。已核实无活动eval后启动outputs/moe_perceptive/stance_squared_802，PID330085，对照/stance相同802三密集地形共6项新指标复评。下次先看driver/completed和summary，不重复启动；再据平方/峰值/通过率/姿态/功率决定设计。


</details>
