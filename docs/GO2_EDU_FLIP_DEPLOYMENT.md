# Go2 Edu 空翻：网站演示与实机部署手册

更新：2026-10-22。当前网页单技能演示使用**侧翻 v8 / model_7697**和**后翻 v6 / model_9495**，均为本项目服务器续训所得。选择依据是本轮“FIXSTAND → 完整翻转 → FIXSTAND”目标的首尾站姿表现，不是所有历史指标的全局最优。这两份候选首尾站姿均为独立四种子4/4，**完整验收仍为0/4，没有实机运行记录**。

## 1. 当前交付和限制

| 项目 | 侧翻 v8 | 后翻 v6 |
|---|---|---|
| 首/尾 FIXSTAND | 各4/4 | 各4/4 |
| 主轴累计旋转 | 约−6.23 rad | 约−6.28 rad |
| 主轴占比 | 0.7235–0.7247，要求>0.75 | 0.7989–0.8026 |
| 最大关节速度 / URDF额定速度 | 1.0855–1.0871 | 1.0986–1.1092 |
| 硬限位超出 | 0 rad | 0 rad |
| 非足端机身接地 | 0/4 | 4/4 |
| 训练/导出接口 | 69维 TorchScript | 60维 ONNX / TorchScript |
| 当前用途 | 网页研究仿真、下载、续训 | 网页研究仿真、下载、续训 |

FIXSTAND包含姿态误差、关节/机身速度、重力、站高和四足承重，不能仅凭最后一帧外观判断。评估仅覆盖固定平地和小幅初始关节扰动，不代表地形、摩擦、负载或实机鲁棒性。

全部资源位于 `artifacts/fixstand_best_flips_20261022/`：最终训练检查点、推理模型、实际训练YAML、导出合约、四种子 summary/trace、各一段视频和侧翻参考NPZ；`SHA256SUMS`记录文件校验值。后翻 v4/v5 的错误入口训练合约已作废，不能替代 v6。

## 2. 网站操作

1. 打开工作台的“在线仿真”，选择侧空翻或后空翻，点击启动；实时 MuJoCo 使用这次发布的固定候选。
2. 左键拖动旋转，右键或 Shift 拖动平移，滚轮缩放。侧翻执行8秒、后翻5秒，结束后可重新启动；页面失联8秒停止仿真。
3. “模型资产”按技能筛选，选择带有“FIXSTAND 首尾优先候选 v8/v6”的条目下载。训练恢复用 `model_*.pt`，推理用 `exported/policy.pt` / `policy.onnx`；同名文件务必连同完整目录保存。
4. “报告”搜索本手册；工作台技能卡片的视频也对应新版种子8800/10000。模型库的 selected 仅表示选定候选，不表示通过验收。

这次更新的是**单技能原生仿真和模型库**。统一编排的 `sideflip69/backflip60` 仍是历史隔离配置，尚未换成新权重，也未取得新版连续热切换成绩；不要把单技能演示当作统一部署已通过。

## 3. 在 Linux 复现完全相同的仿真

在项目已配置的 MuJoCo 环境运行（服务器为 `/root/gpufree-data/envs/go2-mapless/bin/python`）。新机器使用独立 Python 环境安装 NumPy、PyTorch、ONNX Runtime、MuJoCo、PyYAML、imageio/ffmpeg、Pillow 和 SciPy；版本以项目已有验收环境为基准，ARM机载板需选择与其架构相符的包。此步骤不需要 Isaac Sim，也不初始化 DDS。

本次服务器实际环境为x86_64、Python3.11.13。以下命令用于新建相同主要依赖的CPU仿真环境；需事先安装Python3.11及venv支持，EGL渲染还需要系统OpenGL/EGL驱动：

```bash
python3.11 -m venv ~/venvs/rob2nav-flips
source ~/venvs/rob2nav-flips/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 onnxruntime==1.23.2 mujoco==3.2.3 \
  PyYAML==6.0.3 imageio==2.38.1 imageio-ffmpeg==0.6.0 pillow==13.0.0 scipy==1.17.1
```

```bash
git clone https://github.com/YuTaoV5/Rob2Nav.git
cd Rob2Nav
git clone https://github.com/Robot-Nav/GO2_backflip.git ../GO2_backflip
git -C ../GO2_backflip checkout 954aa61ceb62c515689677f44a21e32109f6b62d
(cd artifacts/fixstand_best_flips_20261022 && sha256sum -c SHA256SUMS)
BASE=artifacts/fixstand_best_flips_20261022/outputs
SIDE=$BASE/fixstand_sideflip_peak_pose_v8_20261022
BACK=$BASE/fixstand_backflip_verified_entry_v6_20261022
export MUJOCO_GL=egl
python -m rob2nav.mujoco_skill --kind sideflip \
  --policy "$SIDE/exported/policy.pt" --backflip-source ../GO2_backflip \
  --motion "$BASE/fixstand_finetune_20261021/sideflip.npz" \
  --physics-hz 1000 --seconds 8 --seed 8800 --fixstand-roundtrip \
  --video --output outputs/sideflip_v8_local_001
python -m skills.backflip.simulate --source ../GO2_backflip \
  --onnx "$BACK/exported/policy.onnx" --physics-hz 1000 --seconds 5 \
  --seed 10000 --fixstand-roundtrip --video --output outputs/backflip_v6_local_001
```

每次复现使用新输出目录。检查 `summary.json` 中 `metrics.fixstand`、`successes`、`hardware_constraints_pass`，而非只看退出码。四种子分别为8800–8803、10000–10003。训练重跑入口为 `scripts/train_sideflip_fixstand_v8.sh` / `scripts/train_backflip_fixstand_v6.sh`；它们依赖原Isaac环境与固定父检查点，不在Go2机载电脑训练。

## 4. 连接你的 Go2 Edu 机载电脑

训练服务器不是机载电脑。先从设备网络配置确认**机载电脑IP、用户名、SSH端口、与机器人相连的物理网卡**。在控制工作站执行：

```bash
ssh -p <机载SSH端口> <机载用户名>@<机载IP>
uname -m
cat /etc/os-release
ip -br addr
ip route
```

`--interface`必须是运行控制程序的Linux主机上连接机器人DDS网络的网卡，不是Windows网卡名或SSH隧道。建议固定有线链路。SSH终端可用于键盘操作，终端断线会丢失人工输入，因此仍需设备端独立停止路径。

将仓库克隆/同步到机载电脑。只运行MuJoCo复现时不装SDK；实际控制另建Python3.11环境，使用已离线核对的 NumPy1.26.4、CycloneDDS C库0.10.2和以下SDK固定提交，避免覆盖Noetic/Foxy系统环境：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git ../unitree_sdk2_python
git -C ../unitree_sdk2_python checkout 814556d15970dd2ecf1c9984e845ca02ab07e206
# 在已安装git、cmake、C编译器的机器上，将DDS安装到独立前缀。
git clone --branch 0.10.2 --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds.git ../cyclonedds-0.10.2
cmake -S ../cyclonedds-0.10.2 -B ../cyclonedds-0.10.2/build \
  -DCMAKE_INSTALL_PREFIX="$HOME/tools/cyclonedds-0.10.2"
cmake --build ../cyclonedds-0.10.2/build --parallel 2
cmake --install ../cyclonedds-0.10.2/build
export CYCLONEDDS_HOME="$HOME/tools/cyclonedds-0.10.2"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
python -m pip install -e ../unitree_sdk2_python
python scripts/verify_sdk_codec.py --sdk-root ../unitree_sdk2_python \
  --output outputs/go2edu_sdk_codec.json
python -m rob2nav.deployment profiles
```

SDK编码检查不发送电机命令。板卡架构、固件、驱动和实时延迟尚未在你的设备验证，安装成功不能代替设备测试。具体设备/模型/资格记录字段见 [统一部署](UNIFIED_GO2_DEPLOYMENT.md)。

安装方法依据仓库固定的 [Unitree SDK README](https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206)。若pip尝试升级NumPy，保留1.26.4并选择兼容的OpenCV依赖，重新跑离线编码检查。供应商示例文件可能包含电机写入，不能仅凭名称含“lowlevel”把它当只读诊断。

## 5. 把候选接入统一运行器时必须保持的控制合约

| 合约 | MoE行走/楼梯 | 侧翻v8 | 后翻v6 |
|---|---|---|---|
| 观测维数 | 45 | 69 | 60 |
| 策略频率 | 50 Hz | 50 Hz | 50 Hz |
| Kp / Kd | 20 / 0.5 | 40 / 1 | 40 / 1 |
| 动作缩放 | 0.25 | 0.5，中段原始偏移 | 0.5 |
| 参数化 | joint_offset_v1 | reference_residual_v3 | backflip_entry_v3 |
| 特有资源/参数 | 原始RoboGauge MoE | 本发布NPZ；残差0.20、桥0.05、末50帧髋衰减 | 1秒入口、1–1.2秒释放、hold=0.1、clip=8 |

侧翻关节顺序为四腿hip组、thigh组、calf组（组内FL/FR/RL/RR）；后翻为FL/FR/RL/RR逐腿三关节；SDK为FR/FL/RR/RL逐腿。必须按名称重排，动作历史也经SDK顺序传递再转回各策略顺序。后翻每腿速度曲线x2是30.1/30.1/15.7 rad/s，目标速度13.5 rad/s；完整驱动曲线以 `exported/export.json.control_contract` 为准。不能用一个统一Kp/Kd或clip替代训练配置。

新权重的 `policy`、`sha256`、`reference`、`reference_sha256`、动作参数化和FIXSTAND入口需一起写入隔离的候选配置；加载参数化相关字段定义见 `rob2nav/deployment/types.py`。首尾名义站姿：四个hip=0、前thigh=0.8、后thigh=1.0、calf=−1.5 rad。接口替换后必须重新做动态切换验收；本次没有将候选写入生产配置。

## 6. 按键与完整状态机

设备验收完成后，在Linux交互终端使用唯一低层服务：

```bash
python -m rob2nav.deployment.teleop -- run --profile stand \
  --interface <机器人网卡> --sdk-root <固定SDK路径> \
  --calibration <本机标定.json> --qualification <本机资格.json> \
  --model <核对过的Go2.xml> --output outputs/go2edu_session_001 --enable-motors
```

当前仓库配置会因未获设备资格而拒绝执行。去掉 `--enable-motors` 仅输出“不连接DDS”的说明，**不会验证标定或完成实机预检**。标定/资格示例是字段模板，不能通过改布尔值代替验收。

| 键 | 操作 |
|---|---|
| 1 / 2 | 标准站立 / 原始MoE（平地和楼梯同一权重） |
| 3 / 4 | 请求侧翻 / 后翻；当前隔离配置会拒绝 |
| 5 | 预留直立行走，尚未达标 |
| T | 对已进入ARMED的原子技能显式触发 |
| W / S | 前进0.18 / 后退−0.12 m/s |
| A / D | 横移±0.10 m/s |
| Q / E | 转向±0.35 rad/s |
| 空格 / Esc | 归零速度 / 停止服务 |

终端通过按键重复保持速度，超过约0.18秒无速度键即归零；不是浏览器键盘控制电机。

```text
CAPTURE（捕获实测姿态/承重）→ NEUTRAL（站稳）→ READY
→ ENTRY（姿态、增益连续过渡）→ WARMUP（重建观测历史）
→ BLEND → ACTIVE                         [普通步态]
→ ARMED → T → ACTIVE → 完整动作/末段站稳   [空翻]
ACTIVE → EXIT → NEUTRAL → 下一技能ENTRY
已验证短路径：ACTIVE → 下一技能ENTRY
故障 → FAULT → 请求阻尼停止
```

切换等待真实站稳。腾空期间不抢切、不混合两个空翻网络；通过统一500Hz发送器桥接Kp/Kd和目标，不启动第二个LowCmd发布器。MoE同一技能的多段任务只改速度，不重新载入或添加站立动作。减少过渡只对精确版本、方向已验收的 `compact_switch_pairs` 启用；本轮两份候选尚没有该记录。

## 7. 实机前还缺什么，如何判断完成

按顺序完成：①侧翻主轴/超速、后翻超速/机身接地全部修复，冻结模型后四种子严格通过；②同一统一运行器测试 `stand → 动作 → stand`、`MoE → 动作 → MoE` 及重复序列，核对历史、PD和退出时序；③你的Go2 Edu上核对关节符号/零点、IMU/足力、设备序列号/固件、模型质量/COM、DDS独占和端到端延迟；④现场验证停止/恢复与控制行为并生成该设备、该模型、该完整代码版本的资格记录。

没有上述记录时，本手册交付的是可复现仿真和真实部署流程，不是“当前空翻可直接上狗”的结论。当前失败证据见 [候选归档](../artifacts/fixstand_best_flips_20261022/README.md)；总体进展见 [续训报告](FIXSTAND_CONTINUATION_20261022.md)。
