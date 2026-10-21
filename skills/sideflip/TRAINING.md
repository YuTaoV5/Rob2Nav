# Go2 侧空翻训练手册

本目录使用官方Unitree mimic MDP、Go2 URDF及独立PPO。已完成4096环境5000轮训练，再做显式电机模型适配；固定model_5200在MuJoCo的8个独立新种子回合中动作通过8/8，关节速度门限仍未通过。以下命令覆盖参考预处理、原生训练与已通过动作的适配版本，不借用上游预训练特技权重。

服务器环境：Isaac Sim5.1、IsaacLab2.3.2、torch2.7 CUDA12.8、RSL-RL3.1.2。使用Isaac解释器训练，CPU MuJoCo环境用于导出/评估。

```bash
cd /root/gpufree-data/Rob2Nav-upstreams
git clone --branch PPO-backflip https://github.com/Robot-Nav/GO2_backflip.git
git -C GO2_backflip checkout 954aa61ceb62c515689677f44a21e32109f6b62d
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
git -C unitree_rl_lab checkout 4960b84732b0c2ec593dccbfe963fda1bcd7b1e3
```

目录已存在时跳过clone并核对HEAD，启动器检查上述提交。Go2资产为MIT来源；官方mimic代码为Apache-2.0来源。动作数据从固定的LeggedSkillDeploy提交单独获取，未随本仓库重新分发：

```bash
mkdir -p /root/gpufree-data/capability-work/sideflip
cd /root/gpufree-data/capability-work/sideflip
curl -fL https://raw.githubusercontent.com/haozhang04/LeggedSkillDeploy/9126dad84708fc2714c6a1158897c86844d0cc1c/policy/unitree_rl_lab/go2/go2_silde_filp/go2_sildfilp_0428_for_csv_to_npz.csv -o input.csv
sha256sum input.csv
```

CSV必须得到`a3216d6236de9391286f1ff7493be161d00813b762d11a64aef1fa2ea370eb12`。CSV为XYZ、XYZW四元数、FR/FL/RR/RL各hip/thigh/calf共19列。转换器按实际Isaac关节名称重排，输出原生关节/刚体数组和哈希清单：

```bash
cd /root/gpufree-data/Rob2Nav
/isaac-sim/python.sh -m skills.sideflip.build_motion --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --csv /root/gpufree-data/capability-work/sideflip/input.csv --output /root/gpufree-data/capability-work/sideflip/reference_new.npz --headless --device cuda:0
```

这是正向运动学数据预处理，不进行物理积分，不作为动作通过证据。输出目录使用独立USD缓存，不改写正在运行的后空翻资产。参考已存在时使用新名称，禁止覆盖正在训练的数据。

```bash
/isaac-sim/python.sh -m skills.sideflip.train --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab --motion /root/gpufree-data/capability-work/sideflip/reference_new.npz --output outputs/sideflip_interface_new --num-envs 32 --iterations 2 --seed 43 --device cuda:0 --headless
ls outputs/sideflip_interface_new/model_*.pt outputs/sideflip_interface_new/completion.json
```

同时核对模型、TensorBoard事件和completion.json；Isaac关闭过程可能掩盖Python异常，退出0不足以证明训练完成。接口通过后完整训练：

```bash
/isaac-sim/python.sh -m skills.sideflip.train --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab --motion /root/gpufree-data/capability-work/sideflip/reference_new.npz --output outputs/sideflip_full_new --num-envs 4096 --iterations 5000 --seed 43 --device cuda:0 --headless
```

本次相同参数完整任务已在outputs/sideflip_native_full_20261021完成，最后检查点为model_4999.pt；不要覆盖已有输出。日志capability-work/sideflip/full.log；checkpoint每100更新及末次保存。恢复可增加--resume旧检查点并选择新--output；--iterations表示继续的更新数。Actor69维、12动作、50Hz，物理200Hz、PD40/1、动作缩放0.5；力矩及速度限制按URDF逐关节读取。训练中的动作参考相位重采样属于episode reset；独立MuJoCo评估仅初始化时设置根状态。

下一步：[仿真手册](SIMULATION.md)、[SDK方案](DEPLOYMENT.md)。奖励上升和训练结束均不等于侧空翻通过。

## 复现已通过动作的DC5200版本

旧5000轮结果保留，不覆盖。新增 `--transfer-motors` 将PhysX硬限速替换为显式线性力矩—速度包络，物理步长2ms、策略50Hz；两引擎使用相同包络。参数取URDF额定边界，仅为仿真模型，不代表已辨识的真实Go2电机。该对照不自动证明跨仿真成功。

```bash
cd /root/gpufree-data/Rob2Nav
/isaac-sim/python.sh -m skills.sideflip.train --headless --device cuda:0 \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab --motion /root/gpufree-data/capability-work/sideflip/reference_new.npz \
  --resume outputs/sideflip_full_new/model_4999.pt --transfer-motors \
  --num-envs 2048 --iterations 202 --seed 43 --output outputs/sideflip_dc_reproduce_202
```

完整检查点含优化器和迭代数；恢复后文件编号继续沿用原编号，不应假定最终文件名为model_999.pt。按目录实际编号选择最高检查点，使用CPU导出器保存新目录；policy.json中的motor_model应为dc_linear_v1。评估时不应把DC模型权重套到旧理想PD执行器上。侧空翻适配同时将非足端接触惩罚设为−10，必须重新评估接触和落稳，不能借用旧模型成绩。

上述202次继续更新对应model_5200.pt。服务器的适配任务使用1000轮预算并继续运行；已通过8回合的模型是其中固定的5200检查点，不是未来末次检查点。先用seed 1000作模型检查，再冻结策略，以3100–3107八个种子独立测试。已执行策略SHA256为7616e53e2c95434f824c2728621f8052e2d41a4e033fd00a25f316de211e07e1。重训受GPU数值与依赖影响，不承诺逐位相同；以自身模型的独立仿真为准。


## 标准站姿首尾专项续训（2026-10-21）

新增[统一训练目标与操作说明](../FIXSTAND_ROUNDTRIP.md)。本次三项32环境短程接口均完成，但新首尾站姿验收未通过；1024环境、每项500次更新的专项续训正在独立目录执行。新候选不自动覆盖旧部署权重，具体状态和四种子独立验收以该说明中的status.json与evaluation.json为准。
