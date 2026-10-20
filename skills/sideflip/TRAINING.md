# Go2 侧空翻训练手册

本目录是Rob2Nav新适配：官方Unitree mimic MDP、Go2 URDF及独立PPO。已实际生成118帧参考，完成32环境2次PPO更新并导出69维actor；4096环境5000更新完整训练正在运行。短程模型的MuJoCo回合没有完成翻转，记录为0/1；完整能力尚未通过。

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

本次相同参数完整任务已在outputs/sideflip_native_full_20261021运行，不要重复启动。日志capability-work/sideflip/full.log；checkpoint每100更新及末次保存。恢复可增加--resume旧检查点并选择新--output；--iterations表示继续的更新数。Actor69维、12动作、50Hz，物理200Hz、PD40/1、动作缩放0.5；力矩及速度限制按URDF逐关节读取。训练中的动作参考相位重采样属于episode reset；独立MuJoCo评估仅初始化时设置根状态。

下一步：[仿真手册](SIMULATION.md)、[SDK方案](DEPLOYMENT.md)。奖励上升和训练结束均不等于侧空翻通过。
