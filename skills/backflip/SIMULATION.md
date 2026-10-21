# Go2 后空翻仿真手册

CPU环境固定为已执行的Python3.11依赖。首次建立环境时执行（已有环境不重复创建）：

```bash
cd /root/gpufree-data/Rob2Nav
python3.11 -m venv /root/gpufree-data/envs/go2-skills-cpu
/root/gpufree-data/envs/go2-skills-cpu/bin/python -m pip install -r skills/requirements-stunts-cpu.txt
```

本次服务器已具备相同版本的/root/gpufree-data/envs/go2-mapless；下文沿用它。新环境使用者把后续PY/解释器替换为go2-skills-cpu/bin/python。训练仍必须使用Isaac解释器，不能用这个CPU环境启动IsaacLab训练。

本次实际执行上游model_6000_actor.onnx的6.02秒MuJoCo回合：累计俯仰-6.270rad，最高机身0.668m，最终倾角0.002rad，无头部接触。动作任务通过，但最大关节速度39.10rad/s、关节越限0.1077rad，**上游硬件约束与总体验证失败**。这是诊断结果，不能作为实机可部署认证。独立重训模型的仿真待训练完成后验证。

先完成[训练手册](TRAINING.md)的来源与环境准备。

```bash
PY=/root/gpufree-data/envs/go2-mapless/bin/python
$PY -m pip install onnxruntime==1.23.2
cd /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
export OMP_NUM_THREADS=1 MUJOCO_GL=egl
$PY mujoco/go2/play_onnx.py --onnx model_6000_actor.onnx --headless --no-realtime --start --duration 6
```

交互桌面环境可去掉--headless和--no-realtime，SPACE触发一次动作，R显式物理重置。服务器无桌面时使用无窗口模式。默认MJCF是同目录go2_isaac.xml，不能随意替换其他Go2模型而保持原合约。

导出本次训练的最终检查点：

```bash
cd /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
RUN=$(find logs/rsl_rl/go2_backflip -maxdepth 1 -type d -name '*rob2nav_reproduction' | sort | tail -1)
MODEL=$(find "$RUN" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -n "$MODEL" && test -f "$MODEL"
MODEL=$(realpath "$MODEL")
cd /root/gpufree-data/Rob2Nav
/isaac-sim/python.sh skills/backflip/train_isaac51.py --entry play --source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --checkpoint "$MODEL" --headless --device cuda:0 --num_envs 1 --steps 1 --export
cd /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
/root/gpufree-data/envs/go2-mapless/bin/python mujoco/go2/play_onnx.py --onnx "$(dirname "$MODEL")/exported/policy.onnx" --headless --no-realtime --start --duration 6
```

训练过程中的中间模型可诊断，但不是完整训练交付。验收需分开记录：完整约2π俯仰、落地后稳定、机身/头部碰撞、关节速度、硬限位、力矩与时序约束。上游脚本会打印任务、硬件约束、总体三个判定；即使脚本进程退出0也不得忽略FAIL。后续独立多种子、初始条件和模型扰动测试仍未完成。
