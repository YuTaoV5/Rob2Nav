# Go2 统一 SDK 部署与权重热切换框架（验证中）

当前交付包含设计、控制核心、独立策略进程、严格技能契约、官方SDK消息/CRC编码及受控传输适配器。**首个真实MuJoCo切换试验未通过：中立站姿获取超时，尚未提交任何MLP/MoE切换。不能将逻辑测试通过当作完整切换能力或实机验收通过。**

SDK固定 `814556d15970dd2ecf1c9984e845ca02ab07e206`。已在独立环境验证真实LowCmd及CRC，未初始化DDS、未创建发布器、未连接机器人。所有随附部署配置 `hardware_qualified=false`；已知未通过约束的三种特技另设隔离标记，不允许通过热切换启动。

## 架构与差异处理

| 层 | 实现与职责 |
|---|---|
| 严格契约 | `rob2nav/deployment/types.py`：模型SHA、12关节名称双射、观测格式、默认姿态、动作缩放、PD、策略频率、入口与资格 |
| 独立推理 | `policies.py`：45/48/60/69维分别适配；TorchScript/ONNX在独立进程加载与推理，不在500Hz核心加载文件 |
| 热切换事务 | `runner.py`：后台校验/预加载/预推理/清理历史，旧策略继续运行；只有新模型就绪且实测状态满足条件才推进 |
| 控制核心 | `core.py`：捕获实测状态→中立站姿→目标入口→实际观测预热→渐入→提交；失败锁存；旧代次输出不能进入新策略 |
| 单发送器 | `unitree.py`：标准SDK顺序FR/FL/RR/RL；统一软件PD后只发送力矩，SDK的kp/kd均为0，避免重复PD |

相同12维输出不能直接混合。每个策略保留自己的默认关节位置、缩放、观测归一化和历史：45维本体策略与48维直立策略、60维相位策略、69维参考跟踪分别构建输入。SDK LowState没有直立策略所需的机身COM线速度，缺少经验证估计器时明确拒绝，不能用零值冒充传感器。69维还需要固定SHA参考与方向对齐；当前不具备完整实机资格。

SDK电机顺序只在传输边界使用。策略关节名称映射校验一一对应，不靠数组维数猜顺序。符号、零位、IMU轴向、固件、传动/电机参数和足压阈值须随具体机器人完成标定。目前的硬件资格记录门禁不能替代这些尚未实现完整闭环的标定与健康检查。

## 切换时序

1. 后台按清单校验权重SHA、模型输入/输出及有限值，清理候选历史；加载失败不改变当前策略。
2. 请求切换后将当前速度目标按变化率降为零，等待真实四足支撑、低关节速度、低角速度和接近直立连续稳定。
3. 动态原子动作未完成时只暂存请求。腾空、两足直立或没有已验证退出控制器时禁止强行插入站姿。
4. 从最近实际控制参考走五次平滑轨迹至中立与目标入口，同时限制关节参考速度/加速度和PD变化率。
5. 用新状态、实际执行参考换算的动作历史预热候选策略；候选期间相位冻结，不向电机直接发送候选动作。
6. 渐入后提交新代次。输出包含代次与时间戳，旧代次、迟到、非有限数输出会被拒绝。

软件伺服持续约束参考速度2rad/s、加速度20rad/s²、kp变化15/s、kd变化2/s、力矩变化150N·m/s，并使用URDF近似DC包络。这是当前站姿/缓慢动作验证配置，**不适用于直接放行特技**；特技必须重新训练/评估与其一致的执行器与滤波约束。力矩包络与变化率无交集时显式故障，不能同时承诺两种冲突约束。

## 已执行验证

- 8项核心检查：关节双射、PD/参考/力矩变化界、陈旧/未来/非法状态、旧代次动作、腾空原子动作禁止切换、硬件/失败技能隔离、不同观测要求、看门狗锁存。
- 真实官方SDK IDL/CRC离线检查：20个协议电机槽，812字节序列化；篡改力矩能被CRC检测；前12电机顺序正确，未使用槽力矩为0；没有双重PD。1000次编码P99约0.152ms，最大约0.204ms。这不是完整500Hz实机循环的实时性证明。
- 首轮真实自由机身MuJoCo：`seed9201`，3501物理步，最大相邻力矩变化约0.0694N·m，kp变化0.03，最大倾斜9.67°，最低机身0.301m。但中立获取超时，提交次数0；故整体失败。原始报告中backend字符串描述的是试验计划，实际没有执行到MLP/MoE提交，以提交记录为空为准。

原始证据见[资源归档](../artifacts/release_20261021/README.md)中的`validation/`。不得删掉失败记录或仅放宽站稳门槛来宣布成功。

## 离线复现命令

```bash
cd /root/gpufree-data/Rob2Nav
PY=/root/gpufree-data/envs/go2-mapless/bin/python
$PY -m rob2nav.deployment profiles
$PY -m rob2nav.deployment plan --from walk_moe --to backflip60
$PY -m unittest discover -s tests -p test_deployment_core.py -v
# 新目录；该命令运行真正物理仿真，当前基线预期可能仍失败
MUJOCO_GL=egl $PY -m rob2nav.deployment.verify_mujoco \
 --seed 9202 --video --output outputs/unified_switch_new
```

离线SDK环境（已建好，编译/安装记录须按环境重建）：

```bash
export CYCLONEDDS_HOME=/root/gpufree-data/tools/cyclonedds-0.10.2
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
/root/gpufree-data/envs/go2-sdk-check/bin/python scripts/verify_sdk_codec.py \
 --sdk-root /root/gpufree-data/Rob2Nav-upstreams/unitree_sdk2_python \
 --output outputs/sdk_codec_new.json
```

## 实机阶段方案与当前缺口

只读录制→标定映射/符号/零位/IMU→离线重放→带保护支撑的低幅检查→四足站立及断流恢复→缓慢策略切换→单独动态技能。每个阶段保存机器人序列号、固件、SDK提交、模型/配置/标定SHA、热状态与电压、传输丢失、足压基线、周期P99/最大值及恢复策略。

传输器默认只读。写入前要求设备/SDK/模型匹配的未过期资格记录、硬件合格策略、进程写入租约以及运动服务未占用；不会自动调用StandDown或ReleaseMode。文件租约只约束本框架，不能保证网络上不存在其他发送器。资格记录是运维门禁，不是加密证明。

尚需完成：中立支撑与退出控制器诊断、多种子真实模型切换；完整符号/零位标定应用及电压/温度/掉线健康保护；真实500Hz发布循环时序、独立机载看门狗和故障恢复；69维参考yaw对齐、48维线速度估计与各特技约束验收。Python调度、DDS失联和硬件力矩响应无法由上述离线结果保证，实际接机前仍须逐阶段验证。

官方接口依据：[固定SDK低层示例](https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/go2/low_level/go2_stand_example.py)。本仓库没有将其自动释放服务/起立动作带入离线测试。
