# C6 仿真与资产锁定

## 资产哈希（sha256 前 24 位，完整值见文件本身）

| 文件 | sha256 | 大小 |
| --- | --- | --- |
| `as2w.urdf`（宇树官方 AS2-W） | `696aec2c0915ec1e06bcc91a…` | 15,964 B |
| `as2w.usda`（Isaac 引用层） | `0c5786834c5deda135d31950…` | 780 B |
| `r11_mjcf_excl.xml`（MuJoCo） | `dc8a4d33a687327e021fc6f6…` | 12,371 B |
| `as2w_isaac_flat_env_all.py`（观测/奖励/课程） | `8d6438705e3bf911439a3bd8…` | 14,653 B |
| `as2w_isaac_robust_env.py`（环境/DR） | `1af438db2ff3babc064a3a56…` | 11,287 B |
| `as2w_training_safety.py`（硬约束） | `b8d6ea183d70d80fb97a2d6f…` | 12,115 B |

## 仿真器版本

| 项 | 值 |
| --- | --- |
| MuJoCo | **3.12.0** |
| torch | 2.11.0+cu130 |
| numpy | 2.5.1 |
| Isaac Sim | 6.2.0-rc.14（Isaac Lab 源码 `/workspace/isaaclab/source`） |

## 时间步合同

| 项 | 值 |
| --- | --- |
| 物理步长 | 0.005 s |
| 策略周期 | **0.02 s**（decimation 4） |
| **MuJoCo 参考合同** | `timestep = 0.000625 × 32 子步 = 0.02 s`（脚本内断言，不沿用 MJCF 的 0.00125） |
| `apply_actions()` 调用频次 | **每物理子步 1 次 = 每策略步 4 次**（运行时实测 `calls/step = 4.00`） |

## 已记录的资产改动

`r11_mjcf_excl.xml` 基于官方模型，附加：
1. `<exclude base_link ↔ FL_thigh / FR_thigh>` 接触排除
2. 网格路径改为**绝对路径**（相对路径在复制后会失效）
3. 地面摩擦 `0.62`；轮 geom 未指定摩擦（走 MuJoCo 默认）

## 缺口

- **未附 STL 网格清单哈希**（数量多、体积大，未逐个记录）
- 未记录 USD 引用层的展开内容哈希
