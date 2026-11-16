# MuJoCo 资产已入库（可移植）

操作员要求「mujoco 相关资产也上传 github」。本轮完成，并顺带发现两个真实缺陷。

## 1. 发现的两个缺陷

| # | 缺陷 | 影响 |
| --- | --- |
| 1 | **我实际使用的 MJCF 从未入库** —— 它在 `/root/gpufree-data/as2w-runs/...` 的 run 目录里 | 任何人从仓库复现 MuJoCo 评测都**缺模型** |
| 2 | **MJCF 里 18 条 mesh 路径是绝对路径**（`/root/gpufree-data/Rob2Nav/...`） | 换机器/换检出位置即**加载失败**，完全不可移植 |

## 2. 修复

- 复制到交付包：`runtime/r11_mjcf_excl.xml`- **18 条绝对 mesh 路径 → 裸文件名**，并加 `<compiler meshdir="..."/>` 指向仓库内已跟踪的 STL 目录  （`training/as2w-backup/as2w-current/assets/r237-train-root/official-model/robots/as2w_description/meshes`）- mesh 资产**本来就在仓库里**，无需重复归档

## 3. 验证（两层）

| 层 | 结果 |
| --- | --- |
| 从仓库位置直接加载 | ✅ `nq=23 nv=22 nbody=18 ngeom=19 nu=16` |
| **从 GitHub 干净克隆加载** | ✅ `nq=23 nu=16 ngeom=19` —— **真正可移植** |

| 项 | 值 |
| --- | --- |
| 文件 | `runtime/r11_mjcf_excl.xml` |
| sha256 前缀 | `56bfa3ffa4cb2dcf4b6ba51bfd3145cd` |
| 原始来源 | `/root/gpufree-data/as2w-runs/r11_fanziqi_lab/mjcal-20261115-224217/r11_mjcf_excl.xml`（12,371 B，sha `dc8a4d33a687327e`） |
| 模型 XML 内 timestep | **0.00125 s**；评测脚本会覆写为 **0.000625 ×32 = 0.020 s**（合同值），已在脚本中体现 |

## 4. 与既有资产的关系

| 资产 | 位置 | 状态 |
| --- | --- | --- |
| MuJoCo XML | `deliveries/.../runtime/r11_mjcf_excl.xml` | ✅ **新增入库，相对路径** |
| STL 网格（18 个） | `training/as2w-backup/.../as2w_description/meshes/` | ✅ 原本已跟踪 |
| URDF | `training/as2w-backup/dependencies/unitree_ros_as2w/.../as2w.urdf` | ✅ 原本已跟踪 |
| USD | `training/as2w-backup/as2w-current/runs/r280-isaac-flat-preflight/...` | ✅ 原本已跟踪 |
| MuJoCo 评测脚本（12 个 `r11_mj_*.py`） | `deliveries/.../runtime/` | ✅ 早前已入库 |

## 5. 教训（已补入 ENGINEERING_LESSONS 的同类）

**「资产在服务器上能跑」不等于「资产在仓库里能复现」。** 两条判据缺一不可：① 文件必须在仓库内被跟踪；② **路径必须相对**，且**从干净克隆实加载一次**才可宣称可复跑。
这也是 `E1`（静态检查通过≠入口可用）在**资产**上的同构问题。
