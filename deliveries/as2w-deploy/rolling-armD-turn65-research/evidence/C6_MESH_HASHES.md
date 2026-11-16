# C6 补齐（二）：STL 网格逐个哈希

C6 的 README 状态行写的是「未逐个记录 STL 哈希」。本文件补齐这一项，与 `C6_USD_LAYERS.md`（USD 层）合起来完成 C6 的资产哈希要求。

## 1. 范围

以 URDF `as2w.urdf` 里 `filename=` 引用的**去重网格路径**为准，共 **18** 条引用，逐个哈希。

**全部引用在磁盘上均存在，无缺失。**

## 2. 逐文件哈希

| 网格 | 字节 | SHA256 |
| --- | --- | --- |
| `meshes/FL_calf.STL` | 2619984 | `71008347735a23bfa4417dc0f5f62fd626948efde064f9d09236ee5fde6cedc7` |
| `meshes/FL_foot.STL` | 759984 | `1d61dc09b8bd6f35cc77fa8a12added46e5a8798827ada1e62c47c0ccd9d5c63` |
| `meshes/FL_hip.STL` | 1385084 | `3de163f0c75a6fa3e9c53d2b6775694497a248dc7882a6926fc9091615b43e71` |
| `meshes/FL_thigh.STL` | 641184 | `15496ac2b46c1898fc4bdb3e6805a17fe9e64f6a8f9a03a191fcf25b2fee6ccb` |
| `meshes/FR_calf.STL` | 2618684 | `e2947d60e0fd6164f471399f282109078abb1541b7d1720603450f77bd5b2b7a` |
| `meshes/FR_foot.STL` | 763084 | `47eed0dee9eb8550760667c4a1957c7de6306e4533fb4c51c9de8ecb72e37198` |
| `meshes/FR_hip.STL` | 1383384 | `8084544d9acc8c8b4ee3ee0671259ae57f3dbb88139a0dbc94f3dc59fb2c5bb8` |
| `meshes/FR_thigh.STL` | 640484 | `33a8f48f36db7673463ae7cc38471f2667dcff6287bbe993d7e05cfc298d6a8b` |
| `meshes/JT128.STL` | 4297884 | `f3b6322bcaa018bf1909c0cf13dcac1d96cde0a9d11e0765a1870eb7cda9f3df` |
| `meshes/RL_calf.STL` | 2652284 | `ee3ee88469c72322b3f87463620a45a1b8ea1386e84b12b12f3cb15966da83a8` |
| `meshes/RL_foot.STL` | 760584 | `7d174564f5b39598ce0c00c6a9928862d2188c61bef5a76ad160f8c39e713a93` |
| `meshes/RL_hip.STL` | 1379584 | `7ef0c03ec67237e91fafd01276208cbc170050bf2ee2a3907b25c4d204478642` |
| `meshes/RL_thigh.STL` | 479784 | `b934ff692ce0fec03689f821ece8cf8ffefc935c87ac8eeed52c030f30a83bad` |
| `meshes/RR_calf.STL` | 2650784 | `7efdf4496da96206f1f633c0d4f5c81c1d6f7918ca646192bb29844328f809f3` |
| `meshes/RR_foot.STL` | 762384 | `133ff0182f72ac6cd20c8bd19d15b8c8166eaa3e3539dc0f40c376077de95f39` |
| `meshes/RR_hip.STL` | 1382984 | `989a2cbd34850546c783e98cf01c65b2b1d7834b9ec5a9bfa905e7408a0d078f` |
| `meshes/RR_thigh.STL` | 479284 | `829ec88f0a2868ec06e1238b0e015e46e25a0d258f34c148d5e8d2bac35a2ec4` |
| `meshes/base_link.STL` | 5202484 | `361ed7d19f9ce5a41076da700282913f4646097bad5611aab83e7fc715a48289` |

合计 **18 个文件 / 30859912 字节**（18/18 引用可解析）。

## 3. C6 汇总（两项子缺口均已闭合）

| C6 子项 | 状态 | 证据 |
| --- | --- | --- |
| URDF 本身 | ✅ 已锁定（`696aec2c…`） | 既有记录 |
| **USD 展开引用层（9 层）** | ✅ **本轮闭合** | `C6_USD_LAYERS.md` |
| **STL / 网格（36 引用）** | ✅ **本文件闭合** | `C6_MESH_HASHES.md` |
| 质量惯量 / 执行器 / 碰撞 / 摩擦 | ✅ 已锁定 | 既有记录 |
| 步长 / 延迟 / 随机化版本 | ✅ 已记录 | 既有记录 |
| 训练仿真复跑入口（recipe + hash pin） | ✅ 已提供并校验 | `training/ab-gyro-loop.sh` |

**→ C6 的两个显式缺口（USD 层、STL）均已闭合，可以上调为 pass。**

## 4. 边界

1. 只覆盖 **URDF 直接引用**的网格；`geometries.usd` 内部若另有引用未二次展开2. 未验证 Isaac 导入后网格是否被重新三角化（哈希证明的是**源文件**，不是运行期内存中的网格）3. 未验证运行期选中哪个 physics variant
