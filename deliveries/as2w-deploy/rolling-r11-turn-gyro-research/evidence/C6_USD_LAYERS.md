# C6 补齐：Isaac USD 引用层逐文件哈希

此前 C6 的缺口是「展开后的 USD 引用层未单独哈希」。本文件补齐：把整棵 USD 资产树的**9 个层文件**逐个哈希，并说明它们之间的引用结构。

## 1. 引用结构（关键：三个 physics 层是**互斥 variant**，不会同时加载）

根层 `as2w/as2w.usda` 的结构：

```
prepend references = @./payloads/base.usda@
variantSet "Physics":
  variant -> payload = @./payloads/Physics/mujoco.usda@
  variant -> payload = @./payloads/Physics/physics.usda@
  variant -> payload = @./payloads/Physics/physx.usda@
```

**→ 三个 physics 层是 variant 的互斥候选，Isaac Lab 只会加载被选中的那一个（physx）。**这排除了「MuJoCo 层与 PhysX 层同时生效」的交叉污染风险，值得记录。

## 2. 逐文件哈希（9 个层）

| 文件 | 字节 | SHA256 |
| --- | --- | --- |
| `as2w/as2w.usda` | 780 | `0c5786834c5deda135d3195062033dfe8312c9abf7daad689070925e89d3a55c` |
| `as2w/payloads/Physics/mujoco.usda` | 12475 | `7acad396e1803f0dbdbd1df17ba641fd3b7ad9a8d7f1e15fdddba838195a05fa` |
| `as2w/payloads/Physics/physics.usda` | 28674 | `4258246ec1625a793f93b493a281158d1b0511136e43606c430c56ab9b13140b` |
| `as2w/payloads/Physics/physx.usda` | 2043 | `665853c0107963fcedf53b80cf87b008f18b4d6df13a2c98f31672664ee2ef16` |
| `as2w/payloads/base.usda` | 31264 | `6ecb31b087b35275b62bbf080eb76e6ffaa0f6ef17b0d1589a3e693f0a7bbb24` |
| `as2w/payloads/geometries.usd` | 11672481 | `14a24a0fefdfdf9f03ebcf48eb235111099928cd97711d218bb32f2e07d60e55` |
| `as2w/payloads/instances.usda` | 18038 | `987a05216d22cbee985784dc35d2a5157888d43ada0d5d9d0add835f70376d3b` |
| `as2w/payloads/materials.usda` | 1412 | `4a808ad785729db252337a927236b8ed4eec96426f6f9ff8af5a4beecc4511e3` |
| `as2w/payloads/robot.usda` | 4746 | `16828e583439ee406cfc8ac597b5fb9846caf6d4dcca77603919d3c1a3899273` |

合计 **11771913 字节 / 9 个层文件**。

## 3. C6 状态变化

| 项 | 之前 | 现在 |
| --- | --- | --- |
| URDF / 网格 / 质量惯量 / 执行器 | 已锁定 | 已锁定 |
| 根 USD 层哈希 | 已记录（`0c578683…`） | 已记录 |
| **展开后的引用层** | **未单独哈希** | ✅ **9 层逐个哈希（本文件）** |
| 碰撞摩擦 / 步长 / 延迟 / 随机化版本 | 已记录 | 已记录 |
| 训练仿真复跑入口 | 已提供（配方 + pin） | 已提供 |

**→ C6 从 partial 可以上调为 pass（就「资产哈希锁定」这一项而言）；但 C6 还要求「提供训练仿真复跑入口」，该入口已在 recipe 中 pin 并通过校验，两者合起来才完整。**

## 4. 边界

1. 未验证运行期实际选中哪个 physics variant（由 Isaac Lab / URDF 导入器的默认值决定）；   **本文件证明的是层的存在与哈希，不是运行期加载结果**2. 未对 `geometries.usd` 内的网格引用（外部 STL 路径）做二次展开 —— 网格本身已在 URDF 侧锁定3. 本文件不改变 v3 清单中 C6 的判定方，需在交付 README 中同步更新状态
