# Go2 EDU：在云服务器查看实机 RViz

更新：2026-10-23。新环境先读[环境准备与路径约定](环境准备与路径约定.md)。ROS1现场终端与ROS2仿真终端分开，不能混合source。配套入口：[Noetic / Foxy 实机导航手册](Go2Edu导航与机载连接指南.md)。

本文给出远程可视化部署方案，不代表真机导航、VNC 或端到端连接已经验收。所有地址、用户名和端口通过现场参数填写；不应把 SSH 密码写入脚本或仓库。

## 1. 目标、边界与已确认条件

当前推荐：**在现场运行 ROS1 RViz，通过反向 SSH 把 VNC 画面送到云服务器，在服务器现有桌面里查看。** RViz 进程实际位于现场，服务器运行 VNC Viewer。若要求 RViz 进程和渲染必须位于云服务器，见第 7 节。

2026-10-23 对目标服务器的只读检查结果如下；这些是当时的环境快照，不是所有部署环境的默认值。

| 检查项 | 结果 |
| --- | --- |
| 操作系统与 ROS | Ubuntu 24.04，ROS2 Jazzy，已安装 RViz2 |
| SCAN | 已有 ROS2 工作区及编译产物 |
| 桌面 | XFCE + Selkies，内部 HTTP 端口 3000 返回 200 |
| 图形显示 | Xorg `:20`，NVIDIA L40，OpenGL 正常 |
| SSH | 允许 TCP 转发，`GatewayPorts no` |
| VPN / 容器前提 | 未发现 `/dev/net/tun` 或 Docker / Podman 命令 |
| 现场链路 | 尚未连接 Go2；未验证现场 VNC、隧道和真实点云显示 |

历史排查记录中的 Go2 导航链使用 ROS1 Noetic；最后仍在排查 LIO 的 `time_stamp is out of range` 和 `IMU Data Not Enough`。服务器安装 SCAN 并不意味着已经接入真机。Jazzy RViz2 也不能直接订阅 ROS1 话题。

## 2. 环境与推荐拓扑

```text
Go2 EDU：MID360 → LIO → 坐标适配 → SCAN
                   │ ROS1 话题（现场网络）
                   ▼
        现场电脑：ROS1 RViz + VNC Server
                   │ 反向 SSH，传输桌面画面
                   ▼
        云服务器：现有桌面 + VNC Viewer
                   │ 平台桌面入口
                   ▼
              操作者浏览器
```

现场电脑可以是 Go2 机载电脑，也可以是与 Go2 同局域网的 Linux 电脑。优先使用已有可用图形桌面的现场电脑，减轻机载计算负担。机载无显示器时，需要先验证虚拟桌面的 OpenGL；不能假设安装 VNC 后 RViz 一定能正常渲染。

该方案将点云、TF、轨迹处理留在现场，跨公网传输桌面画面。导航控制继续在机载端执行；关闭远程窗口不会自动停止机器人。

## 3. 现场端：先确认 ROS1 RViz 能显示

在干净的 Noetic shell 中加载已部署工作区，不要混入 Foxy / Jazzy 环境。若使用独立现场电脑，还需要配置 ROS1 双机网络：

```bash
source /opt/ros/noetic/setup.bash
read -rp 'Go2 在现场局域网中的 IP: ' ROBOT_LAN_IP
read -rp '本机可被 Go2 访问的局域网 IP: ' VIEWER_LAN_IP
export ROS_MASTER_URI="http://${ROBOT_LAN_IP}:11311"
export ROS_IP="$VIEWER_LAN_IP"
unset ROS_HOSTNAME
rostopic list
```

Go2 上各节点也必须发布现场电脑可访问的地址；双方主机名解析、防火墙及节点动态端口都要可达。`rostopic list` 成功只证明能查询 master，还需要实际接收消息。若 RViz 直接运行在机载端，沿用已经验证的 ROS 环境，不要盲目覆盖网络变量。

对已经启动坐标适配器的系统：

```bash
rostopic type /rob2nav/cloud_world
rostopic echo -n 1 /rob2nav/cloud_world/header
rostopic hz /rob2nav/cloud_world
# 观察后 Ctrl+C；再检查位姿
rostopic echo -n 1 /rob2nav/body_pose/header
rviz
```

| RViz 项目 | 设置 |
| --- | --- |
| Fixed Frame | 适配器正常输出时为 `nav_odom` |
| PointCloud2 | `/rob2nav/cloud_world` |
| Odometry | `/rob2nav/body_pose` |
| TF | `/tf`、`/tf_static` |
| 轨迹 / 障碍物 | 以真机 `rostopic list`、`rostopic type` 的实际结果添加 Path / Marker / MarkerArray |

这些接口来自仓库部署适配器，不是本次真机观测。如果目前只启动 LIO，应使用 LIO 实际发布的话题及 `header.frame_id`。不要为了消除 RViz 报错而添加不真实的静态 TF。服务器仿真的 RViz2 配置不能直接当成 Noetic RViz 配置使用。

## 4. 现场端：VNC 与反向 SSH

先选定现场机器上的图形会话。已有桌面可使用支持共享该会话的 VNC 服务；没有桌面则先配置虚拟会话并验证 RViz。不同系统、X11 / Wayland 和 JetPack 的启动方式不同，本文不提供未经现场验证的统一 VNC 启动命令。

以下连接命令的前提是：

- 现场 VNC 已启用认证，并监听 `127.0.0.1:5901`。
- RViz 位于该 VNC 对应的图形会话，现场已经能看见真实数据。
- 现场机器能够主动 SSH 登录云服务器。

现场检查：

```bash
ss -lnt | grep ':5901'
# 应看到回环地址监听；端口若不同，下方一起替换。
read -rp '云服务器地址: ' CLOUD_HOST
read -rp '云服务器 SSH 端口: ' CLOUD_SSH_PORT
read -rp '云服务器 SSH 用户: ' CLOUD_USER
ssh -p "$CLOUD_SSH_PORT" -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 127.0.0.1:15901:127.0.0.1:5901 \
  "${CLOUD_USER}@${CLOUD_HOST}"
```

保持该终端运行；密码在 SSH 提示时输入。首次连接需按正常流程核对服务器主机密钥。服务器 `127.0.0.1:15901` 将转发到现场 VNC。无需把 VNC 端口开放到公网，也无需服务器主动连接机器人内网 IP。

`ExitOnForwardFailure` 能检测远程监听建立失败，不能证明 VNC 后端或 RViz 正常。实际仍需连接 viewer 验证。断线后需重新建立隧道；长期运行可在首次验收后配置受管理的 SSH 服务。

### 云服务器：打开画面

在云服务器执行（viewer 尚未安装时才安装）：

```bash
sudo apt update
sudo apt install tigervnc-viewer
ss -lnt | grep ':15901'
```

进入平台原有云桌面，在其终端执行：

```bash
vncviewer 127.0.0.1::15901
```

`::15901` 表示 TCP 端口，而不是 VNC 显示编号。若通过 SSH 启动 viewer，本次检查环境对应 `DISPLAY=:20`；仍需使用有权访问该 X 会话的用户及其 Xauthority，不要关闭 X 访问控制来规避认证问题。

平台桌面入口优先。若环境允许直接转发 Selkies 的 HTTP / WebSocket 服务，可在 Windows PowerShell 执行：

```powershell
$CloudHost = Read-Host '云服务器地址'
$CloudSshPort = Read-Host 'SSH 端口'
$CloudUser = Read-Host 'SSH 用户'
ssh -p $CloudSshPort -NT -o ExitOnForwardFailure=yes `
  -L 127.0.0.1:13000:127.0.0.1:3000 "${CloudUser}@${CloudHost}"
```

然后浏览器打开 `http://127.0.0.1:13000`。HTTP 200 只证明页面可访问；WebRTC 媒体可能还需要额外网络通道。若页面能打开但桌面无法连接，优先使用平台提供的完整桌面入口。本次未完成浏览器桌面端到端验证。

## 5. 验收与常见故障

按层验证，避免将显示问题与 LIO 故障混为一谈：

1. 现场 RViz 显示原始点云，确认数据进入 ROS。
2. LIO 连续输出位姿、更新点云；记录消息频率、时间戳及连续日志。
3. 坐标适配后的点云、位姿与 TF 一致，再显示 SCAN 轨迹。
4. 云服务器 VNC Viewer 显示同一现场会话，确认画面持续更新。
5. 断开隧道，确认机载导航链未因显示连接消失而退出；机器人停止依赖已有机载控制和急停机制。

| 现象 | 检查 |
| --- | --- |
| `remote port forwarding failed` | 服务器端口是否占用、SSH 是否允许转发 |
| viewer 连接失败 | 现场 VNC 监听、隧道是否仍在运行、VNC 认证 |
| 能看到桌面但没有 RViz | RViz 是否运行于该 VNC 会话的 DISPLAY |
| RViz 黑屏 / OpenGL 错误 | 现场显示驱动及虚拟会话 OpenGL；必要时测试软件渲染并评估 CPU 负载 |
| 有话题名却没有数据 | ROS1 节点地址与动态端口、publisher 是否正在发布 |
| Fixed Frame 不存在 | 消息 `frame_id`、实际 TF 和适配器状态 |
| 点云延迟明显 | 先看现场是否延迟；远程才延迟时降低桌面分辨率和 RViz 显示帧率 |
| LIO 持续报时间范围错误 | 检查 IMU / LiDAR 时间基准、有效缓存及初始化旧帧；远程显示不能修复它 |

LIO 排查应同时记录 `t1`、`t2`、`min_time`、`max_time`、`head_`、`count_`。不能仅凭历史日志认定旧帧积压是唯一根因，也不能伪造时间戳或关闭范围检查来获得可视化结果。

## 6. 可选部署与参考

### 让RViz进程真正运行在服务器

```text
Go2 EDU ROS1 ⇄ 双向可达网络 ⇄ 服务器 Ubuntu 20.04 / Noetic 环境
                                           └→ ROS1 RViz → 云桌面
```

需要先具备以下条件：

1. 准备可维护的 Noetic 隔离环境或独立虚拟机，并验证其图形显示。不要把 Noetic 软件包直接混装进当前 Ubuntu 24.04 / Jazzy。
2. 建立双方 ROS 节点可达的专用网络，正确配置 `ROS_MASTER_URI` 和 `ROS_IP` / 主机名。**仅转发 11311 不够**：master 用于发现，节点还使用动态端口传输消息。
3. 确认云平台是否支持所需 VPN / 网络设备及容器能力。当前未发现 TUN 和 Docker / Podman，普通 VPN + Docker 教程不能直接套用；用户态网络方案也需要单独验证节点双向可达性。
4. 控制可视化点云的带宽，在独立显示话题上降采样或限频，保留机载原始导航输入。
5. 验证消息类型、TF、时间戳和所需 RViz 插件。机器人与服务器使用 ROS1 时不必引入 ROS1 / ROS2 桥；若坚持使用 RViz2，需要另行验证桥接环境和消息兼容性。

该方案能利用服务器渲染资源，但当前前置工作明显更多。短期排查先采用第 2–6 节的桌面转发方案。

### 参考

- [TigerVNC 官方说明与手册](https://tigervnc.org/)
- [OpenSSH 官方手册](https://www.openssh.org/manual.html)
- [ROS1 网络配置](https://wiki.ros.org/ROS/NetworkSetup)
- [SCAN-Planner 上游](https://github.com/wuyi2121/SCAN-Planner)

本次交付范围：方案、连接命令、前置条件和验收步骤。服务器基础环境已做只读检查；现场 VNC 配置、Go2 连接、实时话题和整条远程显示链仍待现场验收。
