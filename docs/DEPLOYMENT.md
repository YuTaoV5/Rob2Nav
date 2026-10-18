# 新环境部署与迁移

本说明与 `full-archive-2026-10-19` 数据归档对应。源码、网页、完整模型/记录/视频都可迁移；NVIDIA驱动、Isaac Sim/Isaac Lab和虚拟环境需在新机器重新安装。当前任务探索已暂停，安装、恢复数据与打开网页均不会恢复训练。原AME 20cm窄桩尚未实现稳定通过。

## 1. 选择部署范围

| 目的 | 系统/硬件 | 所需环境 | 数据 |
|---|---|---|---|
| 浏览全部证据、视频、配置、URDF与曲线 | Windows / Linux，现代浏览器；不需要训练GPU | 独立Web Python + Node.js | Git仓库的精选资料；需要时取回完整归档 |
| 运行MuJoCo目标导航与NJU冻结技能仿真 | 推荐Linux x86_64；CPU可运行，离屏渲染需要EGL | 独立MuJoCo Python 3.11 | 仓库内Go2/NJU模型；完整历史数据可选 |
| MoE感知策略训练/Isaac原生评估 | Linux x86_64 + 支持Isaac的NVIDIA GPU/驱动 | Isaac Sim 5.1.0、Isaac Lab v2.3.2、CUDA版torch、rsl-rl-lib 3.1.2 | 基座已在vendor；Isaac机器人资产按原条款另行获取 |

既有实测服务器为NVIDIA L40（约46GB显存）、驱动580.126.09；这不是最低显存承诺。512环境曾运行过，但环境数/相机/录屏改变后必须重新检查显存。完整依赖观测在Release的 `environment-*-freeze.txt`；这些是环境记录，含平台/本地路径，**不要在另一平台直接完整pip安装**。

## 2. 克隆源码并恢复资料

推荐布局：

```text
/srv/robotics/                 # Windows可用D:/robotics
  Rob2Nav/                    # 当前源码、artifacts、outputs
  MoE-runs/                   # 完整MoE实验目录
  AME-runs/                   # AME原生记录
  AME-criticaldamped/          # 固定上游源码/资源快照
  MoE-upstream/               # 原MoE参考源码快照
  envs/                       # 在新机器创建，归档不携带环境二进制
```

```bash
git clone https://github.com/YuTaoV5/Rob2Nav.git /srv/robotics/Rob2Nav
cd /srv/robotics/Rob2Nav
python3 scripts/check_vendor.py
python3 scripts/check_nju.py
# 查看全部资料分组，不下载大分包
python3 scripts/restore_release.py --destination /srv/robotics --list
# 取回所有资料，自动检查每包和每个解压文件的SHA-256
python3 scripts/restore_release.py --destination /srv/robotics
# 独立复核已恢复文件
python3 scripts/restore_release.py --destination /srv/robotics --verify-restored
```

`--destination`是克隆目录的**父目录**；分包内自带 `Rob2Nav/outputs/...` 与 `MoE-runs/...` 路径。各 `.partNNN.tar.gz` 都是可单独解压的tar包，不能用cat拼接。重复执行会复用校验通过的下载和已恢复文件；不同内容的已有文件默认拒绝覆盖，应使用新目标目录，或自行确认后加 `--overwrite`。下载缓存默认在父目录 `.rob2nav-downloads/`，可用 `--download-dir /data/cache` 改位置。

按需恢复例：

```bash
python3 scripts/restore_release.py --destination /srv/robotics \
  --group project-outputs --group moe-runs
```

只需历史回放时，Git clone已带精选权重/评估/视频，无需下载所有中间检查点。原始日志包含旧机器绝对路径，作为来源证据保留，不代表新机器存在对应目录。新任务应在网页重新选择归档检查点，或用新路径传CLI参数，不直接执行旧 `command.json`。

## 3. Web工作台环境（先只读启动）

需要Python 3.11及以上和Node.js 22 LTS（本项目已用Node 22.14验证）。Python/Node版本由系统包管理器或官方安装器安装；Linux可使用Miniconda/uv提供Python 3.11。

Linux：

```bash
cd /srv/robotics/Rob2Nav
python3.11 -m venv ../envs/rob2nav-studio
../envs/rob2nav-studio/bin/python -m pip install -r studio/requirements.txt
cd studio/frontend
npm ci
npm run build
cd ../..
../envs/rob2nav-studio/bin/python scripts/deployment_check.py --mode web
../envs/rob2nav-studio/bin/python -m studio.api --root "$PWD" --port 8765
```

Windows PowerShell（例 `D:\robotics\Rob2Nav`）：

```powershell
cd D:\robotics\Rob2Nav
py -3.11 -m venv ..\envs\rob2nav-studio
..\envs\rob2nav-studio\Scripts\python -m pip install -r studio/requirements.txt
Push-Location studio/frontend
npm ci
npm run build
Pop-Location
..\envs\rob2nav-studio\Scripts\python scripts/deployment_check.py --mode web
..\envs\rob2nav-studio\Scripts\python -m studio.api --root "$PWD" --port 8765
```

浏览 http://127.0.0.1:8765 。默认可浏览/编辑草稿，不执行任务。Windows适合浏览，GPU任务worker仅支持Linux。模型查看与地形预览无需启动物理模拟器。网页不会伪造缺失指标；历史没有状态记录时不提供完整三维关节回放。

## 4. MuJoCo环境与仿真

与Web/Isaac环境分离，避免NumPy、torch和物理引擎依赖互相覆盖。Linux需要EGL/OpenGL与ffmpeg（例如Ubuntu安装 `libegl1 libgl1 libglib2.0-0 ffmpeg`）。Python 3.11示例：

```bash
cd /srv/robotics/Rob2Nav
python3.11 -m venv ../envs/go2-mapless
../envs/go2-mapless/bin/python -m pip install --upgrade pip
# CPU版即可执行目前MuJoCo策略推理；不要把CPU torch装到Isaac环境
../envs/go2-mapless/bin/python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
../envs/go2-mapless/bin/python -m pip install -r requirements.txt
../envs/go2-mapless/bin/python scripts/deployment_check.py --mode mujoco
export MUJOCO_GL=egl
# 小型单回合检查，单独输出，不修改历史数据
../envs/go2-mapless/bin/python -m go2_nav.active_run --scene empty \
  --jump-backend disabled --max-seconds 2 --episodes 1 --video --output outputs/deploy_smoke
# 既有分层导航/技能组合复跑（非新感知MoE已过窄桩）
../envs/go2-mapless/bin/python -m go2_nav.active_run --scene course \
  --goal 13 0 --jump-backend nju --config configs/mission_runup.json \
  --episodes 3 --video --output outputs/deploy_course
```

输出目录需使用新名称。检查summary、轨迹与MP4；两秒检查通常超时，进程执行完成不等于导航成功。EGL不可用时可在图形桌面按MuJoCo官方方式使用GLFW，不把渲染失败解释成策略失败。

## 5. Isaac Lab训练环境

安装流程以[Isaac Lab v2.3官方安装文档](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)为准；先核对GPU/驱动兼容性。Isaac Sim首次启动的许可确认由部署者按原条款处理。可用pip环境，也可用独立Isaac Sim安装目录；不要把原服务器 `/root/isaacsim` 路径当成通用路径。

pip环境例（Linux x86_64）：

```bash
python3.11 -m venv /srv/robotics/envs/isaaclab
source /srv/robotics/envs/isaaclab/bin/activate
python -m pip install --upgrade pip
python -m pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git /srv/robotics/IsaacLab
cd /srv/robotics/IsaacLab
./isaaclab.sh --install rsl_rl
python -m pip install rsl-rl-lib==3.1.2
cd /srv/robotics/Rob2Nav
python scripts/deployment_check.py --mode isaac
```

官方版本标签、CUDA驱动和wheel可用性应在新机器核对；Release记录的是旧机器的实际环境，不表示这里已经重建并跑完新GPU训练。`isaaclab`包的内部版本0.54.2与框架发行版v2.3.2属于不同版本标识。

原生评估例（需先完成Isaac首次启动和机器人资产下载）：

```bash
cd /srv/robotics/Rob2Nav
# --python显式指定新机器的Isaac解释器，覆盖历史默认路径
/srv/robotics/envs/rob2nav-studio/bin/python -m moe_perceptive.suite \
  --python /srv/robotics/envs/isaaclab/bin/python \
  --moe vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt \
  --num-envs 32 --output outputs/deploy_moe_baseline
```

仅在计划明确恢复训练后，可用新目录做小预算接口检查（这会真实运行GPU训练，安装部署本身不执行）：

```bash
python -m moe_perceptive.train --headless \
  --moe vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt \
  --kind flat --num-envs 64 --iterations 2 --seed 42 \
  --output outputs/new_training_smoke
```

有意开始新训练时，先看 [MoE结构与协议](MOE_PERCEPTIVE.md)、[V4及后续失败证据](../V4_VALIDATION.md)；新训练、恢复优化器、只初始化权重不能混淆。完整地图教师必须保持特权标签，不能把它当成相机/雷达可部署学生。AME原项目通过 `critical_ame` 在隔离源码路径运行，固定上游快照在 `AME-criticaldamped/`；详见 [AME复现步骤](CRITICAL_AME_REPRODUCTION.md)。

## 6. 网页连接仿真与训练解释器

Linux上关闭只读服务，再明确启用任务提交：

```bash
cd /srv/robotics/Rob2Nav
../envs/rob2nav-studio/bin/python -m studio.api --root "$PWD" --port 8765 \
  --allow-training \
  --sim-python /srv/robotics/envs/isaaclab/bin/python \
  --mujoco-python /srv/robotics/envs/go2-mapless/bin/python
```

`--allow-training`只允许网页显式提交，不启动队列或探索自动化。创建实验页顶部在有预检/运行/评估任务时显示进展。原AME、NJU原训练等尚未接通的适配器仍不可提交；不要把页面模板当作训练实现。

远程机器只绑定loopback，通过SSH密钥或交互式密码建立隧道：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:8766:127.0.0.1:8765 -p <SSH_PORT> <USER>@<HOST>
```

本机打开 http://127.0.0.1:8766 。不要把SSH密码或GitHub token写入脚本。恢复旧SQLite队列保留历史任务状态；启动前确认没有在途任务，unknown状态需核对，不能直接重新执行。

持久服务可用tmux运行上述命令；或复制 [systemd示例](../deployment/rob2nav-studio.service)，修改User、WorkingDirectory和解释器绝对路径后按系统规范安装。默认示例为只读；启用提交需明确增加两个执行解释器与 `--allow-training`。

## 7. 验收与故障定位

```bash
# 独立Web环境还需安装pytest，测试不启动GPU训练
../envs/rob2nav-studio/bin/python -m pip install pytest
../envs/rob2nav-studio/bin/python -m pytest studio/tests tests/test_release_tools.py -q
curl http://127.0.0.1:8765/api/health
```

验收：历史报告与MP4能打开；训练曲线来自原日志；Go2网格和12关节加载；创建实验地形预览有原尺寸/前置标签；只读页不启动训练；先用MuJoCo小回合检查录屏。完整GPU训练与恢复检查点仍须在新硬件独立验证，不算在归档校验内。

| 现象 | 检查 |
|---|---|
| 网页503、空白 | 根目录正确；先在studio/frontend执行npm ci与npm run build；检查/api/health |
| 浏览器连不上远程页面 | 服务是否运行、SSH隧道是否存在；8766是本机隧道端口 |
| 缺模型或曲线 | 运行check_vendor/check_nju；恢复正确分组；TensorBoard依赖存在；点击刷新索引 |
| 旧命令中的/root路径不存在 | 重新指定--root/--python/--sim-python/--mujoco-python/--upstream，不原样执行历史command.json |
| CPU torch导致Isaac训练失败 | 激活Isaac环境，核对torch.cuda.is_available()和CUDA wheel；不能混用MuJoCo CPU环境 |
| 评估进程退出0但无summary | 检查failed.json、completed.json、最终checkpoint；不认退出码为成功 |
| 归档校验失败 | 不使用文件；重新下载该分包；核对Release SHA256SUMS和data-manifest |
| unknown阻止任务 | 核对旧worker/PID/进程是否仍存活，再人工处理；不要绕过并发保护 |

框架/上游源码、资产与派生模型保留各自条款，见 [第三方说明](../THIRD_PARTY_NOTICES.md)。归档包含成功、失败与停滞实验；不能将前置宽桩成绩或训练完成标记当成原20cm窄桩通过。
