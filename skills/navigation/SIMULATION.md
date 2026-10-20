# Go2 实时导航仿真手册

网页地址：/navigation-live，工作台导航页内嵌同一窗口。点击地图空地选目标，再点击开始；可在运行中改目标、停止和重置。绕障/空旷/完全阻塞场景可切换后重置。

实际验证分开记录：规则导航网页鼠标点击(2,.5)m与(7.5,1.6)m，两次到达且无碰撞、无跌倒，约1×实时；关闭页面2.5秒后停止。批量测试另运行空旷种子300–303与绕障400–403，共8回合到达并在零速度命令下持续站立2秒。空旷固定场景的多种子轨迹相同，不能解释为8个不同环境泛化样本。接口回归28项通过，包括禁止姿态修改和拒绝非有限目标。

```bash
cd /root/gpufree-data/Rob2Nav
PY=/root/gpufree-data/envs/go2-mapless/bin/python
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY -m skills.navigation.live --port 8768
```

该命令在前台启动。服务器8768已由supervisord管理时不要重复启动；查看/重启：

```bash
supervisorctl -c /root/gpufree-data/deployment/rob2nav-supervisord.conf status rob2nav-navigation
supervisorctl -c /root/gpufree-data/deployment/rob2nav-supervisord.conf restart rob2nav-navigation
tail -f /root/gpufree-data/deployment/navigation-live.log
```

Studio8765通过白名单代理连接物理worker8768，浏览器访问Studio而不是直接写worker。物理500Hz、底层策略50Hz、导航10Hz，显示约6–10Hz；实时倍率由实际仿真时钟与墙钟计算，负载变化会影响倍率。暂停/到达后仍持续计算站立物理，没有修改或锁定机器人根位姿。仅明确重置时重建场景。

本次PPO模型也完成了同样两种场景、另设种子1300–1303/1400–1403的8回合测试，均通过到达及2秒稳定停止。网页鼠标两目标和断线停止再次通过，state策略SHA256与交付文件一致；服务默认已切换为该PPO。规则与PPO的成绩分别保存在evidence/rule_*.json和evidence/ppo_*.json。

无窗口重复评估：

```bash
$PY -m skills.navigation.benchmark --scene empty --goal 2 .5 --episodes 4 --seed 300 --output outputs/nav_eval_empty_new
$PY -m skills.navigation.benchmark --scene slalom --goal 7.5 1.6 --episodes 4 --seed 400 --output outputs/nav_eval_slalom_new
```

验收要求：进入0.40m范围后发零速度并继续积分2秒；全回合无碰撞/跌倒，最终距离<0.55m、水平速度<0.20m/s。原始逐步trace和summary均保存。规则导航器只看到仿真雷达返回、机身里程计和目标；地图障碍真值仅供显示和拒绝落在障碍内的目标，不进入导航器。该实现不是SCAN或SEA原生算法。

验证本次训练PPO：

```bash
$PY -m skills.navigation.benchmark --checkpoint outputs/navigation_ppo_full_20261021/navigation.zip --config outputs/navigation_ppo_full_20261021/navigation_config.json --scene slalom --goal 7.5 1.6 --episodes 4 --seed 1400 --output outputs/nav_ppo_eval_new
```

直接使用仓库附带的已验证PPO策略打开独立窗口；使用新端口，不覆盖Studio服务：

```bash
$PY -m skills.navigation.live --port 8769 --checkpoint skills/navigation/policy/navigation.zip --config skills/navigation/policy/navigation_config.json
```

另开SSH隧道`ssh -N -L 8769:127.0.0.1:8769 -p 30752 root@120.209.70.195`，浏览器访问http://127.0.0.1:8769/。该独立窗口直接访问自身worker；Studio代理仍指8768。省略--checkpoint时为规则导航对照。页面显示实际后端，state包含已加载策略SHA256。未经评估的PPO可能碰撞/超时，不能将默认规则导航的成绩归给PPO。

超界/障碍内目标拒绝；碰撞或跌倒锁定故障；页面心跳失联2.5秒停止；单次导航最多120仿真秒。静态局部导航可能陷入局部最优，不提供动态人群、任意迷宫或未知地面的完成保证。
