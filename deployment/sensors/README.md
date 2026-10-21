# 传感器标定数据模板与离线检查

完整流程见[安装与标定总手册](../../docs/GO2_D435I_MID360_INSTALL_CALIBRATION.md)。

- `calibration.template.json`：必须填写真实序列号、校正针孔内参、米制外参、独立验证点、工作区样本、实测时间摘要；未填写时检查拒绝通过。
- `calibration.synthetic_60deg.json`：完全虚构的理想60°几何示例，不能上机，只用于测试矩阵方向和投影算法。
- `T_base_color_optical`将color optical坐标变到base，最后一行[0,0,0,1]；法向只使用旋转。其他矩阵的命名同理，`T_imu_lidar`是lidar到IMU。
- `holdout_pairs`是未参与拟合的相机米制点与独立base参考点；至少6个，但真实验收应覆盖左右/距离/多个高度。工具无法替你证明这些点实际独立。
- `workspace_points_base`应覆盖拟允许掌面及脚垫余量；工具只检查针孔/深度范围，不检查遮挡、深度噪声、腿可达性或接触力。
- `timing`来自录制日志的P99测量摘要，不能以配置目标值替代。

```bash
python3 scripts/check_sensor_calibration.py --input measured.json --output check.json
python3 scripts/check_sensor_calibration.py --input deployment/sensors/calibration.synthetic_60deg.json --allow-synthetic --output synthetic_check.json
```

退出码0仅表示提供的数据通过离线检查，2表示拒绝/失败；输出始终`hardware_release=false`。脚本不采集数据、不求解标定、无ROS/相机/SDK连接或电机操作。模板不是现有ROS启动参数文件，不能将字段名当作已经实现的驱动配置。
