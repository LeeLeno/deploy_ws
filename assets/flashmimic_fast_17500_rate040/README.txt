FlashMimic 17,500 更新 / action_rate_l2=-0.40 可选人工 A/B 版本

训练任务：g1_flash_mimic/mjwarp_fast；从头训练 17,500 更新；不加 action_acc_l2。
标准 body-frame gravity R^T*[0,0,-1]，prior 与 residual_scale=0.25 已内嵌 ONNX。
PD、关节顺序、动作尺度、观测及一拍延迟与已实测 baseline 完全相同。

此版降低动作跳变，但尚未证明整体实机效果更好：
无人工观测噪声、保留 DR 的全片评估，动态下肢 target delta RMS 7.627°→7.124°；
P95 14.525°→13.423°；踝 RMS 9.517°→8.941°；力矩指标 9.916→9.578 Nm。
代价：躯干姿态误差 4.793°→5.366°；右脚高 RMSE 16.66→18.71 mm。
含噪和无噪全片分别 256/256 完成。完整比较及训练随机性限制见 tuning_report.md。

现有 flashmimic_fast_17500 baseline 保持默认。本目录是独立候选，未自动启用。
人工测试时 policy.onnx、policy.onnx.data、deploy_config.yaml 与 dance1_subject2_part.bin
应作为完整匹配资产组使用；保留 baseline 以便比较。只替换这些资产不涉及 C++，无需重新编译，
由用户人工重新启动原有程序时读取。此处不包含启动器，不连接设备，不发送电机指令。
physical_robot_tested=false；本候选尚未经人工实机测试。
