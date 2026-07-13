# Day 2 日志 — Isaac Sim + ROS2 桥接

> **日期**: 2026-07-13  
> **目标**: 打通 Isaac Sim 仿真 → ROS2 → Franka 机器人控制完整链路  
> **结果**: ✅ 全部完成

---

## 最终效果

三个终端协同工作，实现仿真内 Franka Panda 机械臂的 ROS2 控制：

```
终端 3 (action_pub)  ──发送 JointTrajectory──▶  终端 1 (Isaac Sim)
  "发送目标关节角度"                              Franka 动起来
                                                 发布 /joint_states
                                                发布 /ee_pose
                                                发布 /rgb /depth
                                                         │
终端 2 (state_sub)  ◀─────────────────────────────────────┘
  "实时显示关节角度和末端位姿"
```

---

## 新增文件

| 文件 | 说明 |
|------|------|
| `scripts/run_franka_bridge.py` | Isaac Sim 端：加载 Franka + ROS2 Bridge + 相机 |
| `src/ros2_bridge/action_pub.py` | 发布 `/joint_trajectory` 控制机器人（支持 --home / --pick / --place / --multi-step） |
| `src/ros2_bridge/state_sub.py` | 订阅 `/joint_states` + `/ee_pose`（支持 CSV 日志） |
| `src/ros2_bridge/camera_sub.py` | 订阅 `/rgb` + `/depth` + `/camera_info`（支持保存/显示） |
| `scripts/02_test_ros2_bridge.py` | 7 项集成测试，全部通过 ✅ |
| `scripts/setup_ros2.sh` | ROS2 环境助手（解决 conda Python 3.14 无法加载 rclpy 的问题） |

## 修改文件

| 文件 | 变更 |
|------|------|
| `TASKS.md` | Day 2 全部勾选完成 |
| `src/ros2_bridge/__init__.py` | 导出 ActionPublisher、StateSubscriber、CameraSubscriber |

---

## 关键架构设计

### 双 Python 环境

Isaac Sim 自带 Python 3.12 + 内部 rclpy，系统也装了 ROS2 Jazzy (Python 3.12)。conda 的 Python 3.14 无法使用 rclpy。

| 组件 | Python | rclpy |
|------|--------|-------|
| `run_franka_bridge.py`（Isaac Sim 内） | Isaac Sim 自带 3.12 | 系统 rclpy（`source /opt/ros/jazzy/setup.bash`） |
| `action_pub.py` / `state_sub.py`（外部） | `/usr/bin/python3.12` | 系统 rclpy |

### ROS2 Topic 设计

| Topic | 类型 | 方向 | 说明 |
|-------|------|------|------|
| `/joint_trajectory` | JointTrajectory | action_pub → Isaac Sim | 控制机器人关节运动 |
| `/joint_states` | JointState | Isaac Sim → state_sub | 9 个关节实时角度 |
| `/ee_pose` | PoseStamped | Isaac Sim → state_sub | 末端执行器位姿 |
| `/rgb` | Image | Isaac Sim → camera_sub | 640×480 RGB 图像 |
| `/depth` | Image | Isaac Sim → camera_sub | 深度图 |
| `/camera_info` | CameraInfo | Isaac Sim → camera_sub | 相机内参 |

---

## 踩坑记录

| # | 问题 | 根因 | 解决 |
|---|------|------|------|
| 1 | `import rclpy` 失败 | Isaac Sim 内部 rclpy 的 .so 依赖不完整，系统 ROS2 未 source | 启动前 `source /opt/ros/jazzy/setup.bash` |
| 2 | `No module named 'isaacsim.robot.experimental.manipulators'` | Franka 扩展未加载 | `app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")` |
| 3 | conda Python 3.14 无法 `import rclpy` | rclpy 只兼容 Python 3.12 | 编写 `scripts/setup_ros2.sh`，把 `/usr/bin` 加到 PATH 前面 |
| 4 | `ros2 --version` 报 unrecognized arguments | ros2 CLI 不支持 `--version` | 改用 `echo $ROS_DISTRO` |

---

## 正确运行方式

```bash
# 终端 1 — Isaac Sim
source /opt/ros/jazzy/setup.bash
~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/run_franka_bridge.py

# 终端 2 — 状态订阅
source scripts/setup_ros2.sh
python3 src/ros2_bridge/state_sub.py

# 终端 3 — 发送指令
source scripts/setup_ros2.sh
python3 src/ros2_bridge/action_pub.py --multi-step
```

---

*Day 2 完成 | 2026-07-13*
