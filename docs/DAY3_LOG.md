# Day 3 日志 — 任务场景搭建

> **日期**: 2026-07-13  
> **目标**: 在 Isaac Sim 中搭建桌面操作场景 + Gym 风格环境封装  
> **结果**: ✅ 全部完成，7/7 测试通过

---

## 最终效果

Isaac Sim 视口中可见：

```
  ┌──────────────────────────────────────────┐
  │              目标板 (金色圆盘, r=10cm)      │
  │                                          │
  │    🔴 红方块    🔵 蓝方块    🟢 绿方块      │  ← 桌面 (70×50 cm, 高 25cm)
  │                                          │
  └──────────────────────────────────────────┘
            Franka Panda 机械臂
          （IsaacEnv 创建时加载）
```

- 相机在后方高处 (0.30, 0, 1.10)，俯瞰工作区
- 方块是动态刚体，可以被机械臂抓取推移
- 桌面和目标板是静态刚体

---

## 新增文件

| 文件 | 说明 |
|------|------|
| `src/envs/task_scene.py` | 场景构建器：桌面、方块、目标板、RGB + Depth 相机 |
| `src/envs/isaac_env.py` | Gym 风格环境封装：`reset()` / `step()` / `close()`，含 reward 设计 |
| `src/envs/__init__.py` | 导出 IsaacEnv + 场景构建函数 |
| `scripts/03_test_task_scene.py` | 7 项集成测试，全部通过 |

## 修改文件

| 文件 | 变更 |
|------|------|
| `TASKS.md` | Day 3 全部勾选完成 |

---

## 场景架构

### USD Stage 结构

```
/World/
├── GroundPlane              # 地面 (z=0)
├── DistantLight             # 光源 (强度 3000)
├── Table                    # 桌面 (kinematic 刚体, 70×50×2 cm)
├── Cube0 (红色)             # 3 cm³ 动态刚体，质量 27g
├── Cube1 (蓝色)
├── Cube2 (绿色)
├── TargetPad/               # 目标板 (金色圆盘, 半径 10cm, 厚 5mm)
│   └── Disc                 #    kinematic 刚体，平放桌面
├── CameraRGB                # 640×480, 焦距 18mm
└── CameraDepth              # 640×480
```

### 环境接口

```python
from envs.isaac_env import IsaacEnv

env = IsaacEnv(simulation_app=simulation_app)

# reset — 随机化方块位置，机械臂回到 home 位姿
obs = env.reset(seed=42)
# obs = {
#     "joint_positions":  (9,)  float32,   # 关节角度 [rad]
#     "joint_velocities": (9,)  float32,   # 关节速度 [rad/s]
#     "ee_position":      (3,)  float32,   # 末端位置 [m]
#     "ee_orientation":   (4,)  float32,   # 末端姿态四元数 (w,x,y,z)
#     "gripper_width":    (1,)  float32,   # 夹爪开度 [m]
#     "rgb":              (480, 640, 3) uint8,
#     "depth":            (480, 640, 1) float32,
# }

# step — 执行动作，返回 (obs, reward, terminated, truncated, info)
action = obs["joint_positions"][:7] + noise
obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Reward 设计（密集 → 稀疏）

| 阶段 | 条件 | Reward |
|------|------|--------|
| Reach | 末端到最近方块的距离 | `-L2_distance` |
| Grasp | 末端靠近方块 + 夹爪闭合 (< 2cm) | `+1.0` |
| Lift | 方块被抓起高于桌面 5cm | `+2.0` |
| Place | 抓取方块放到目标板上 | `+5.0` |
| Success | 任意方块在目标板上 | `+10.0` |

---

## 踩坑记录

| # | 问题 | 根因 | 解决 |
|---|------|------|------|
| 1 | `ModuleNotFoundError: No module named 'pxr'` | `pxr` (USD Python binding) 必须在 `SimulationApp` 启动后才能导入；顶层 import 在 `SimulationApp` 创建前执行 | 把 `from pxr import ...` 移到每个函数内部（懒加载） |
| 2 | 所有物体堆叠在原点 | 使用 `_get_pxr()` 懒加载辅助函数 + `as` 别名（如 `from pxr import Gf as _Gf`），导致 USD C++ 绑定的 `Vec3f`/`Vec3d` 构造返回零向量，所有坐标变成 (0,0,0) | 改为每个函数内直接写 `from pxr import Gf, UsdGeom`，取消别名和辅助函数 |
| 3 | 桌面是预期的 2 倍大，桌腿在桌面内部 | USD Cube 默认大小是 [-1,1]³（边长 2m），`SetScale(w, d, t)` 应该传半尺寸 `(w/2, d/2, t/2)`，之前传了全尺寸 | 修正所有 `SetScale` 参数为半尺寸 |
| 4 | `XformCommonAPI.SetTransform()` 不存在 | `XformCommonAPI` 没有直接设置 4×4 矩阵的方法，`SetTransform` 只存在于 `UsdGeom.XformOp` | 改用 `ExtractTranslation()` + `ExtractRotation().Decompose()` 分解为平移 + Euler 角，分别调用 `SetTranslate` + `SetRotate` |
| 5 | 目标板垂直于桌面竖立 | USD Cylinder 默认高度沿 Y 轴，世界坐标系 Z 轴朝上，圆柱体躺在桌面（沿 Y 轴水平）。不加旋转时侧面朝上 | 绕 X 轴旋转 90°，使圆柱体高度方向对齐世界 Z 轴 |
| 6 | conda Python 3.14 与 Isaac Sim Python 3.12 冲突 | `python.sh` 检测到 conda 环境激活时会报 warning 并加载错误库 | 运行前 `conda deactivate`，继承 Day 2 的 `setup_ros2.sh` 方案 |

---

## 设计与架构决策

### 1. 目标板代替篮子

最初设计了一个多层的圆柱体篮子（外墙 + 地板 + 内衬），但视觉上不像篮子且结构复杂。

改为**扁平圆盘目标板**：
- 更直观——方块放上去即成功，无视觉遮挡
- 匹配标准操作基准（Ravens, RLBench 使用目标板）
- 更少几何体和更简单的物理

### 2. 删除桌腿

桌面本身是 kinematic 刚体，固定在正确高度即可，桌腿没有功能价值（不参与碰撞/抓取）。删掉减少渲染负担。

### 3. XformCommonAPI > AddScaleOp/AddTranslateOp

高层 `XformCommonAPI` 自动管理 xform op 的顺序和去重，`AddScaleOp().Set()` 每次调用都会追加新的 op（可能产生重复变换）。

### 4. reward 和 success 合并计算

最初 `_compute_reward()` 和 `_check_success()` 各自独立计算方块-目标距离（重复工作）。合并为 `_compute_reward_and_done()` 一次算完。

---

## 代码审查改进

通过 4 个并行 agent 审查（Reuse / Simplification / Efficiency / Altitude），发现并修复了以下问题：

| 类别 | 问题 | 修复 |
|------|------|------|
| 重复计算 | reward 和 success 各自算距离 | 合并为一个方法 |
| 内存泄漏 | `close()` 不清空 `_ros2_node`，ROS2 timer 不取消 | 取消 timer + `self._ros2_node = None` |
| 死代码 | `get_latest_rgb/depth` 永远返回 None | 删除，等 Day 4 再实现 |
| 硬编码 | 目标板坐标写死在两处 | 从 `self._scene["config"]` 读取 |
| 性能 | 每步分配 2.1MB 零数组 | `__init__` 预分配 |
| 性能 | 每步 null-check `_simulation_app` (~2000次/ep) | 构造时绑定 `self._app_update` |
| 性能 | 随机化用 `np.sqrt()` 判距 | 改用平方距离比较 |

---

## 正确运行方式

```bash
# 终端 1 — 场景预览（静态，无机器人）
conda deactivate
source /opt/ros/jazzy/setup.bash
~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh src/envs/task_scene.py

# 终端 2 — 集成测试（无头模式，含机器人）
conda deactivate
source /opt/ros/jazzy/setup.bash
~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/03_test_task_scene.py --headless

# GUI 模式测试
~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/03_test_task_scene.py
```

---

*Day 3 完成 | 2026-07-13*
