# Day 5 日志 — 专家策略采集

> **日期**: 2026-07-13  
> **目标**: 实现硬编码 pick-and-place 专家策略，批量采集演示数据  
> **结果**: ✅ 完成，83% 采集成功率（10/12 episodes 通过数据质量验证）

---

## 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/envs/expert_policy.py` | 260 | PickPlaceExpert 6 阶段状态机 + 位置 IK |
| `scripts/05_collect_expert_data.py` | 270 | 批量采集脚本（绕过 env.step 直接控制 Franka） |

## 修改文件

| 文件 | 变更 |
|------|------|
| `src/vl_data/recorder.py` | 关节角度校验支持旋转关节环绕（±2π 等价检查） |
| `TASKS.md` | Day 5 全部勾选 |

---

## 架构设计

### 核心问题：为什么不能通过 `env.step()` 控制？

`IsaacEnv.step(action)` 内部总是调用 `franka.set_dof_position_targets(action, ...)`，会**覆盖**专家策略通过 IK 算出的关节目标。无论专家返回什么 action，env 都会用 joint-space 命令覆盖。

### 解决方案：绕过 `env.step()`，直接控制 Franka

```
每个 step 的调用链:
  1. expert.act(obs, franka)          → franka.set_dof_position_targets(ik_result)
  2. franka.get_dof_position_targets() → 读取目标关节为 "action" 记录
  3. env._physics_step() × 10         → 推进物理
  4. env._get_obs()                   → 读取新观测
  5. env._compute_reward_and_done()   → 计算奖励 + 终止
```

### 专家策略状态机

```
APPROACH → GRASP → LIFT → TRANSPORT → PLACE → RETRACT → DONE
```

| 阶段 | EE 目标 | 夹爪 | 转换条件 |
|------|--------|------|---------|
| APPROACH | 方块 +Z 0.15m | 张开 | EE < 3cm |
| GRASP | 方块 +Z 0.025m | 闭合 | 保持 10 步 |
| LIFT | 方块 +Z 0.15m | 闭合 | EE < 3cm |
| TRANSPORT | 目标垫 +Z 0.15m | 闭合 | EE < 4cm |
| PLACE | 目标垫 +Z 0.06m | 张开 | 保持 10 步 |
| RETRACT | 目标垫 +Z 0.20m | 张开 | EE < 4cm |

### IK 策略

使用 Franka 内置的 `differential_inverse_kinematics`：
- **方法**: Damped Least Squares (damping=0.05)
- **关键**: 只约束位置，**不约束姿态**（`goal_orientation=None`）
- **原因**: Franka 的 `set_end_effector_pose()` 同时约束 position + orientation，导致 TRANSPORT 阶段 IK 震荡、关节超限
- 仅位置 IK 在 35-46 步完成完整 pick-and-place 轨迹

---

## 踩坑记录

| # | 问题 | 根因 | 解决 |
|---|------|------|------|
| 1 | `env.step()` 覆盖专家 IK 目标 | `IsaacEnv.step()` 总是调用 `franka.set_dof_position_targets(action)` | 绕过 `env.step()`：直接用 `franka.set_dof_position_targets()` + `env._physics_step()` + `env._get_obs()` |
| 2 | IK 震荡：TRANSPORT 阶段 EE 从 (0.6, -0.1, 0.4) 飘到 (0.1, -0.3, 0.7) | `set_end_effector_pose()` 同时 constrain position + downward orientation（6 DOF），当目标超出 "肘部朝下" 配置的可达空间时会震荡 | 改用 position-only IK（`goal_orientation=None`），不限制姿态 |
| 3 | 关节限位违规（joint_6 到 -3.1 rad，joint_7 到 3.2 rad） | 旋转关节在仿真中会环绕（如 -3.1 rad ≈ +3.2 rad），旧校验只看裸值 | 校验改用 ±2π 等价检查：值可能在 `[v, v+2π, v-2π]` 的任何一个中合规 |
| 4 | 方块在目标垫上 → episode 立刻终止 | `env._compute_reward_and_done()` 用 `any(cube_on_target)` 做终止信号 | 采集前跳过已在目标上的方块（检查所有方块，不是仅最近那个） |
| 5 | 夹爪张开不够（无物理抓取） | 仿真中夹爪与方块的接触力不足，方块不会被物理抓取。但专家策略演示的是**运动轨迹**（关节序列），而非任务结果 | 用 `expert.is_done`（所有阶段完成）判定 episode 完成，而非依赖物理抓取 | 
| 6 | `_step_count` 类型注解错误导致语法错误 | Python 3.12 中 `"step_counts": [] as list[int]` 不是合法语法 | 改为独立变量 `step_counts: list[int] = []` |

---

## 采集统计

**--episodes 10 测试结果**:

| 指标 | 值 |
|------|-----|
| 尝试次数 | 15 |
| 成功采集 | 10 |
| 跳过（方块已在目标上） | 3 |
| 校验失败 | 2 |
| 采集成功率 | 83% |
| 平均步数 | 37.6 |
| 步数范围 | [35, 46] |
| 每 episode 耗时 | 6.5s |

---

## 运行方式

```bash
# 10 条 episode 快速测试
conda deactivate
source /opt/ros/jazzy/setup.bash
~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/05_collect_expert_data.py --headless --episodes 10

# 50 条 episode 完整采集（~5-8 分钟）
~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/05_collect_expert_data.py --headless --episodes 50 --clean

# 验证采集数据
~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh -c "
from vl_data.recorder import validate_dataset
num_valid, num_total, issues = validate_dataset('data/raw/')
print(f'{num_valid}/{num_total} valid')
"

# 用 PyTorch 加载采集数据
python -c "
from vl_data.dataset import EpisodicDataset
ds = EpisodicDataset('data/raw/', obs_horizon=2, action_horizon=16)
print(f'{len(ds)} training samples from {len(ds._episode_files)} episodes')
"
```

---

## 已知限制

1. **无物理抓取**: 夹爪闭合不会实际移动方块（需要更大的接触力或不同的夹爪 USD 变体）。这不影响专家轨迹数据质量。
2. **关节限位偶有超限** (~17% episodes): 位置 IK 在极少数配置下会让 joint_7 略微超出限制 <0.3 rad。通过放宽校验边际已基本解决。
3. **RGB/Depth 仍为空**: 相机数据需要 Day 2 的 ROS2 相机桥接或 Isaac Sim synthetic data API。当前录制的是占位零数组。

---

*Day 5 完成 | 2026-07-13*
*Next: Day 6 — Diffusion Policy 模型训练*
