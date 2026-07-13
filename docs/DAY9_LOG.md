# Day 9 日志 — Isaac Sim 推理与评估

> **日期**: 2026-07-13
> **目标**: 在 Isaac Sim 中加载训练好的模型，运行 rollout 评估成功率
> **结果**: ✅ 完成 — 双进程 rollout 架构工作正常，发现无视觉模型的关键局限

---

## 架构设计

### 双进程 IPC 架构

Isaac Sim Python 3.12 没有 PyTorch（pip install 卡住），采用文件 IPC 方案：

```
┌─────────────────────────────┐     ┌──────────────────────────┐
│  Isaac Sim Python (3.12)    │     │  conda vlarm (Python 3.11)│
│                             │     │                          │
│  scripts/09_rollout.py      │     │  scripts/09_model_server │
│                             │     │                          │
│  ┌───────────────────────┐  │     │  ┌────────────────────┐  │
│  │ IsaacEnv              │  │     │  │ DiffusionPolicy    │  │
│  │ (simulation)          │  │     │  │ (3.23M params)     │  │
│  └───────┬───────────────┘  │     │  └─────────┬──────────┘  │
│          │                  │     │            │              │
│          │ observation      │     │            │              │
│          ▼                  │     │            ▼              │
│  ┌───────────────┐          │     │  ┌────────────────────┐  │
│  │ ModelClient   │───IPC──▶ │     │  │ DDPM (100 steps)   │  │
│  │ (file writer) │◀──IPC─── │     │  │ predict_action()   │  │
│  └───────┬───────┘          │     │  └────────────────────┘  │
│          │ action_traj       │     │                          │
│          ▼                  │     │                          │
│  ┌───────────────┐          │     │                          │
│  │ Execute +     │          │     │                          │
│  │ Step physics  │          │     │                          │
│  └───────────────┘          │     │                          │
└─────────────────────────────┘     └──────────────────────────┘
         ▲                                    ▲
         │ /tmp/vlarm_server/                 │
         │   request.npz  ← observation       │
         │   response.npy ← action_traj       │
         │   ready         ← flag             │
         └────────────────────────────────────┘
```

**通信流程**:
1. Isaac Sim 收集 2 帧观测 → 写入 `request.npz`
2. 模型服务轮询检测到 request → 加载 → DDPM 推理 → 写入 `response.npy`
3. 创建 `ready` 标志文件
4. Isaac Sim 检测到 `ready` → 读取 action → 执行 → 继续下一步

### Receding-Horizon 控制

```
Model predicts 16 future actions
         │
         ▼
Execute first 8 actions  ──→  Gather new observations  ──→  Re-predict
```

每 episode 约 19 次预测（300 步 / 8 exec_horizon ≈ 38，实际=19 因为有 action buffer）。

---

## 评估结果

### 模型

| 参数 | 值 |
|------|-----|
| Checkpoint | `checkpoints/20260713_165202/best.pt` |
| Epoch | 415 |
| Val loss | 0.089 |
| 参数量 | 3.23M |
| 推理方式 | DDPM 100-step |
| 推理速度 | **418 ms/prediction** (IPC overhead ~150ms) |

### Rollout 结果

| Episode | Seed | Steps | 结果 | EE→Cube | EE→Target |
|---------|------|-------|------|---------|-----------|
| 1 | 42 | 300 | ❌ | 0.40m | 0.15m |
| 2 | 100 | 300 | ❌ | 0.25m | 0.14m |
| 3 | 101 | 300 | ❌ | 0.31m | 0.14m |
| 4 | 102 | 300 | ❌ | 0.30m | 0.14m |
| 5 | 103 | 300 | ❌ | 0.24m | 0.14m |

**成功率: 0/5 (0%)**

### 失败模式分析

所有 episode 表现出相同的模式：

```
EE 轨迹（所有 episode）:
  起点 (home) ──────────────────────→ 终点 (near target pad)
                                        ee→target ≈ 0.14m
  
方块:
  始终在初始位置不动，ee→cube ≈ 0.25-0.40m
```

**根因**: 训练时 `use_vision=False`，模型输入只有：
- `joint_positions` (9D)
- `ee_position` (3D)
- `gripper_width` (1D)

**没有方块位置信息**。模型学到的只是"从 home 位姿走到目标垫附近"的平均轨迹。所有 expert 演示的终点都是目标垫上方，所以模型学会了这个方向，但无法根据方块位置调整。

---

## 关键发现

### 1. 无视觉模型的局限

| 有方块位置 | 无方块位置 |
|-----------|-----------|
| 模型可根据 cube 位置调整轨迹 | 模型只能学平均轨迹 |
| 泛化到新 cube 位置 ✅ | 泛化失败 ❌ |
| 需要 RGB/Depth 或显式传入坐标 | 当前状态 |

### 2. 数据中其实有 cube 位置

每个 episode 的 `config_json` 中存储了完整的场景配置：

```json
{
  "cube_positions": [[0.5, -0.12, 0.275], [0.55, 0.0, 0.275], ...],
  "target_position": [0.7, -0.18, 0.263]
}
```

但 `EpisodicDataset` 没有把这个信息作为观测提供给模型。**修复方向**: 将最近的 cube 位置（3D）加入观测，state_dim 从 13 → 16。

### 3. 双进程 IPC 可靠

- 文件 IPC 简单可靠，无需安装额外依赖
- 推理延迟 <500ms（DDPM 100 步 + IPC 开销）
- 可用于 Day 10-14 的后续实验

---

## 新增/修改文件

| 文件 | 变更 |
|------|------|
| `scripts/09_rollout.py` | 新建 — Isaac Sim 端 rollout（ModelClient + GripperHeuristic） |
| `scripts/09_model_server.py` | 新建 — conda 端模型推理服务（文件 IPC） |
| `TASKS.md` | Day 9 全部勾选 |

---

## 下一步 (Day 10)

1. **添加方块位置观测**: 修改训练数据 pipeline，将最近的 cube 位置加入 state（13D → 16D），重新训练
2. **或集成视觉**: 使用真实相机数据（RGB → ResNet-18 编码器）
3. **语言集成**: 添加语言 embedding 作为条件输入

---

*Day 9 完成 | 2026-07-13*
*Next: Day 10 — 语言集成（语言编码器 + 多任务指令）*
