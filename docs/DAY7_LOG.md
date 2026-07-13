# Day 7 日志 — 训练管道

> **日期**: 2026-07-13
> **目标**: 搭建完整的 Diffusion Policy 训练管道（配置系统 + 训练循环 + TensorBoard + Checkpoint）
> **结果**: ✅ 完成，全部 9 项任务通过

---

## 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `configs/train_config.yaml` | 80 | 训练超参数配置（扩散、数据、模型、日志） |
| `configs/robot_config.yaml` | 78 | Franka Panda 机器人属性配置 |
| `configs/task_config.yaml` | 92 | Pick-and-Place 任务与场景配置 |
| `src/utils/config.py` | 272 | Dataclass 配置加载器 + YAML/命令行合并 |
| `scripts/07_train.py` | 420 | 完整训练脚本（训练循环 + TensorBoard + Checkpoint） |

## 修改文件

| 文件 | 变更 |
|------|------|
| `TASKS.md` | Day 7 全部勾选 |

---

## 架构设计

### 配置系统

```
configs/train_config.yaml  ──┐
configs/robot_config.yaml  ──┤ (可选)
configs/task_config.yaml   ──┘
                               │
                    ┌──────────┴──────────┐
                    │  utils/config.py    │
                    │  load_config()      │
                    │  + CLI overrides    │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │  TrainConfig (dataclass) │
                    │  ├── diffusion           │
                    │  ├── data                │
                    │  ├── training            │
                    │  ├── lr_schedule         │
                    │  ├── model               │
                    │  ├── logging             │
                    │  └── seed                │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │  scripts/07_train.py │
                    └─────────────────────┘
```

支持嵌套点号覆盖：`--batch-size 32 --lr 1e-3` 自动映射到 `cfg.training.batch_size` 和 `cfg.training.lr`。

### 训练流程

```
数据加载:
  67 HDF5 episodes
    ├── 10 异常 episodes ──→ 过滤（关节超出 ±6.28 rad 或 success=False）
    └── 57 正常 episodes
           │
    1170 training samples (obs_horizon=2, action_horizon=16)
           │
    ┌──────┴──────┐
  Train (80%)   Val (20%)
  936 samples    234 samples

训练循环 (per step):
  1. 从 DataLoader 取 mini-batch: actions (B, 16, 7)
  2. 采样噪声: ε ~ N(0, I)
  3. 采样时间步: t ~ Uniform(0, 99)
  4. 前向扩散: a_t = √(ᾱ_t)·a_0 + √(1-ᾱ_t)·ε
  5. 模型预测: ε_θ = model(a_t, t, obs)
  6. 损失: MSE(ε_θ, ε)
  7. 反向传播 + 梯度裁剪 (max_norm=1.0)
  8. 学习率调度: warmup(5 epochs) + cosine annealing
  9. TensorBoard 记录: loss, lr, grad_norm

验证 (每 5 epochs):
  1. 切换到 eval 模式
  2. 计算 val loss (max 20 batches)
  3. 如果 val_loss 改善 → 保存 best.pt

Checkpoint:
  - best.pt:  最低验证 loss 的 checkpoint
  - last.pt:  最新 checkpoint（resume 用）
  - epoch_*.pt: 定期快照（每 20 epochs）
  - config.yaml: 配置快照（可复现性）
```

### 异常 Episode 过滤

从 67 个 episode 中自动识别并排除 10 个异常：

| Episode | 步数 | 关节范围 | 原因 |
|---------|------|---------|------|
| 002 | 300 (maxed) | [-2.5, 25.4] | 关节溢出 + 失败 |
| 006 | 300 (maxed) | [-10.5, 10.5] | 关节溢出 + 失败 |
| 007 | 300 (maxed) | [-2.2, 3.8] | 失败（但关节正常） |
| 013 | 41 | [-2.8, 9.0] | 关节溢出 |
| 014 | 45 | [-2.4, 7.3] | 关节溢出 |
| 022 | 300 (maxed) | [-5.4, 5.4] | 关节溢出 + 失败 |
| 023 | 82 | [-3.2, 24.2] | 关节溢出 |
| 046 | 300 (maxed) | [-2.4, 3.8] | 失败（但关节正常） |
| 051 | 300 (maxed) | [-2.3, 3.8] | 失败（但关节正常） |
| 066 | 300 (maxed) | [-2.4, 3.8] | 失败（但关节正常） |

> 注：007、046、051、066 关节数值在正常范围内但 `success=False`（可能是碰撞或 IK 失败导致提前终止）。

### 模型架构（无视觉模式）

由于当前无真实相机数据，训练使用 `use_vision=False`：

```
joint_positions (B, 2, 9)  ──┐
ee_position (B, 2, 3)       ──┤ last frame → (B, 13) → StateEncoder (3-layer MLP) → (B, 256)
gripper_width (B, 2, 1)     ──┘
                                                        │
                                         ObsProjection (256→512→256→256) → (B, 256)
                                                        │
noisy_actions (B, 16, 7) ─────────────────────┐        │
                                              │        │
timestep t (B,) → SinusoidalEmbedding(128) ──→│────→ concat → FiLM-conditioned
                                        time_proj │        1D Conv U-Net → ε_θ (B, 16, 7)
                                                  │
```

- **参数量**: 3.23M（含视觉时为 14.67M，ResNet-18 占 ~11M）
- **模型大小**: ~13 MB FP32

---

## 验证结果

### 快速训练测试（3 epochs, CPU）

```
Epoch 1: avg_loss=1.003  → val_loss=0.9915  (🏆 new best)
Epoch 2: avg_loss=0.987  → (no validation this epoch)
Epoch 3: avg_loss=0.681  → (no validation this epoch)
```

| 指标 | 值 |
|------|------|
| 初始 loss | 1.00（接近 1.0，预期行为） |
| 3 epoch 后 loss | 0.68（从 1.0 开始有明显下降） |
| 梯度范数 | 0.5 → 1.0（被 clip 在 1.0） |
| 每 epoch 耗时 | ~10.5s（CPU，117 batches × batch_size=16） |
| 学习率 | 1.72e-05 → 5.17e-05（warmup 阶段） |

### Checkpoint 恢复测试

- `--resume checkpoints/last.pt` ✅ 成功恢复 epoch/step/optimizer state
- 恢复后继续训练无异常

---

## 运行方式

```bash
# 基本训练（CPU）
conda activate vlarm
PYTHONPATH=src python scripts/07_train.py

# 自定义参数
PYTHONPATH=src python scripts/07_train.py --batch-size 32 --epochs 500 --lr 5e-5

# GPU 训练（自动检测）
PYTHONPATH=src python scripts/07_train.py --device cuda

# 恢复训练
PYTHONPATH=src python scripts/07_train.py --resume checkpoints/last.pt --epochs 500

# 查看 TensorBoard
tensorboard --logdir logs
```

---

## 关键设计决策

| # | 决策 | 理由 |
|---|------|------|
| 1 | 纯 PyTorch 训练循环（不用 PyTorch Lightning） | 项目规模适中，保持依赖最少；Lightning 对 Diffusion Policy 无特殊优势 |
| 2 | Dataclass 配置系统（不用 OmegaConf/Hydra） | 避免额外依赖，dataclass 提供 IDE 自动补全和类型检查 |
| 3 | Step-level warmup + cosine LR | Diffusion Policy 论文推荐；warmup 防止训练早期不稳定 |
| 4 | `use_vision=False` 默认 | 当前采集数据中 RGB 全为零（占位相机），视觉分支无意义 |
| 5 | 自动过滤异常 episode | 10/67 episodes 存在关节溢出或失败，需在训练前排除 |
| 6 | `drop_last=True` 训练 / `drop_last=False` 验证 | 避免最后一个不完全 batch 导致 BN-like 问题；验证需使用所有数据 |
| 7 | `weights_only=False` 在 checkpoint 加载 | PyTorch 2.13 默认安全模式会拒绝自定义类，训练 checkpoint 不受此限制 |

---

## 踩坑记录

| # | 问题 | 根因 | 解决 |
|---|------|------|------|
| 1 | 训练脚本需 `PYTHONPATH=src` 前缀 | conda 环境未安装 vlarm 包 | 在脚本内部自动添加 `src/` 到 `sys.path`（与 Day 6 一致） |
| 2 | `Subset` 对象缺少 `collate_fn` | `random_split` 返回的 `Subset` 不继承 dataset 方法 | 从原始 `full_dataset` 取 `collate_fn` |
| 3 | `EpisodicDataset` 默认包含 RGB/depth 字段 | 模型中 `use_vision=False` 时这些字段被忽略但仍在加载 | 可接受的开销（HDF5 不加载未访问的数据集） |
| 4 | Resume 时 `--epochs 1` 不训练 | 从 epoch 3 恢复后 `range(3, 1)` 为空 | 正确行为——需要指定比当前 epoch 更大的 `--epochs` |

---

*Day 7 完成 | 2026-07-13*
*Next: Day 8 — 模型训练（加载 Day 5 专家数据，完整 500 epoch 训练，调参）*
