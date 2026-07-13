# Day 6 日志 — Diffusion Policy 核心模块

> **日期**: 2026-07-13  
> **目标**: 实现 Diffusion Policy 的三大核心模块（Noise Scheduler、Vision Encoder、1D Conv U-Net）  
> **结果**: ✅ 完成，全部 5 项测试通过

---

## 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/models/noise_scheduler.py` | 290 | DDPM/DDIM 噪声调度器 |
| `src/models/vision_encoder.py` | 310 | ResNet-18 视觉编码器 + FiLM |
| `src/models/diffusion_policy.py` | 420 | 1D Conv U-Net 去噪网络（完整 Diffusion Policy） |
| `scripts/06_test_models.py` | 400 | 5 项单元测试 + 端到端验证 |

## 修改文件

| 文件 | 变更 |
|------|------|
| `src/models/__init__.py` | 模型包导出 |
| `src/vl_data/dataset.py` | `compute_statistics()` 增加 `skip_keys` 参数（跳过 RGB 避免 OOM） |
| `TASKS.md` | Day 6 全部勾选 |

---

## 架构设计

### 整体数据流

```
训练:
  actions (B, 16, 7)  ──→ add_noise(a_0, ε, t)  ──→ a_t (B, 16, 7)
  obs (dict)          ──→ VisionEncoder + StateEncoder ──→ cond (B, 896)
  timestep t          ──→ SinusoidalEmbedding ──→ time_embed (B, 128)
                                                       │
  a_t ──→ 1D Conv U-Net ←── FiLM(cond + time_embed) ──┘
                │
          predicted_noise ε_θ (B, 16, 7)

  loss = MSE(ε_θ, ε)
```

```
推理 (DDIM 16 步):
  x_T ~ N(0, I)  ──→ for t = T-1, ..., 0:  x = ddim_step(model(x, t, obs))
                                                       ↓
                                                 clean action a_0
```

### 1. Noise Scheduler (`DDPMScheduler`)

- **3 种 β 调度**: linear, cosine, squared_cosine（后者收敛最快）
- **DDPM 反向步**: 随机采样，适合训练早期验证
- **DDIM 加速采样**: 100 步 → 16 步（6.25× 加速），确定性推理
- **关键属性**: ᾱ_T ≈ 0（cosine/squared_cosine），保证充分加噪

### 2. Vision Encoder (`ResNet18Encoder`)

- ResNet-18 backbone（可选 ImageNet 预训练）
- 支持冻结 backbone + 可训练 FiLM
- FiLM（Feature-wise Linear Modulation）用于时间步注入
- 输入 `(B, H, W, 3)` → 输出 `(B, 512)`
- 对于零值 RGB（当前占位相机），输出接近零向量，状态分支承担全部条件

### 3. Diffusion Policy (`DiffusionPolicy`)

- **State Encoder**: 3 层 MLP（13D → 256D × 2 → 256D）
- **Observation Projection**: vision(512) + state(256) → 256D
- **Timestep Embedding**: Sinusoidal（128D，类似 Transformer 位置编码）
- **1D Conv U-Net**（核心去噪网络）:
  - 3 层下采样（MaxPool1d×2）+ 瓶颈 + 3 层上采样
  - 每层 2 个 FiLM 条件化 Conv1d + GroupNorm
  - 跳跃连接（U-Net 标准设计）
  - 通道: 7→64→128→256→256→128→64→7
  - 输出层用 σ=1e-6 近零初始化（防止训练早期梯度消失）
- **参数量**: 14.67M（~56 MB FP32）

### 条件注入机制

FiLM（Feature-wise Linear Modulation）是主要的条件注入方式：
```
feature = (1 + γ(cond)) * feature + β(cond)
```
- γ 和 β 由 cond（obs_embed + time_embed）通过线性层预测
- γ 和 β 初始化为零 → FiLM 初始等价于恒等映射
- 每个 U-Net block 都有独立的 FiLM 层

---

## 测试结果

| 测试 | 内容 | 结果 |
|------|------|------|
| Test 1 | Noise Scheduler：前向加噪 SNR 递减、DDIM 采样步调度 | ✅ |
| Test 2 | Vision Encoder：基础前向、FiLM、自定义维度、冻结 | ✅ |
| Test 3 | Diffusion Policy：形状、无视觉模式、梯度流、DDIM 推理 | ✅ |
| Test 4 | 真实数据训练：67 episodes → 3268 samples，5 步训练 | ✅ |
| Test 5 | 数据集统计：5 个观测字段的 min/mean/max/std | ✅ |

### 训练验证（Test 4 细节）

```
数据集: 67 episodes, 3268 training samples
初始 loss: 0.98（接近 1.0，即预测零噪声时的 MSE ≈ 1.0）
5 步训练耗时: 0.8s（~154 ms/step，CPU 模式）
```

### 数据统计（Test 5 细节）

| 字段 | Min | Max | Mean | Std |
|------|-----|-----|------|-----|
| joint_positions | -10.49 | 25.44 | 0.43 | 1.81 |
| joint_velocities | -228.1 | 472.6 | 0.06 | 6.25 |
| ee_position | -0.32 | 0.84 | 0.32 | 0.30 |
| ee_orientation | -0.99 | 1.00 | -0.12 | 0.48 |
| gripper_width | 0.0 | 0.42 | 0.05 | 0.04 |

> 注意：极值（±10 rad 关节、±200 rad/s 速度）来自验证失败的 truncated episode。Day 7 训练前需过滤这些异常数据。

---

## 踩坑记录

| # | 问题 | 根因 | 解决 |
|---|------|------|------|
| 1 | `compute_statistics()` OOM 30 GiB | RGB 图像 (480×640×3) × 4407 帧 = 30 GB | 增加 `skip_keys=("rgb", "depth")` 参数跳过图像字段 |
| 2 | 仅 2 个参数有梯度 | 输出层零初始化阻塞梯度回传 | 改用 σ=1e-6 近零正态初始化，初始输出 ≈ 0 但梯度可流通 |
| 3 | 线性 schedule ᾱ_T=0.36 | β_end=0.02 太小，100 步不够 | 默认使用 cosine schedule（ᾱ_T≈0），线性仅作参考 |

---

## 运行方式

```bash
# 单元测试（需要 conda vlarm 环境，不需要 Isaac Sim）
conda activate vlarm
PYTHONPATH=src python scripts/06_test_models.py

# 单独测试各模块
PYTHONPATH=src python src/models/noise_scheduler.py
PYTHONPATH=src python src/models/vision_encoder.py
PYTHONPATH=src python src/models/diffusion_policy.py
```

---

## 已知限制

1. **无真实相机**: RGB 全部为零，vision 分支输出接近零向量。训练时 `use_vision=False` 避免浪费计算。
2. **数据质量**: ~19% episode 存在关节异常值（Day 5 遗留），Day 7 训练前需过滤。
3. **DDIM 随机权重推理**: 未训练的模型 DDIM 推理输出范围巨大（±4000），正常现象。

---

*Day 6 完成 | 2026-07-13*
*Next: Day 7 — 训练管道（配置系统 + PyTorch 训练循环 + TensorBoard）*
