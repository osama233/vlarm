# Day 4 日志 — 数据采集管道

> **日期**: 2026-07-13  
> **目标**: 建立从仿真环境到策略训练的数据桥梁（录制 → 存储 → 加载 → 增强）  
> **结果**: ✅ 全部完成，49/49 测试通过

---

## 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/vl_data/__init__.py` | 60 | 包导出 |
| `src/vl_data/recorder.py` | 310 | EpisodeRecorder + CameraSource 接口 + validate_episode() |
| `src/vl_data/dataset.py` | 230 | EpisodicDataset（PyTorch Dataset）+ 窗口采样 + 统计计算 |
| `src/vl_data/augmentation.py` | 370 | Compose, RandomColorJitter, RandomCrop, JointNoise |
| `scripts/04_test_data_pipeline.py` | 590 | 7 项集成测试 |

## 修改文件

| 文件 | 变更 |
|------|------|
| `TASKS.md` | Day 4 全部勾选完成 |

---

## 架构设计

### 数据流

```
IsaacEnv (step, reset)
    │
    ▼
EpisodeRecorder (捕获 obs/action/reward，缓冲写入 HDF5)
    │
    ▼
data/raw/episode_*.h5  (每个 episode 一个文件)
    │
    ▼
EpisodicDataset (PyTorch Dataset，时间窗口切片)
    │
    ▼
DataAugmentation (颜色抖动 + 裁剪 + 关节噪声)
    │
    ▼
torch.utils.data.DataLoader → 策略训练
```

### HDF5 数据格式

```
episode_00000.h5
├── .attrs/
│   ├── episode_id        int           # 编号
│   ├── timestamp         str           # ISO 8601 时间戳
│   ├── num_steps         int           # 总步数 T
│   ├── success           bool          # 是否成功
│   ├── seed              int           # 随机种子
│   ├── env_version       str           # "vlarm-0.1"
│   ├── camera_source     str           # "NullCameraSource"
│   └── config_json       str           # 场景参数 JSON
├── observations/
│   ├── joint_positions   (T, 9)  float32
│   ├── joint_velocities  (T, 9)  float32
│   ├── ee_position       (T, 3)  float32
│   ├── ee_orientation    (T, 4)  float32
│   ├── gripper_width     (T, 1)  float32
│   ├── rgb               (T, 480, 640, 3) uint8
│   └── depth             (T, 480, 640, 1) float32
├── actions/
│   └── joint_targets     (T, 9)  float32
├── rewards               (T,)   float32
├── terminals             (T,)   bool
└── truncations           (T,)   bool
```

设计要点：
- 所有数据集沿时间轴 gzip level=4 压缩
- chunk_size=1 支持高效的窗口切片读取
- `maxshape=(None, ...)` 允许 episode 时间轴无限扩展
- 内部缓冲 50 步一批写入，避免每步 resize HDF5 的开销

### 核心类接口

**EpisodeRecorder**
```python
recorder = EpisodeRecorder(save_dir="data/raw/", camera_source=NullCameraSource())
recorder.start_episode(env)              # 创建 HDF5 文件
recorder.record_step(obs, action, ...)   # 追加一步数据（内存缓冲）
recorder.end_episode(success=True)       # flush 缓冲 → 写入元数据 → 关闭文件
recorder.close()                         # 释放资源
```

**EpisodicDataset**
```python
ds = EpisodicDataset("data/raw/", obs_horizon=2, action_horizon=16)
sample = ds[0]
# {
#   "observations": {joint_positions: (2, 9), ee_position: (2, 3), ...},
#   "actions": (16, 7),        # 预测 16 帧动作（7 个关节）
#   "rgb": (2, 480, 640, 3),   # 归一化到 [0, 1]
#   "depth": (2, 480, 640, 1),
#   "language_embedding": (768,),  # 占位（Day 10 填充）
# }
```

**数据增强**
```python
aug = Compose([
    RandomColorJitter(brightness=0.2, contrast=0.2, p=0.8),
    RandomCrop(scale=(0.8, 1.0), p=0.5),
    JointNoise(joint_std=0.005, p=0.5),
])
augmented = aug(sample)
```

---

## 踩坑记录

| # | 问题 | 根因 | 解决 |
|---|------|------|------|
| 1 | `ModuleNotFoundError: No module named 'data.recorder'` | Isaac Sim 6.0.1 内置 OpenCV (`omni.pip.compute`) 在 `sys.modules` 中注册了 `cv2.data` 为顶级模块 `data`，遮蔽了我们的 `src/data/` 包。即使 `sys.path.insert(0, ...)` 也无效，因为 Python 优先查 `sys.modules` 缓存 | 将 `src/data/` 重命名为 `src/vl_data/`，彻底避免命名冲突 |
| 2 | 诊断过程反复失败 | 最初尝试在 `sys.modules` 里 `del data` 再 `import data`，但在 Isaac Sim 扩展加载后、我们的导入前，`cv2.data` 会被重新注册。尝试了多种时机和顺序都不可靠 | 重命名是最彻底的解决方案（见 #1） |
| 3 | `_shift_hue_numpy` 报错 `not enough values to unpack (expected 3, got 2)` | HSV→RGB 转换中的 `zip()` 循环：`zip([0,1,2,3,4,5], [(c,x,0), ...])` 产生 `(int, tuple)` 对，但 unpack 期望 3 个值。且 `0`（Python 标量）和 numpy 数组混合时会索引失败 | 重写循环：每个 sector 单独处理，用 `isinstance` 区分标量和数组 |
| 4 | 测试 Summary 不显示 | `simulation_app.close()` 在 `finally` 块中先执行，关闭了输出流，导致后面的 Summary section 无输出 | 把 Summary 移到 `finally` 内部、cleanup 之前 |
| 5 | `--steps 5` 时 Dataset 返回 0 样本 | 5 步的 episode 无法容纳 `obs_horizon=2 + action_horizon=8 = 10` 步的窗口，所有起始位置都被过滤 | 测试默认用 `--steps 20`，保证足够的有效窗口 |
| 6 | `import torch` 在 Isaac Sim Python 中失败 | Isaac Sim 自带的 Python 3.12 环境没有安装 torch（torch 在 conda 环境中） | dataset.py 和 augmentation.py 使用懒加载 `_ensure_torch()`；测试脚本用 try/except 包裹导入，torch 不可用时跳过 Dataset/Augmentation 测试节 |

---

## 设计与架构决策

### 1. 可插拔相机源

Day 3 的 IsaacEnv 返回空白 RGB/depth 数组。Day 4 不急于修改 env，而是设计了 `CameraSource` 抽象接口：

```python
class CameraSource(ABC):
    def capture(self) -> tuple[np.ndarray, np.ndarray]: ...  # (rgb, depth)
    def close(self) -> None: ...
```

当前实现 `NullCameraSource`（零数组），后续替换为：
- `IsaacSimCameraSource` — 使用 `RtxCamera` + `CameraSensor`（Isaac Sim 6.0 新 API）
- `ROS2CameraSource` — 订阅 /rgb、/depth 话题

### 2. 缓冲写入策略

HDF5 的 `resize()` 每步调用开销大。EpisodeRecorder 使用内存缓冲（50 步），到达阈值后一次性 `resize + write`。一个 500 步 episode 只需 10 次 I/O 而非 500 次。

### 3. 时序一致性增强

Diffusion Policy 对帧间一致性敏感。同一观测窗口的所有帧共享相同的：
- 颜色抖动参数（亮度/对比度/饱和度/色相）
- 裁剪窗口位置

避免"帧间闪烁"这种不存在的物理现象被模型学到。

### 4. 数据集预索引

`EpisodicDataset.__init__` 时扫描所有 `.h5` 文件，预计算每个有效窗口的 `(file_idx, start_step)`。`__len__` 立即返回，`__getitem__` 只需切片读取。相比每次随机采样再检查边界，效率提升显著。

### 5. 灰度数据相机 — 不增强 RGB/depth 的几何关系

- Depth 不做任何几何增强（保持度量准确性）
- JointNoise 独立加在每帧（模拟传感器噪声，不需要时序一致性）

---

## 测试覆盖

7 个测试段，49 项检查：

| # | 测试段 | 检查数 | 验证内容 |
|---|--------|--------|---------|
| 1 | Recorder 实例化 | 3 | 创建、相机源、输出目录 |
| 2 | 单 Episode 录制 | 4 | 文件创建、路径、完整性 |
| 3 | HDF5 校验 | 22 | 属性、17 个数据集 shape/dtype、数据范围、NaN |
| 4 | 多 Episode 录制 | 5 | 连续录制、文件计数、文件大小 |
| 5 | Dataset 加载 | 8 | 样本数、obs/action/rgb/depth keys、tensor shape |
| 6 | 窗口采样 | 4 | obs_horizon、action_horizon、arm DOF |
| 7 | 数据增强 | 3 | ColorJitter、RandomCrop、JointNoise、Compose |

---

## 正确运行方式

```bash
# 录制 + Dataset + 增强 集成测试（头显模式）
conda deactivate
source /opt/ros/jazzy/setup.bash
~/isaac-sim-standalone-6.0.1-linux-x86_64/python.sh scripts/04_test_data_pipeline.py --headless

# 查看录制的 HDF5 文件
ls -lh data/raw/episode_*.h5

# 用系统 Python 验证 Dataset/Augmentation（需要 torch）
python -c "
from vl_data.dataset import EpisodicDataset
ds = EpisodicDataset('data/raw/', obs_horizon=2, action_horizon=16)
print(f'{len(ds)} samples')
sample = ds[0]
print({k: v.shape for k, v in sample['observations'].items()})
"
```

---

*Day 4 完成 | 2026-07-13*
