# VLARM — Vision-Language-Action Robot Manipulation

> **Mentor-guided project**: 14-day Embodied AI project from scratch
> **Mentor**: Senior VLA/Robot Learning Engineer (virtual)
> **Student**: Robotics student, targeting Embodied AI/VLA internship
> **Hardware**: RTX 4050 Laptop 6GB, 16GB RAM, Ubuntu 24.04 LTS

---

## 项目定位

在 Isaac Sim 仿真环境中，实现一个**语言条件机器人操作（Language-Conditioned Robot Manipulation）**系统，使用 ROS2 进行通信，使用 Diffusion Policy 作为动作生成模型，最终实现"人说指令 → 机器人执行"的完整闭环。

### 为什么选这个方向？

| 维度 | 说明 |
|------|------|
| **面试价值** | VLA + 仿真 + ROS2 是具身智能岗位的核心技能栈 |
| **硬件友好** | 6GB 显存可完成训练（Diffusion Policy 比自回归 VLA 更高效） |
| **无需机器人** | Isaac Sim 提供完整仿真环境 |
| **工程完整** | 覆盖数据采集 → 训练 → 部署完整流程 |
| **可复现** | GitHub 公开，任何人可跑通 |
| **Demo 出彩** | 语言指令 → 机器人执行，视觉冲击力强 |

---

## 技术架构

```
User: "Pick up the red cube on the left"
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Language Encoder (CLIP / SBERT)                │
│  "pick up the red cube on the left" → embedding │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Vision Encoder (ResNet-18 / EfficientNet)      │
│  RGB Camera → feature map                       │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Diffusion Policy                                │
│  noise → denoise → action sequence              │
│  conditioned on: language + vision + robot state │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  ROS2 Action Publisher                          │
│  /joint_trajectory or /end_effector_pose        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Isaac Sim                                      │
│  Franka Panda / UR5 → execute trajectory        │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  ROS2 Feedback (joint states, camera, force)    │
│  → back to Policy for next step                 │
└─────────────────────────────────────────────────┘
```

---

## ⚠️ Ubuntu 24.04 适配说明

本项目原计划在 Windows 11 上开发，现已迁移至 **Ubuntu 24.04 LTS**。关键适配点：

### ROS2 版本选择

| 方案 | 说明 | 推荐 |
|------|------|------|
| **ROS2 Jazzy** | 原生支持 Ubuntu 24.04，`apt install ros-jazzy-desktop` | ⭐ 推荐 |
| **ROS2 Humble via Docker** | 使用 `osrf/ros:humble-desktop` 镜像，与 Isaac Sim 桥接需额外配置网络 | 备选 |
| **Robostack (conda)** | 通过 conda-forge 安装 ROS2 Humble，跨平台但可能有依赖冲突 | 备选 |

> 推荐使用 ROS2 **Jazzy**：API 与 Humble 几乎一致，原生 apt 安装，免去 Docker 网络配置的麻烦。

### Isaac Sim

- **2023.1 可能不兼容** Ubuntu 24.04（依赖 Python 3.8/3.10 旧版库）
- 建议使用 **Isaac Sim 4.0+**，官方已适配 Ubuntu 24.04
- 安装方式：Omniverse Launcher（Linux 版）或直接下载 `.run` 包
- 额外依赖：`sudo apt install libvulkan1 libglu1-mesa libxcb-cursor0 libnss3`

### NVIDIA 驱动 & CUDA

```bash
# 安装推荐驱动
sudo ubuntu-drivers autoinstall
# 或指定版本
sudo apt install nvidia-driver-550
# CUDA Toolkit（推荐 12.x）
sudo apt install nvidia-cuda-toolkit
```

### Python 版本注意

Ubuntu 24.04 默认 Python 3.12，部分库（如 PyTorch 2.0+、ROS2 Jazzy）已适配，但某些旧库可能不兼容，建议在 conda 环境中指定 Python 3.10 或 3.11。

---

## 14 天里程碑计划

### Week 1: 仿真环境 + 数据管道 (Day 1-7)

| Day | Milestone | 核心产出 |
|-----|-----------|---------|
| **Day 1** | 环境搭建 | NVIDIA 驱动 + conda + PyTorch + ROS2 Jazzy + Isaac Sim 4.0 + 项目结构 ✅ |
| **Day 2** | Isaac Sim + ROS2 桥接 | 打通仿真→ROS2 通信，控制机器人移动 |
| **Day 3** | 任务场景搭建 | 在 Isaac Sim 中创建桌面操作场景（方块+篮子） |
| **Day 4** | 数据采集管道 | 录制 RGB + Depth + Joint State → HDF5 |
| **Day 5** | 专家策略采集 | 编写脚本策略/WASD 遥操作采集 50-100 条演示 |
| **Day 6** | Diffusion Policy 理论 | 理解 DDPM/DDIM，阅读论文，编写核心模块 |
| **Day 7** | 训练管道 | DataLoader + 训练循环 + TensorBoard 可视化 |

### Week 2: 训练 + 集成 + 展示 (Day 8-14)

| Day | Milestone | 核心产出 |
|-----|-----------|---------|
| **Day 8** | 模型训练 | 完整训练 Diffusion Policy，调参，观察 loss 曲线 |
| **Day 9** | 推理与评估 | 在仿真中运行训练好的策略，计算成功率 |
| **Day 10** | 语言集成 | 加入 CLIP/SBERT 文本编码器，实现语言条件控制 |
| **Day 11** | Demo 录制 | 拍摄完整 Demo（语言指令 → 机器人执行） |
| **Day 12** | 文档 + GitHub | README、架构图、安装指南、复现步骤 |
| **Day 13** | 代码审查 + 测试 | 单元测试、代码规范、清理调试代码 |
| **Day 14** | 最终发布 | Release v1.0、简历更新、面试准备 |

---

## 技术选型

| 组件 | 选择 | 原因 |
|------|------|------|
| 仿真器 | Isaac Sim 4.0+ | 工业级，ROS2 原生支持，GPU 加速。4.0+ 适配 Ubuntu 24.04 |
| 中间件 | ROS2 Jazzy (或 Humble via Docker) | Jazzy 原生支持 Ubuntu 24.04；Humble 需 Docker/Robostack |
| 机器人 | Franka Panda | 7-DoF，研究社区标准 |
| 策略 | Diffusion Policy (CNN-based) | 6GB 显存可训练，表现 SOTA |
| 视觉编码 | ResNet-18 (pretrained) | 轻量，有 ImageNet 预训练 |
| 语言编码 | CLIP ViT-B/32 或 all-MiniLM-L6 | 冻结权重，只做前向 |
| 数据格式 | HDF5 + JSON 元数据 | 高效，可扩展 |
| 训练框架 | PyTorch Lightning | 减少样板代码，自动日志 |

---

## 项目仓库结构

```
/home/aiyuan/vlarm/
├── README.md                   # 项目说明
├── .gitignore
├── requirements.txt            # Python 依赖
├── setup.py                    # 安装脚本
│
├── configs/                    # 配置文件
│   ├── task_config.yaml        # 任务配置
│   ├── train_config.yaml       # 训练超参数
│   └── robot_config.yaml       # 机器人配置
│
├── src/                        # 核心源码
│   ├── __init__.py
│   ├── envs/                   # Isaac Sim 环境
│   │   ├── __init__.py
│   │   ├── isaac_env.py        # 环境封装
│   │   └── task_scene.py       # 场景构建
│   ├── data/                   # 数据处理
│   │   ├── __init__.py
│   │   ├── recorder.py         # 数据录制
│   │   ├── dataset.py          # PyTorch Dataset
│   │   └── augmentation.py     # 数据增强
│   ├── models/                 # 模型
│   │   ├── __init__.py
│   │   ├── diffusion_policy.py # Diffusion Policy
│   │   ├── vision_encoder.py   # 视觉编码器
│   │   ├── language_encoder.py # 语言编码器
│   │   └── noise_scheduler.py  # 噪声调度器
│   ├── ros2_bridge/            # ROS2 通信
│   │   ├── __init__.py
│   │   ├── action_pub.py       # 动作发布
│   │   ├── state_sub.py        # 状态订阅
│   │   └── camera_sub.py       # 相机订阅
│   └── utils/                  # 工具函数
│       ├── __init__.py
│       ├── config.py           # 配置加载
│       └── viz.py              # 可视化
│
├── scripts/                    # 脚本
│   ├── 01_setup_env.py         # 环境验证
│   ├── 02_test_ros2_bridge.py  # ROS2 桥接测试
│   ├── 03_collect_data.py      # 数据采集
│   ├── 04_train.py             # 训练脚本
│   ├── 05_eval.py              # 评估脚本
│   └── 06_demo.py              # Demo 脚本
│
├── data/                       # 数据目录
│   ├── raw/                    # 原始采集数据
│   └── processed/              # 预处理后数据
│
├── checkpoints/                # 模型权重
├── logs/                       # 训练日志
├── notebooks/                  # Jupyter 分析
├── docs/                       # 文档
│   └── PROJECT_PLAN.md         # 本文件
│
└── tests/                      # 测试
    └── test_bridge.py
```

---

## 显存预算（6GB RTX 4050）

| 组件 | 估计显存 | 备注 |
|------|---------|------|
| Isaac Sim | ~1.5 GB | 仿真渲染 |
| Vision Encoder (ResNet-18) | ~0.1 GB | 冻结权重 |
| Language Encoder (CLIP) | ~0.5 GB | 冻结权重 |
| Diffusion Policy | ~1.5 GB | 训练时 |
| Batch + Activation | ~1.0 GB | batch_size=32 时 |
| **总计** | **~4.6 GB** | ✅ 6GB 可运行 |

> 如果显存不足：减小 batch_size 到 16 或使用梯度累积，Vision Encoder 换 MobileNetV3。

---

## 面试卖点

完成这个项目后，你可以在简历和面试中展示：

1. **完整 VLA 系统设计**：从 sensor → perception → policy → action 的闭环
2. **ROS2 实战经验**：topic/service/action 通信，与 Isaac Sim 桥接
3. **Diffusion Policy 理解**：DDPM 原理，为什么 diffusion 比 regression 好
4. **仿真数据采集管道**：如何高效采集/处理/增强机器人数据
5. **Sim-to-Real 意识**：虽然本项目在仿真中，但架构考虑了迁移到真实机器人
6. **工程规范**：Git、模块化、配置文件、测试、文档

---

## 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| Isaac Sim 启动/兼容问题 | 中 | 使用 Isaac Sim 4.0+，确保安装 libvulkan1 等依赖 |
| ROS2 版本选择 (Jazzy vs Humble) | 低 | 首选 Jazzy（原生 24.04），API 与 Humble 一致 |
| Isaac Sim + ROS2 Jazzy 桥接 | 中 | Isaac Sim 4.0+ 内置 ROS2 桥接支持 Jazzy，降级排查 |
| 6GB 显存不足训练 | 低 | 减小模型/batch，使用 gradient checkpointing |
| 仿真数据与现实差距大 | N/A | 本项目在仿真中闭环，不涉及 real transfer |
| Python 3.12 兼容性 | 中 | 在 conda 环境中使用 Python 3.10/3.11 避坑 |
| 时间不够 | 中 | 核心 12 天 + 2 天 buffer |

---

*Plan created: 2026-07-10 | Mentor-guided project*
