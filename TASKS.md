# VLARM — 任务清单

> 基于 [PROJECT_PLAN.md](PROJECT_PLAN.md) 的 14 天里程碑拆分，每项可勾选追踪进度。

---

## Week 1: 仿真环境 + 数据管道

### Day 1 — 环境搭建

- [ ] 确认 NVIDIA 驱动已安装 (`nvidia-smi`)
- [ ] 安装 CUDA Toolkit 12.x
- [ ] 安装 Miniconda / Miniforge
- [ ] 创建 conda 环境（Python 3.10 或 3.11）
- [ ] 安装 PyTorch（CUDA 12.x 版本）
- [ ] 安装 ROS2 Jazzy（`apt install ros-jazzy-desktop`）
- [ ] 安装 Isaac Sim 4.0+（Omniverse Launcher 或 .run 包）
- [ ] 安装 Isaac Sim 系统依赖（libvulkan1、libglu1-mesa 等）
- [ ] 创建项目目录结构（src/、scripts/、configs/、data/ 等）
- [ ] 创建 `requirements.txt` 并安装基础 Python 依赖
- [ ] 编写 `01_setup_env.py` 环境验证脚本并跑通
- [ ] 初始化 Git 仓库

### Day 2 — Isaac Sim + ROS2 桥接

- [ ] 启动 Isaac Sim，验证 GUI 正常渲染
- [ ] 加载 Franka Panda 机器人模型
- [ ] 配置 Isaac Sim ROS2 Bridge 扩展
- [ ] 编写 `ros2_bridge/action_pub.py` — 发布关节轨迹
- [ ] 编写 `ros2_bridge/state_sub.py` — 订阅机器人状态
- [ ] 编写 `ros2_bridge/camera_sub.py` — 订阅相机图像
- [ ] 编写 `02_test_ros2_bridge.py` 桥接测试脚本
- [ ] 跑通仿真 → ROS2 → 控制机器人移动的完整链路

### Day 3 — 任务场景搭建

- [ ] 在 Isaac Sim 中搭建桌面操作场景
- [ ] 放置目标物体（红色方块、蓝色方块、绿色方块）
- [ ] 放置目标容器（篮子）
- [ ] 配置 Franka Panda 的初始位置和姿态
- [ ] 添加 RGB 相机并配置视角
- [ ] 添加 Depth 相机
- [ ] 编写 `envs/task_scene.py` 场景加载代码
- [ ] 编写 `envs/isaac_env.py` 环境封装（reset / step / close）
- [ ] 验证场景可通过 Python 脚本加载和重置

### Day 4 — 数据采集管道

- [ ] 编写 `data/recorder.py` — 录制 RGB + Depth + Joint State
- [ ] 定义数据格式：HDF5 存储 + JSON 元数据
- [ ] 实现按 episode 录制（reset → actions → done）
- [ ] 录制一个简单 episode 验证数据完整性
- [ ] 编写 `data/dataset.py` — PyTorch Dataset 读取 HDF5
- [ ] 实现数据验证函数（检查 shape、范围、时间戳）
- [ ] 编写 `data/augmentation.py` — 数据增强（颜色抖动、随机裁剪等）
- [ ] 测试 DataLoader 加载和增强流程

### Day 5 — 专家策略采集

- [ ] 实现硬编码脚本策略（抓取 → 移动 → 放置）
- [ ] 或配置 WASD 键盘遥操作
- [ ] 采集 50-100 条专家演示 episode
- [ ] 每条 episode 包含：RGB 序列 + Depth + Joint States + End-Effector Pose
- [ ] 编写 `03_collect_data.py` 统一采集脚本
- [ ] 验证采集数据质量（检查轨迹平滑度、视觉清晰度）
- [ ] 将原始数据存入 `data/raw/`

### Day 6 — Diffusion Policy 理论 & 核心模块

- [ ] 阅读 Diffusion Policy 论文（C. Chi et al., 2023）
- [ ] 理解 DDPM / DDIM 原理（前向加噪 + 反向去噪）
- [ ] 编写 `models/noise_scheduler.py` — 噪声调度器（linear / cosine）
- [ ] 编写 `models/diffusion_policy.py` — U-Net 去噪网络
- [ ] 编写 `models/vision_encoder.py` — ResNet-18 特征提取器
- [ ] 实现条件注入机制（视觉特征 + 机器人状态 → U-Net）
- [ ] 单元测试：验证 noise scheduler 的 forward/reverse
- [ ] 单元测试：验证 diffusion policy 的输入输出 shape

### Day 7 — 训练管道

- [ ] 编写 `configs/train_config.yaml` — 训练超参数
- [ ] 编写 `configs/robot_config.yaml` — 机器人配置
- [ ] 编写 `configs/task_config.yaml` — 任务配置
- [ ] 编写 `utils/config.py` — 配置加载工具
- [ ] 编写 `04_train.py` — 完整训练脚本
- [ ] 集成 PyTorch Lightning（或纯 PyTorch 训练循环）
- [ ] 集成 TensorBoard 日志（loss、learning rate、gradient norm）
- [ ] 用随机数据跑通一个 epoch 验证 pipeline
- [ ] 设置 checkpoint 保存策略

---

## Week 2: 训练 + 集成 + 展示

### Day 8 — 模型训练

- [ ] 加载 Day 5 采集的专家数据
- [ ] 划分训练集 / 验证集（80/20）
- [ ] 设置训练超参数（batch_size=32, lr=1e-4, epochs=500）
- [ ] 启动完整训练
- [ ] 监控 loss 曲线和梯度状态
- [ ] 观察过拟合/欠拟合，调参
- [ ] 如果显存不足：减小 batch_size 到 16 或开启 gradient checkpointing
- [ ] 保存最佳模型 checkpoint

### Day 9 — 推理与评估

- [ ] 编写 `05_eval.py` — 推理与评估脚本
- [ ] 加载训练好的 checkpoint
- [ ] 在 Isaac Sim 中运行推理（observation → action prediction）
- [ ] 实现 rollout 循环（每步执行动作，获取新观测）
- [ ] 计算成功率（成功抓取并放置 / 总尝试次数）
- [ ] 记录失败案例（碰撞、掉物体、未到达目标）
- [ ] 分析失败原因并记录优化方向

### Day 10 — 语言集成

- [ ] 选择语言编码器（CLIP ViT-B/32 或 all-MiniLM-L6）
- [ ] 编写 `models/language_encoder.py` — 文本 → embedding
- [ ] 将语言 embedding 注入 Diffusion Policy 的条件输入
- [ ] 定义语言指令集合（"pick up the red cube" 等）
- [ ] 修改 `data/dataset.py` 支持语言标注
- [ ] 修改训练脚本支持语言条件
- [ ] 训练语言条件版本（从头或 fine-tune）
- [ ] 评估：同一场景不同指令，验证策略是否理解语言

### Day 11 — Demo 录制

- [ ] 设计 3-5 个 Demo 场景（不同物体、不同指令）
- [ ] 编写 `06_demo.py` — 一键 Demo 脚本
- [ ] 录制屏幕 + Isaac Sim 画面
- [ ] 确保 Demo 展示完整流程：指令输入 → 策略推理 → 机器人执行
- [ ] 剪辑/整理视频，添加字幕说明
- [ ] 确保 Demo 可在 GitHub 上直接展示（GIF/MP4）

### Day 12 — 文档 + GitHub

- [ ] 编写 `README.md`（项目介绍、架构图、安装步骤、复现指南）
- [ ] 绘制系统架构图（使用 draw.io 或 Mermaid）
- [ ] 编写 `docs/INSTALL.md` — Ubuntu 24.04 详细安装指南
- [ ] 编写 `docs/USAGE.md` — 使用说明
- [ ] 补充代码 docstring
- [ ] 完善 `requirements.txt` 版本号
- [ ] 创建 GitHub 仓库并推送代码
- [ ] 添加 GitHub Actions CI（可选，自动测试）

### Day 13 — 代码审查 + 测试

- [ ] 编写 `tests/test_bridge.py` — ROS2 桥接测试
- [ ] 编写 `tests/test_dataset.py` — 数据加载测试
- [ ] 编写 `tests/test_policy.py` — 策略推理测试
- [ ] 运行所有测试，确保通过
- [ ] 代码格式化（black / ruff）
- [ ] 清理调试 print / 注释掉的代码
- [ ] 检查硬编码路径，改为配置文件
- [ ] 统一命名规范和代码风格

### Day 14 — 最终发布

- [ ] 打 Git tag `v1.0.0`
- [ ] 编写 Release Notes
- [ ] 上传 Demo 视频到 GitHub Release 或 YouTube
- [ ] 更新简历项目经历描述
- [ ] 准备面试回答（架构设计、技术选型、遇到的挑战）
- [ ] 在 GitHub 项目页添加 Topics 标签（vla、robotics、isaac-sim、ros2、diffusion-policy）

---

## 完成后

- [ ] 可选项：尝试 Sim-to-Real 迁移到真实机械臂
- [ ] 可选项：替换为 ACT (Action Chunking Transformer) 策略做对比实验
- [ ] 可选项：添加多视角视觉（增加 wrist camera）
- [ ] 可选项：部署到 Docker 容器方便他人复现

---

*Created: 2026-07-12 | 共 14 天 · ~90 项子任务*
