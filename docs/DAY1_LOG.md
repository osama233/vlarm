# Day 1 日志 — 环境搭建

> **日期**: 2026-07-12  
> **目标**: 在 Ubuntu 24.04 上搭建 VLARM 完整开发环境  
> **结果**: ✅ 全部完成

---

## 最终环境

| 组件 | 版本 | 状态 |
|------|------|------|
| 操作系统 | Ubuntu 24.04.4 LTS (Noble) | ✅ |
| GPU | NVIDIA GeForce RTX 4050 Laptop 6GB | ✅ |
| NVIDIA 驱动 | 580.159.03 | ✅ |
| CUDA (驱动自带) | 13.0 | ✅ |
| CUDA Toolkit (nvcc) | 12.0.140 | ✅ |
| Miniconda | latest (py3.14 base) | ✅ |
| conda 环境 | vlarm, Python 3.11 | ✅ |
| PyTorch | 2.13.0+cu130 | ✅ |
| torchvision | 配套版本 | ✅ |
| torchaudio | 配套版本 | ✅ |
| ROS2 | Jazzy (Ubuntu 24.04 原生) | ✅ |
| Isaac Sim | 6.0.1 standalone | ✅ |
| 磁盘剩余 | ~47 GB（清理后） | ✅ |

---

## 安装过程 (共耗时约 4 小时)

### 1. NVIDIA 驱动升级 (535 → 580)

**初始状态**: 系统自带 535.309.01 驱动 (CUDA 12.2)

**问题 1**: PyTorch 2.13.0 默认装 cu130 版本，驱动 535 只支持到 CUDA 12.2，导致 `torch.cuda.is_available() = False`

**解决方案**: 升级驱动到 580.159.03

```bash
sudo apt install -y nvidia-driver-580
sudo reboot
```

**问题 2**: 重启后 `nvidia-smi` 失败，外接显示器无法识别。`lsmod | grep nvidia` 没有主模块。

**根因**: Secure Boot 开启，580 驱动的内核模块未签名被拦截。

**解决方案**: 
- 进 BIOS 关闭 Secure Boot
- 或给模块签名（较复杂）

结果: `nvidia-smi` 恢复，`torch.cuda.is_available() = True`

---

### 2. CUDA Toolkit

```bash
sudo apt install -y nvidia-cuda-toolkit
```

nvcc 版本: 12.0.140。注意驱动自带的 CUDA runtime 是 13.0，nvcc 是独立版本，两者可以不同。

---

### 3. Miniconda + 虚拟环境

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
```

需要接受 Anaconda ToS:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

创建环境:
```bash
conda create -n vlarm python=3.11 -y
```

> 选 Python 3.11 而非系统 3.12，避免部分库（如 Isaac Sim Python API）的兼容问题。

---

### 4. 配置国内镜像源

**apt 镜像** (`/etc/apt/sources.list.d/ubuntu.sources`):
```
URIs: http://mirrors.cn99.com/ubuntu/    # 后续改为清华源
```

**ROS2 镜像** (`/etc/apt/sources.list.d/ros2.list`):
```
deb https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu noble main
```

**pip 镜像** (`~/.config/pip/pip.conf`):
```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 5. PyTorch

```bash
# 本来想装 cu121（适配老驱动），但装了 cu130 后：
pip install torch torchvision torchaudio
```

升级驱动后 cu130 正常工作。**教训**: 先确定驱动版本，再选 PyTorch 版本，驱动 CUDA 版本 >= PyTorch CUDA 版本即可。

---

### 6. ROS2 Jazzy

**问题**: `sudo apt install ros-jazzy-desktop` 报 `E: 无法定位软件包`

**根因**: Ubuntu 24.04 默认源不含 ROS2，需要手动添加仓库。

**解决方案**:
```bash
# 添加 ROS2 官方密钥和源
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list

# 更换清华镜像加速
sudo sed -i 's|http://packages.ros.org/ros2/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu|' /etc/apt/sources.list.d/ros2.list

sudo apt update
sudo apt install -y ros-jazzy-desktop
```

验证:
```bash
source /opt/ros/jazzy/setup.bash
ros2 run demo_nodes_cpp talker    # 正常输出 Hello World
```

---

### 7. Isaac Sim 6.0.1

**错误尝试 1**: 从 GitHub `isaac-sim/IsaacSim/releases` 下载了源码版 (~100MB)，非预编译版。

**错误尝试 2**: 浏览器直接下载 PyTorch whl 被 CDN 拦截 (AccessDenied)。

**正确方式**: 从 NVIDIA 官网下载 standalone 版
- 文件: `isaac-sim-standalone-6.0.1-linux-x86_64.zip` (13 GB)
- 位置: `~/下载/`

```bash
unzip ~/下载/isaac-sim-standalone-6.0.1-linux-x86_64.zip -d ~/
~/isaac-sim-standalone-6.0.1-linux-x86_64/isaac-sim.sh
```

启动正常，首次编译 shader 缓存约 5-10 分钟。警告都是正常的（材质库缓存、CPU powersave、IOMMU 等）。

系统依赖（提前装好无影响）:
```bash
sudo apt install -y libvulkan1 libglu1-mesa libxcb-cursor0 libnss3
```

---

### 8. Python 依赖

```bash
~/miniconda3/envs/vlarm/bin/pip install -r requirements.txt
```

全部安装成功，包括: PyTorch Lightning, h5py, OpenCV, Transformers, OpenCLIP, TensorBoard 等。

---

### 9. Git + GitHub

```bash
git init
git add -A
git commit -m "Day 1: VLARM environment setup complete"
git remote add origin https://github.com/osama233/vlarm.git
git branch -M main
git push -u origin main
```

**问题**: GitHub 仓库自动生成的 README 与本地冲突 (rejected: fetch first)
**解决**: `git push --force` 覆盖（本地内容优先）

---

## 关键踩坑总结

| # | 问题 | 根因 | 解决 | 预防 |
|---|------|------|------|------|
| 1 | `torch.cuda.is_available()=False` | cu130 需要驱动 >=525，但 535 只到 CUDA 12.2 | 升级驱动到 580 | 先查 `nvidia-smi` 的 CUDA 版本再装 PyTorch |
| 2 | 升级驱动后 `nvidia-smi` 失效、外接屏不亮 | Secure Boot 拦截未签名模块 | 关闭 Secure Boot | 装驱动前检查 `mokutil --sb-state` |
| 3 | `apt` 找不到 `ros-jazzy-desktop` | Ubuntu 24.04 默认不含 ROS 源 | 手动添加 ROS2 仓库 + 镜像 | 先配好源再 apt install |
| 4 | ROS2/pip 下载慢 | 默认源在国外 | 换清华镜像 | 装任何大包前先配镜像 |
| 5 | Isaac Sim 下载的是源码版 | GitHub release 页有源码和二进制，容易点错 | 直接去 NVIDIA 官网下载 | 看文件大小：源码 ~100MB，二进制 6GB+ |
| 6 | `git push` 被拒绝 (rejected) | GitHub 仓库创建时自动生成了 README | `git push --force` | 创建空仓库时不勾选 "Add a README" |

## 磁盘空间管理

| 清理项 | 释放 |
|--------|------|
| Isaac Sim .zip 安装包 | 13 GB |
| pip 缓存 | 1.7 GB |
| apt 缓存 | 0.6 GB |
| **合计** | **~15 GB** |

清理后剩余 ~47 GB，足够后续数据采集和训练。

---

*Day 1 完成，用时 ~4 小时 | 2026-07-12*
