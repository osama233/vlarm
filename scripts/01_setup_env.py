#!/usr/bin/env python3
"""VLARM Environment Setup Verification Script
Usage: python 01_setup_env.py
"""

import sys
from pathlib import Path


def check_header(title: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "✅" if ok else "❌"
    msg = f"  {status}  {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return ok


def main():
    errors = []

    # --- Python ---
    check_header("Python")
    v = sys.version_info
    ok = check("Python Version", v.major == 3 and v.minor in (10, 11, 12),
               f"{v.major}.{v.minor}.{v.micro}")
    if not ok:
        errors.append("Python 3.10/3.11/3.12 required")

    # --- CUDA ---
    check_header("NVIDIA / CUDA")
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        ok = check("nvidia-smi", r.returncode == 0, "GPU driver loaded")
        if not ok:
            errors.append("NVIDIA driver not working - check nvidia-smi")
    except Exception:
        check("nvidia-smi", False, "not found")
        errors.append("nvidia-smi not found")

    # --- PyTorch ---
    check_header("PyTorch")
    try:
        import torch
        ver = torch.__version__
        cuda_ok = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if cuda_ok else "N/A"
        check("torch", True, f"v{ver}")
        check("CUDA Available", cuda_ok, f"CUDA {cuda_ver}")
        check("GPU Device", cuda_ok, torch.cuda.get_device_name(0) if cuda_ok else "N/A")
        if not cuda_ok:
            errors.append("CUDA not available in PyTorch")
    except ImportError:
        check("torch", False, "not installed")
        errors.append("PyTorch not installed")

    # --- ROS2 ---
    check_header("ROS2")
    ros2_path = "/opt/ros/jazzy/setup.bash"
    if Path(ros2_path).exists():
        check("ROS2 Jazzy", True, "installed")
    else:
        check("ROS2 Jazzy", False, f"{ros2_path} not found")
        errors.append("ROS2 Jazzy not installed")

    # --- Isaac Sim Dependencies ---
    check_header("Isaac Sim System Dependencies")
    libs = ["libvulkan1", "libglu1-mesa", "libxcb-cursor0", "libnss3"]
    import subprocess as sp
    for lib in libs:
        r = sp.run(["dpkg", "-l", lib], capture_output=True, text=True)
        check(lib, r.returncode == 0)

    # --- Python Packages ---
    check_header("Python Packages")
    pkgs = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("h5py", "h5py"),
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("yaml", "pyyaml"),
        ("cv2", "opencv-python"),
    ]
    for mod, pkg in pkgs:
        try:
            __import__(mod)
            check(pkg, True)
        except ImportError:
            check(pkg, False, "not installed")

    # --- Project Structure ---
    check_header("Project Structure")
    project_root = Path(__file__).resolve().parent
    dirs = ["src", "scripts", "configs", "data/raw", "data/processed",
            "checkpoints", "logs", "notebooks", "docs", "tests"]
    for d in dirs:
        p = project_root / d
        check(d, p.is_dir(), str(p))

    # --- Summary ---
    check_header("Summary")
    if errors:
        print(f"  ❌ {len(errors)} issue(s) found:")
        for e in errors:
            print(f"     - {e}")
        sys.exit(1)
    else:
        print("  ✅ All checks passed! Environment ready.")
        sys.exit(0)


if __name__ == "__main__":
    main()
