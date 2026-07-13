#!/usr/bin/env python3
"""Quick demo — run one pick-and-place episode with GUI to watch the robot."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Parse args BEFORE importing SimulationApp
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
args, unknown = parser.parse_known_args()

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "renderer": "RayTracedLighting",
    "headless": args.headless,
})

import carb
import isaacsim.core.experimental.utils.app as app_utils

app_utils.enable_extension("isaacsim.ros2.bridge")
app_utils.enable_extension("isaacsim.robot.experimental.manipulators.examples")
simulation_app.update()

import numpy as np

from envs.expert_policy import PickPlaceExpert
from envs.isaac_env import IsaacEnv

# Phase names for logging
PHASE_NAMES = {
    0: "APPROACH",
    1: "GRASP",
    2: "LIFT",
    3: "TRANSPORT",
    4: "PLACE",
    5: "RETRACT",
    None: "DONE",
}

print("=" * 55)
print("  VLARM — Live Pick-and-Place Demo")
print("=" * 55)

env = IsaacEnv(
    headless=args.headless,
    enable_ros2=False,
    simulation_app=simulation_app,
)
simulation_app.update()

obs = env.reset(seed=args.seed)
simulation_app.update()

# Show cube and target info
cube_pos = env._cube_positions_np
target = np.array(env._scene["config"]["target_position"], dtype=np.float32)
print(f"\n  目标方块: {cube_pos[0]}")
print(f"  目标垫:   {target}")
print(f"  方块→目标垫 距离: {np.linalg.norm(cube_pos[0] - target):.2f} m")
print()

# Create expert
franka = env.franka
expert = PickPlaceExpert()
expert.set_scene_info(cube_pos, target)

last_phase = None
ep_steps = 0

print(f"{'Step':<6} {'Phase':<12} {'EE (x,y,z)':<28} {'Action'}")
print("-" * 70)

for step in range(args.max_steps):
    expert.act(obs, franka)

    try:
        action = franka.get_dof_position_targets().numpy().flatten().astype(np.float32)
    except Exception:
        action = obs["joint_positions"].copy()

    for _ in range(env._substeps):
        env._physics_step()

    obs = env._get_obs()
    env._step_count += 1

    reward, terminated = env._compute_reward_and_done()
    ep_steps = step + 1

    # Log phase transitions
    phase = expert._phase
    phase_name = PHASE_NAMES.get(phase.value if phase else None, "DONE")
    if phase != last_phase:
        ee = obs["ee_position"]
        dist = np.linalg.norm(ee - target)
        print(f"{step:<6} → {phase_name:<10} "
              f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})  "
              f"dist_to_target={dist:.3f}m")
        last_phase = phase

    if expert.is_done:
        print(f"\n  ✅ 所有阶段完成！专家策略在 {step} 步内成功完成 pick-and-place。")
        break

    if terminated:
        ee = obs["ee_position"]
        print(f"\n  ✅ 方块已在目标垫上！步骤 {step}，EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})")
        break

    if step + 1 >= args.max_steps:
        print(f"\n  ⏰ 达到最大步数 {args.max_steps}")
        break

print(f"\n  总步数: {ep_steps}")
ee_final = obs["ee_position"]
print(f"  最终 EE: ({ee_final[0]:.3f}, {ee_final[1]:.3f}, {ee_final[2]:.3f})")
print(f"  最终奖励: {reward:.2f}")
print()

# Keep GUI alive for a few seconds so the user can see the final scene
if not args.headless:
    print("  GUI 窗口将保持 5 秒，查看最终场景...")
    for _ in range(300):
        simulation_app.update()
    print("  关闭中...")

env.close()
app_utils.stop()
simulation_app.close()
print("  完成。")
