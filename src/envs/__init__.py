"""VLARM Environment Package.

Provides Isaac Sim task scene and Gym-style environment wrapper.

Modules:
    task_scene.py  — Build desktop manipulation scene (table, cubes, basket, cameras).
    isaac_env.py   — Gym-style RL environment (reset / step / close) wrapping Franka + scene.

Usage (inside Isaac Sim Python):
    from envs.task_scene import build_task_scene
    from envs.isaac_env import IsaacEnv

    scene = build_task_scene()
    env = IsaacEnv()
    obs = env.reset()
    obs, reward, done, info = env.step(action)
    env.close()
"""

from envs.isaac_env import IsaacEnv
from envs.task_scene import (
    build_table,
    build_task_scene,
    setup_cameras,
    spawn_cubes,
    spawn_target_pad,
)

__all__ = [
    "IsaacEnv",
    "build_task_scene",
    "build_table",
    "spawn_cubes",
    "spawn_target_pad",
    "setup_cameras",
]
