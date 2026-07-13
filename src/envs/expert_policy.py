#!/usr/bin/env python3
"""VLARM Expert Policy — Scripted pick-and-place via IK.

Uses Isaac Sim's built-in ``Franka.set_end_effector_pose()`` to generate
smooth, repeatable pick-and-place trajectories for expert demonstration
collection.

The policy is a 6-phase state machine:

    APPROACH → GRASP → LIFT → TRANSPORT → PLACE → RETRACT

Unlike a standard Gym policy, the expert **directly controls the Franka**
via ``set_end_effector_pose()`` rather than returning actions.  This is
necessary because ``IsaacEnv.step()`` always calls
``franka.set_dof_position_targets()``, which would overwrite any IK targets
computed externally.

Usage (inside Isaac Sim's Python)::

    from envs.expert_policy import PickPlaceExpert
    from envs.isaac_env import IsaacEnv

    env = IsaacEnv(simulation_app=simulation_app)
    obs = env.reset(seed=42)
    expert = PickPlaceExpert()
    franka = env.franka

    for step in range(200):
        expert.act(obs, franka)                 # sets EE + gripper targets
        action = franka.get_dof_position_targets().flatten()  # record
        # ... step physics, get obs, compute reward ...
        if terminated or truncated:
            break
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How long to hold the gripper closed/open before transitioning.
_GRASP_HOLD_STEPS = 10
_PLACE_HOLD_STEPS = 10

# IK convergence tolerances (metres)
_APPROACH_TOL = 0.03   # how close EE must be to above-cube waypoint
_LIFT_TOL = 0.03       # how close EE must be to above-cube after grasp
_TRANSPORT_TOL = 0.04  # how close EE must be to above-target waypoint
_RETRACT_TOL = 0.04    # how close EE must be to retract waypoint

# Z offsets (metres)
_APPROACH_Z_OFFSET = 0.15   # EE above cube during approach
_GRASP_Z_OFFSET = 0.025     # EE at cube (slightly above to avoid collision)
_LIFT_Z_OFFSET = 0.15       # EE above cube after grasp
_TRANSPORT_Z_OFFSET = 0.15  # EE above target during transport
_PLACE_Z_OFFSET = 0.06      # EE above target during placement
_RETRACT_Z_OFFSET = 0.20    # EE above target after release


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class _Phase(Enum):
    APPROACH = 0   # move EE above the target cube
    GRASP = 1      # lower EE, close gripper
    LIFT = 2       # raise EE while holding cube
    TRANSPORT = 3  # move EE above the target pad
    PLACE = 4      # lower EE, open gripper
    RETRACT = 5    # move EE up and away


_NEXT_PHASE = {
    _Phase.APPROACH:  _Phase.GRASP,
    _Phase.GRASP:      _Phase.LIFT,
    _Phase.LIFT:       _Phase.TRANSPORT,
    _Phase.TRANSPORT:  _Phase.PLACE,
    _Phase.PLACE:      _Phase.RETRACT,
    _Phase.RETRACT:    None,  # terminal
}


# ---------------------------------------------------------------------------
# PickPlaceExpert
# ---------------------------------------------------------------------------

class PickPlaceExpert:
    """Scripted pick-and-place expert using Franka built-in IK.

    The ``act()`` method directly sets EE targets on the Franka via
    ``set_end_effector_pose()`` and controls the gripper.  The caller is
    responsible for stepping physics and reading observations afterwards.
    """

    def __init__(self) -> None:
        # Per-episode state
        self._phase: _Phase | None = _Phase.APPROACH
        self._target_cube_pos: np.ndarray | None = None
        self._target_pad_center: np.ndarray | None = None
        self._hold_counter: int = 0
        self._step_count: int = 0
        self._episode_initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, obs: dict[str, np.ndarray], franka: Any) -> None:
        """Set EE and gripper targets on the Franka for the current step.

        Parameters
        ----------
        obs : dict
            Observation dict from ``IsaacEnv``.
        franka : Franka
            The Franka robot instance (``env.franka``).
        """
        if not self._episode_initialized:
            self._init_episode(obs, franka)

        self._step_count += 1

        if self._phase is None:
            # Episode complete — hold position with open gripper
            franka.open_gripper()
            return

        # --- Set EE target (position-only IK — no orientation constraint) ---
        target_pos = self._phase_target_position(obs)

        # Read current state from obs dict (already available; avoids
        # franka.get_current_state() which can fail with "physics tensor
        # entity not valid" when the timeline was restarted).
        current_pos = obs["ee_position"].reshape(1, -1).astype(np.float32)
        current_quat = obs["ee_orientation"].reshape(1, -1).astype(np.float32)
        current_q = obs["joint_positions"].reshape(1, -1).astype(np.float32)

        # Compute IK manually with no orientation constraint.
        # franka.set_end_effector_pose() enforces downward orientation,
        # which can cause instability during long transport motions.
        try:
            jacobian = franka.get_jacobian_matrices().numpy()
            J = jacobian[:, franka.end_effector_link_index - 1, :, :7]  # (1, 6, 7)

            delta_q = franka.differential_inverse_kinematics(
                jacobian_end_effector=J,
                current_position=current_pos,
                current_orientation=current_quat,
                goal_position=target_pos.reshape(1, -1),
                goal_orientation=None,  # don't constrain orientation → stable
                method="damped-least-squares",
            )

            if np.all(np.isfinite(delta_q)):
                arm_targets = current_q[:, :7] + delta_q
                arm_targets = arm_targets.flatten().astype(np.float32)
                # Clip to joint limits (with 2% margin to prevent
                # physics overshoot at the limit edge).
                from envs.isaac_env import IsaacEnv
                lo = IsaacEnv.JOINT_LIMITS_LOW[:7]
                hi = IsaacEnv.JOINT_LIMITS_HIGH[:7]
                span = hi - lo
                safe_lo = lo + 0.02 * span
                safe_hi = hi - 0.02 * span
                arm_targets = np.clip(arm_targets, safe_lo, safe_hi)
                franka.set_dof_position_targets(
                    arm_targets.reshape(1, -1), dof_indices=list(range(7))
                )
        except AssertionError:
            # Physics tensor entity invalid (timeline may have restarted).
            # Skip IK this step — gripper still controlled below.
            pass

        # --- Gripper control ---
        if self._phase in (_Phase.GRASP, _Phase.LIFT, _Phase.TRANSPORT):
            franka.close_gripper()
        else:
            franka.open_gripper()

        # --- Phase transition ---
        self._check_phase_transition(obs)

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self._phase = _Phase.APPROACH
        self._target_cube_pos = None
        self._target_pad_center = None
        self._hold_counter = 0
        self._step_count = 0
        self._episode_initialized = False

    @property
    def is_done(self) -> bool:
        """True when the episode is complete (all phases finished)."""
        return self._phase is None

    # ------------------------------------------------------------------
    # Episode initialisation
    # ------------------------------------------------------------------

    def _init_episode(self, obs: dict[str, np.ndarray], franka: Any) -> None:
        """Called once at the start of an episode to pick a target cube."""
        self._episode_initialized = True

        # Read cube positions — they're stored on the env, not the franka.
        # We get them from the obs context by reading the env's internal state.
        # The caller should provide these via a setter or we discover them.
        # For now, use the observation to determine the nearest cube.
        # We access env internals through a back-reference set by the caller.
        pass

    def set_scene_info(
        self,
        cube_positions: np.ndarray,
        target_center: np.ndarray,
    ) -> None:
        """Set scene information before the episode starts.

        Called by the collection script after ``env.reset()`` to provide
        cube and target positions.

        Parameters
        ----------
        cube_positions : np.ndarray
            Shape ``(N, 3)`` — world positions of all cubes.
        target_center : np.ndarray
            Shape ``(3,)`` — world position of the target pad centre.
        """
        self._target_pad_center = target_center.copy()

        # Pick the cube nearest to the target pad (farthest to travel = more data)
        # Actually, nearest to home position is more reliable.
        # Use the first cube for deterministic behaviour.
        distances = np.linalg.norm(
            cube_positions - target_center.reshape(1, 3), axis=1
        )
        # Pick cube farthest from target (needs to travel most)
        farthest_idx = int(np.argmax(distances))
        self._target_cube_pos = cube_positions[farthest_idx].copy()

    # ------------------------------------------------------------------
    # Phase target positions
    # ------------------------------------------------------------------

    def _phase_target_position(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Return the EE position target for the current phase."""
        cube = self._target_cube_pos
        target = self._target_pad_center

        if self._phase == _Phase.APPROACH:
            return cube + np.array([0.0, 0.0, _APPROACH_Z_OFFSET], dtype=np.float32)
        elif self._phase == _Phase.GRASP:
            return cube + np.array([0.0, 0.0, _GRASP_Z_OFFSET], dtype=np.float32)
        elif self._phase == _Phase.LIFT:
            return cube + np.array([0.0, 0.0, _LIFT_Z_OFFSET], dtype=np.float32)
        elif self._phase == _Phase.TRANSPORT:
            return target + np.array([0.0, 0.0, _TRANSPORT_Z_OFFSET], dtype=np.float32)
        elif self._phase == _Phase.PLACE:
            return target + np.array([0.0, 0.0, _PLACE_Z_OFFSET], dtype=np.float32)
        elif self._phase == _Phase.RETRACT:
            return target + np.array([0.0, 0.0, _RETRACT_Z_OFFSET], dtype=np.float32)
        else:
            return np.asarray(obs["ee_position"], dtype=np.float32)

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def _check_phase_transition(self, obs: dict[str, np.ndarray]) -> None:
        """Check whether the current phase is complete and advance."""
        if self._phase is None or self._target_cube_pos is None:
            return

        ee_pos = np.asarray(obs["ee_position"], dtype=np.float32)
        target_pos = self._phase_target_position(obs)
        dist = float(np.linalg.norm(target_pos - ee_pos))

        if self._phase == _Phase.APPROACH:
            if dist < _APPROACH_TOL:
                self._advance_phase()

        elif self._phase == _Phase.GRASP:
            self._hold_counter += 1
            if self._hold_counter >= _GRASP_HOLD_STEPS:
                self._hold_counter = 0
                self._advance_phase()

        elif self._phase == _Phase.LIFT:
            if dist < _LIFT_TOL:
                self._advance_phase()

        elif self._phase == _Phase.TRANSPORT:
            if dist < _TRANSPORT_TOL:
                self._advance_phase()

        elif self._phase == _Phase.PLACE:
            self._hold_counter += 1
            if self._hold_counter >= _PLACE_HOLD_STEPS:
                self._hold_counter = 0
                self._advance_phase()

        elif self._phase == _Phase.RETRACT:
            if dist < _RETRACT_TOL:
                self._advance_phase()

    def _advance_phase(self) -> None:
        """Move to the next phase."""
        self._phase = _NEXT_PHASE.get(self._phase)  # type: ignore[arg-type]
