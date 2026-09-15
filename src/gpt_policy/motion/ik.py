"""Continuous, tolerance-bounded IK on top of the official ARX solver."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry.poses import rotation_vector, rpy_to_matrix
from .coordination import TrajectoryIKError


class ContinuousIK:
    """Keep one IK branch while refining and auditing Cartesian residuals."""

    def __init__(
        self,
        solver: Any,
        joint_min: object,
        joint_max: object,
        refine_iterations: int,
        numeric_translation_tolerance_m: float,
        numeric_rotation_tolerance_rad: float,
        execution_translation_tolerance_m: float,
        execution_rotation_tolerance_rad: float,
    ) -> None:
        self.solver = solver
        self.lower = np.asarray(joint_min, dtype=np.float64)
        self.upper = np.asarray(joint_max, dtype=np.float64)
        self.refine_iterations = int(refine_iterations)
        self.numeric_translation_tolerance_m = float(
            numeric_translation_tolerance_m
        )
        self.numeric_rotation_tolerance_rad = float(numeric_rotation_tolerance_rad)
        self.execution_translation_tolerance_m = float(
            execution_translation_tolerance_m
        )
        self.execution_rotation_tolerance_rad = float(
            execution_rotation_tolerance_rad
        )

    def solve(
        self,
        target_pose: np.ndarray,
        seed: np.ndarray,
        segment: int,
        sample: int,
    ) -> np.ndarray:
        status, positions = self.solver.multi_trial_ik(target_pose, seed, 5)
        candidate = np.asarray(positions, dtype=np.float64)
        initial = (
            candidate.copy()
            if candidate.shape == seed.shape and np.isfinite(candidate).all()
            else np.asarray(seed, dtype=np.float64).copy()
        )
        solved = self._refine(target_pose, initial)
        translation_error, rotation_error = self.error(target_pose, solved)
        if (
            translation_error <= self.execution_translation_tolerance_m
            and rotation_error <= self.execution_rotation_tolerance_rad
        ):
            return solved
        raise TrajectoryIKError(
            segment=segment,
            sample=sample,
            status=status,
            status_name=self.solver.get_ik_status_name(status),
            translation_error_m=translation_error,
            rotation_error_rad=rotation_error,
        )

    def error(
        self,
        target_pose: np.ndarray,
        joint_positions: np.ndarray,
    ) -> tuple[float, float]:
        current = np.asarray(
            self.solver.forward_kinematics(joint_positions), dtype=np.float64
        )
        translation = float(np.linalg.norm(target_pose[:3] - current[:3]))
        orientation = float(
            np.linalg.norm(
                rotation_vector(
                    rpy_to_matrix(target_pose[3:]) @ rpy_to_matrix(current[3:]).T
                )
            )
        )
        return translation, orientation

    def _refine(self, target_pose: np.ndarray, initial: np.ndarray) -> np.ndarray:
        if self.refine_iterations == 0:
            return initial
        q = initial.copy()
        best = q.copy()
        best_error = float("inf")
        characteristic_length = 0.1
        epsilon = 1e-5
        target_rotation = rpy_to_matrix(target_pose[3:])
        for _ in range(self.refine_iterations):
            current = np.asarray(self.solver.forward_kinematics(q), dtype=np.float64)
            position_error = target_pose[:3] - current[:3]
            current_rotation = rpy_to_matrix(current[3:])
            orientation_error = rotation_vector(target_rotation @ current_rotation.T)
            score = float(
                np.linalg.norm(position_error)
                + characteristic_length * np.linalg.norm(orientation_error)
            )
            if score < best_error:
                best_error, best = score, q.copy()
            if (
                np.linalg.norm(position_error)
                <= self.numeric_translation_tolerance_m
                and np.linalg.norm(orientation_error)
                <= self.numeric_rotation_tolerance_rad
            ):
                return q

            jacobian = np.empty((6, q.size), dtype=np.float64)
            for joint in range(q.size):
                perturbed = q.copy()
                perturbed[joint] = min(self.upper[joint], q[joint] + epsilon)
                step = perturbed[joint] - q[joint]
                if step < epsilon / 2:
                    perturbed[joint] = max(self.lower[joint], q[joint] - epsilon)
                    step = perturbed[joint] - q[joint]
                candidate = np.asarray(
                    self.solver.forward_kinematics(perturbed), dtype=np.float64
                )
                jacobian[:3, joint] = (candidate[:3] - current[:3]) / step
                candidate_rotation = rpy_to_matrix(candidate[3:])
                jacobian[3:, joint] = rotation_vector(
                    candidate_rotation @ current_rotation.T
                ) / step

            weighted_jacobian = jacobian.copy()
            weighted_jacobian[3:] *= characteristic_length
            weighted_error = np.r_[
                position_error,
                characteristic_length * orientation_error,
            ]
            singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
            minimum = float(singular_values[-1]) if singular_values.size else 0.0
            damping = max(1e-5, 1e-3 * max(0.0, 0.02 - minimum) / 0.02)
            system = (
                weighted_jacobian @ weighted_jacobian.T
                + damping**2 * np.eye(6)
            )
            delta = weighted_jacobian.T @ np.linalg.solve(system, weighted_error)
            maximum_step = float(np.max(np.abs(delta)))
            if maximum_step > 0.05:
                delta *= 0.05 / maximum_step
            q = np.clip(q + delta, self.lower, self.upper)
        return best
