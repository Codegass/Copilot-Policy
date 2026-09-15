"""Cartesian waypoint planning independent of CAN execution and arm coordination."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry.poses import quaternion_to_rpy, rotation_distance
from .ik import ContinuousIK
from .trajectory import (
    MotionLimits,
    retime_path_segment,
    sample_pose_segment,
)


class EefTrajectoryPlanner:
    """Convert TCP targets to timed joint arrays; no hardware SDK objects."""

    def __init__(
        self,
        read_positions: Any,
        frames: Any,
        ik: ContinuousIK,
        trajectory_hz: float,
        limits: MotionLimits,
        endpoint_hold_s: float,
    ) -> None:
        self.read_positions = read_positions
        self.frames = frames
        self.ik = ik
        self.trajectory_hz = float(trajectory_hz)
        self.limits = limits
        self.endpoint_hold_s = float(endpoint_hold_s)

    def plan(
        self,
        requested: list[Any],
        note: str,
        held_gripper: float,
    ) -> dict[str, Any]:
        if not requested:
            raise ValueError("EEF trajectory must contain at least one pose")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("movement tool requires a non-empty note")

        seed = np.asarray(self.read_positions(), dtype=np.float64).copy()
        start_joint_positions = seed.copy()
        current_sdk_eef = np.asarray(
            self.ik.solver.forward_kinematics(seed), dtype=np.float64
        ).copy()
        points = self._model_points(requested, self.frames.sdk_to_tcp(current_sdk_eef))

        commands: list[Any] = []
        relative_times: list[float] = []
        tcp_samples: list[list[float]] = []
        gripper_samples: list[float] = []
        elapsed_s = 0.0
        reports: list[dict[str, Any]] = []
        for segment_index, (start_pose, end_pose) in enumerate(
            zip(points, points[1:]), start=1
        ):
            solved = self._solve_segment(
                start_pose,
                end_pose,
                seed,
                segment_index,
            )
            seed = solved["joints"][-1].copy()
            timing = solved["timing"]
            for local_index in range(1, len(solved["joints"])):
                pose = solved["poses"][local_index]
                command = solved["joints"][local_index].copy()
                commands.append(command)
                relative_times.append(
                    elapsed_s + float(timing.times_s[local_index])
                )
                tcp_samples.append(pose.tolist())
                gripper_samples.append(held_gripper)
            elapsed_s += timing.duration_s
            endpoint_error = self.ik.error(self.frames.tcp_to_sdk(end_pose), seed)
            reports.append(
                self._segment_report(segment_index, solved, endpoint_error)
            )

        motion_waypoints = len(commands)
        hold_count = max(2, int(np.ceil(self.endpoint_hold_s * self.trajectory_hz)))
        for _ in range(hold_count):
            elapsed_s += 1.0 / self.trajectory_hz
            commands.append(seed.copy())
            relative_times.append(elapsed_s)
            tcp_samples.append(points[-1].tolist())
            gripper_samples.append(held_gripper)

        result = {
            "source": "eef",
            "ik": getattr(self.ik, "name", "arx5_solver.multi_trial_ik + continuous_dls_refinement"),
            "timing": "ruckig_scalar_path_parameterization",
            "path_policy": (
                "model_tcp_targets_preserved_with_bounded_ik_residual_"
                "no_rejection_by_derivative_limits"
            ),
            "ik_execution_tolerance_m_rad": [
                self.ik.execution_translation_tolerance_m,
                self.ik.execution_rotation_tolerance_rad,
            ],
            "eef_model_points": len(requested),
            "joint_waypoints": len(commands),
            "motion_waypoints": motion_waypoints,
            "endpoint_hold_waypoints": hold_count,
            "control_hz": self.trajectory_hz,
            "planned_duration_s": elapsed_s,
            "gripper_during_motion": "held",
            "note": note,
            "_trace": {
                "tcp_samples_xyzrpy": tcp_samples,
                "joint_waypoints_rad": [item.tolist() for item in commands],
                "gripper_waypoints_normalized": gripper_samples,
                "relative_times_s": relative_times,
                "model_tcp_points_xyzrpy": [point.tolist() for point in points],
                "segments": reports,
            },
        }
        return {
            "joint_positions_rad": np.asarray(commands),
            "relative_times_s": relative_times,
            "start_joint_positions_rad": start_joint_positions,
            "start_gripper_normalized": held_gripper,
            "result": result,
        }

    def _solve_segment(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        start_seed: np.ndarray,
        segment_index: int,
    ) -> dict[str, Any]:
        translation_m = float(np.linalg.norm(end_pose[:3] - start_pose[:3]))
        rotation_rad = rotation_distance(start_pose[3:], end_pose[3:])
        fractions, sampled_poses = sample_pose_segment(
            start_pose, end_pose, self.limits
        )
        seed = start_seed.copy()
        joints = [seed.copy()]
        poses = [start_pose.copy()]
        for sample, pose in enumerate(sampled_poses[1:], start=1):
            seed = self.ik.solve(
                self.frames.tcp_to_sdk(pose),
                seed,
                segment_index,
                sample,
            )
            joints.append(seed.copy())
            poses.append(pose)
        timing = retime_path_segment(
            fractions,
            np.asarray(joints),
            translation_m,
            rotation_rad,
            self.limits,
        )
        return {
            "joints": joints,
            "poses": poses,
            "timing": timing,
            "samples": len(joints) - 1,
            "translation_m": translation_m,
            "rotation_rad": rotation_rad,
        }

    def _model_points(
        self,
        requested: list[Any],
        current: np.ndarray,
    ) -> list[np.ndarray]:
        points = [current]
        for item in requested:
            if item is None:
                points.append(points[-1].copy())
            elif isinstance(item, dict):
                points.append(_pose_from_item(item))
            else:
                raise ValueError("each EEF pose must be an object")
        return points

    @staticmethod
    def _segment_report(
        segment_index: int,
        solved: dict[str, Any],
        endpoint_error: tuple[float, float],
    ) -> dict[str, Any]:
        timing = solved["timing"]
        return {
            "segment": segment_index,
            "samples": solved["samples"],
            "translation_m": solved["translation_m"],
            "rotation_rad": solved["rotation_rad"],
            "duration_s": timing.duration_s,
            "time_scale": timing.time_scale,
            "peak_joint_velocity_rad_s": timing.peak_velocity_rad_s.tolist(),
            "peak_joint_acceleration_rad_s2": timing.peak_acceleration_rad_s2.tolist(),
            "peak_joint_jerk_rad_s3": timing.peak_jerk_rad_s3.tolist(),
            "endpoint_fk_translation_error_m": endpoint_error[0],
            "endpoint_fk_rotation_error_rad": endpoint_error[1],
        }


def copy_sdk_vector(owner: Any, accessor: str) -> np.ndarray:
    """Copy a pybind Eigen reference while its owning SDK object is alive."""
    return np.asarray(getattr(owner, accessor)(), dtype=np.float64).copy()


def _pose_from_item(item: dict[str, Any]) -> np.ndarray:
    if "pose_xyzquat" in item:
        pose = np.asarray(item["pose_xyzquat"], dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("pose_xyzquat must be a finite seven-vector")
        return np.r_[pose[:3], quaternion_to_rpy(pose[3:])]
    if "pose_xyzrpy" in item:
        pose = np.asarray(item["pose_xyzrpy"], dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("pose_xyzrpy must be a finite six-vector")
        return pose
    raise ValueError("pose requires pose_xyzquat")
