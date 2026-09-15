"""Path-preserving trajectory timing built on the open-source Ruckig OTG.

Ruckig is deliberately applied to the scalar path coordinate ``s`` rather
than independently to every robot joint.  The input joint and TCP samples are
therefore never moved, removed, reordered, or rounded: this module only
assigns later timestamps when a derivative limit requires more time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ruckig import InputParameter, Result, Ruckig, Trajectory

from ..geometry.poses import interpolate_pose, rotation_distance


_EPS = 1e-9


@dataclass(frozen=True)
class MotionLimits:
    """Execution limits; exceeding one stretches time instead of rejecting a path."""

    output_hz: float
    cartesian_step_m: float
    cartesian_step_rad: float
    tcp_velocity_m_s: float
    tcp_angular_velocity_rad_s: float
    joint_velocity_rad_s: np.ndarray
    joint_acceleration_rad_s2: np.ndarray
    joint_jerk_rad_s3: np.ndarray

    def __post_init__(self) -> None:
        scalars = np.asarray(
            [
                self.output_hz,
                self.cartesian_step_m,
                self.cartesian_step_rad,
                self.tcp_velocity_m_s,
                self.tcp_angular_velocity_rad_s,
            ],
            dtype=np.float64,
        )
        vectors = (
            np.asarray(self.joint_velocity_rad_s, dtype=np.float64),
            np.asarray(self.joint_acceleration_rad_s2, dtype=np.float64),
            np.asarray(self.joint_jerk_rad_s3, dtype=np.float64),
        )
        if not np.isfinite(scalars).all() or np.any(scalars <= 0):
            raise ValueError("motion timing scalars must be finite and positive")
        if any(
            value.ndim != 1
            or not np.isfinite(value).all()
            or np.any(value <= 0)
            for value in vectors
        ):
            raise ValueError("joint motion limits must be finite positive vectors")
        if len({value.size for value in vectors}) != 1:
            raise ValueError(
                "joint velocity, acceleration, and jerk limits must have equal lengths"
            )
        object.__setattr__(self, "joint_velocity_rad_s", vectors[0])
        object.__setattr__(self, "joint_acceleration_rad_s2", vectors[1])
        object.__setattr__(self, "joint_jerk_rad_s3", vectors[2])


@dataclass(frozen=True)
class TimedSegment:
    """The original samples with a monotonically increasing relative timeline."""

    times_s: np.ndarray
    time_scale: float
    duration_s: float
    peak_velocity_rad_s: np.ndarray
    peak_acceleration_rad_s2: np.ndarray
    peak_jerk_rad_s3: np.ndarray


def sample_count_for_segment(
    translation_m: float,
    rotation_rad: float,
    limits: MotionLimits,
) -> int:
    """Choose density from geometry and nominal duration, never from waypoint count."""
    geometric = max(
        translation_m / limits.cartesian_step_m,
        rotation_rad / limits.cartesian_step_rad,
    )
    nominal_duration = max(
        translation_m / limits.tcp_velocity_m_s,
        rotation_rad / limits.tcp_angular_velocity_rad_s,
        1.0 / limits.output_hz,
    )
    temporal = nominal_duration * limits.output_hz
    return max(2, int(np.ceil(max(geometric, temporal))))


def sample_pose_segment(
    start_pose: object,
    end_pose: object,
    limits: MotionLimits,
) -> tuple[np.ndarray, np.ndarray]:
    """Densify exactly on the input line/SLERP, including both immutable endpoints."""
    start = np.asarray(start_pose, dtype=np.float64)
    end = np.asarray(end_pose, dtype=np.float64)
    if start.shape != (6,) or end.shape != (6,):
        raise ValueError("pose endpoints must be six-vectors")
    translation_m = float(np.linalg.norm(end[:3] - start[:3]))
    rotation_rad = rotation_distance(start[3:], end[3:])
    count = sample_count_for_segment(translation_m, rotation_rad, limits)
    fractions = np.linspace(0.0, 1.0, count + 1)
    poses = np.asarray([interpolate_pose(start, end, value) for value in fractions])
    # Avoid even round-off-level endpoint drift from quaternion conversion.
    poses[0] = start
    poses[-1] = end
    return fractions, poses


def retime_path_segment(
    fractions: object,
    joint_positions: object,
    translation_m: float,
    rotation_rad: float,
    limits: MotionLimits,
) -> TimedSegment:
    """Assign jerk-limited timestamps while preserving every supplied sample.

    The Ruckig profile starts and ends at zero velocity and acceleration.  A
    final analytic scale then makes measured finite differences respect all
    joint limits.  That scale can only increase duration.
    """
    path = np.asarray(fractions, dtype=np.float64)
    joints = np.asarray(joint_positions, dtype=np.float64)
    if path.ndim != 1 or path.size < 3 or path[0] != 0.0 or path[-1] != 1.0:
        raise ValueError("fractions must run from 0 to 1 with at least three samples")
    if joints.ndim != 2 or joints.shape[0] != path.size:
        raise ValueError("joint_positions must have one row per path fraction")
    if joints.shape[1] != limits.joint_velocity_rad_s.size:
        raise ValueError("joint_positions DOF does not match motion limits")
    if (
        not np.isfinite(path).all()
        or not np.isfinite(joints).all()
        or np.any(np.diff(path) <= 0)
    ):
        raise ValueError("trajectory samples must be finite and strictly ordered")

    q_s = np.gradient(joints, path, axis=0, edge_order=2)
    q_ss = np.gradient(q_s, path, axis=0, edge_order=2)
    max_q_s = np.max(np.abs(q_s), axis=0)
    max_q_ss = np.max(np.abs(q_ss), axis=0)

    velocity_candidates: list[float] = []
    if translation_m > _EPS:
        velocity_candidates.append(limits.tcp_velocity_m_s / translation_m)
    if rotation_rad > _EPS:
        velocity_candidates.append(limits.tcp_angular_velocity_rad_s / rotation_rad)
    active_q_s = max_q_s > _EPS
    if np.any(active_q_s):
        velocity_candidates.append(
            float(np.min(limits.joint_velocity_rad_s[active_q_s] / max_q_s[active_q_s]))
        )

    # Reserve half of each acceleration allowance for path curvature q''(s)s_dot^2.
    active_q_ss = max_q_ss > _EPS
    if np.any(active_q_ss):
        curvature_velocity = np.sqrt(
            0.5
            * limits.joint_acceleration_rad_s2[active_q_ss]
            / max_q_ss[active_q_ss]
        )
        velocity_candidates.append(float(np.min(curvature_velocity)))
    max_path_velocity = max(_EPS, min(velocity_candidates or [1.0]))

    if float(np.max(np.abs(joints - joints[0]))) <= _EPS:
        times = np.linspace(0.0, 2.0 / limits.output_hz, path.size)
        zeros = np.zeros(joints.shape[1], dtype=np.float64)
        return TimedSegment(
            times_s=times,
            time_scale=1.0,
            duration_s=float(times[-1]),
            peak_velocity_rad_s=zeros,
            peak_acceleration_rad_s2=zeros,
            peak_jerk_rad_s3=zeros,
        )

    if np.any(active_q_s):
        max_path_acceleration = float(
            np.min(
                0.5
                * limits.joint_acceleration_rad_s2[active_q_s]
                / max_q_s[active_q_s]
            )
        )
        max_path_jerk = float(
            np.min(limits.joint_jerk_rad_s3[active_q_s] / max_q_s[active_q_s])
        )
    else:
        max_path_acceleration = 1.0
        max_path_jerk = 1.0
    max_path_acceleration = max(_EPS, max_path_acceleration)
    max_path_jerk = max(_EPS, max_path_jerk)

    otg = Ruckig(1)
    request = InputParameter(1)
    request.current_position = [0.0]
    request.current_velocity = [0.0]
    request.current_acceleration = [0.0]
    request.target_position = [1.0]
    request.target_velocity = [0.0]
    request.target_acceleration = [0.0]
    request.max_velocity = [max_path_velocity]
    request.max_acceleration = [max_path_acceleration]
    request.max_jerk = [max_path_jerk]
    profile = Trajectory(1)
    result = otg.calculate(request, profile)
    if result not in (Result.Working, Result.Finished):
        raise RuntimeError(f"Ruckig could not parameterize scalar path: {result}")

    times = np.asarray(
        [profile.get_first_time_at_position(0, float(value)) for value in path],
        dtype=np.float64,
    )
    # Ruckig's root finder can return an endpoint a few microseconds early.
    times[0] = 0.0
    times[-1] = float(profile.duration)
    times = np.maximum.accumulate(times)
    if np.any(np.diff(times) <= 0):
        raise RuntimeError("Ruckig produced a non-increasing scalar timeline")

    velocity, acceleration, jerk = derivatives(joints, times)
    ratio_v = _peak_ratio(velocity, limits.joint_velocity_rad_s)
    ratio_a = _peak_ratio(acceleration, limits.joint_acceleration_rad_s2)
    ratio_j = _peak_ratio(jerk, limits.joint_jerk_rad_s3)
    scale = max(1.0, ratio_v, np.sqrt(ratio_a), np.cbrt(ratio_j))
    if scale > 1.0:
        # A tiny margin prevents floating-point rechecks landing just above a limit.
        scale *= 1.001
        times *= scale
        velocity /= scale
        acceleration /= scale**2
        jerk /= scale**3

    return TimedSegment(
        times_s=times,
        time_scale=float(scale),
        duration_s=float(times[-1]),
        peak_velocity_rad_s=np.max(np.abs(velocity), axis=0),
        peak_acceleration_rad_s2=np.max(np.abs(acceleration), axis=0),
        peak_jerk_rad_s3=np.max(np.abs(jerk), axis=0),
    )


def derivatives(
    joint_positions: object, times_s: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-uniform finite-difference q-dot, q-double-dot, and jerk."""
    joints = np.asarray(joint_positions, dtype=np.float64)
    times = np.asarray(times_s, dtype=np.float64)
    edge_order = 2 if times.size >= 3 else 1
    velocity = np.gradient(joints, times, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, times, axis=0, edge_order=edge_order)
    jerk = np.gradient(acceleration, times, axis=0, edge_order=edge_order)
    return velocity, acceleration, jerk


def _peak_ratio(values: np.ndarray, limits: np.ndarray) -> float:
    return float(np.max(np.max(np.abs(values), axis=0) / limits))
