"""Trajectory diagnostics and time-only synchronization shared by robot layers."""

from __future__ import annotations

from typing import Any

import numpy as np


def path_check_result(plans):
    """Report the existing planner's checks without implying hardware execution."""
    return {"path_check": {"accepted": True, "executed": False,
        "checks": ["inverse_kinematics", "joint_limits", "trajectory_timing"],
        "collision_checked": False, "replanned_before_execution": True,
        "arms": {side: {"planned_duration_s": p["result"]["planned_duration_s"],
                        "planned_tcp_points_xyzrpy": p["result"]["_trace"]["model_tcp_points_xyzrpy"]}
                 for side, p in plans.items()}}}


class TrajectoryIKError(RuntimeError):
    """A requested TCP sample has no solution inside the configured tolerance."""

    def __init__(
        self,
        *,
        segment: int,
        sample: int,
        status: int,
        status_name: str,
        translation_error_m: float,
        rotation_error_rad: float,
        reason: str | None = None,
        arm: str | None = None,
    ) -> None:
        self.details = {
            "reason": reason or _ik_failure_reason(status),
            "segment": int(segment),
            "sample": int(sample),
            "sdk_status": int(status),
            "sdk_status_name": str(status_name),
            "best_translation_error_m": float(translation_error_m),
            "best_rotation_error_rad": float(rotation_error_rad),
        }
        if arm is not None:
            self.details["arm"] = arm
        super().__init__(
            f"IK has no solution at segment {segment} sample {sample}: "
            f"{status_name} ({status}); residual={translation_error_m:.6g} m, "
            f"{rotation_error_rad:.6g} rad"
        )

    def add_arm(self, arm: str) -> TrajectoryIKError:
        self.details["arm"] = arm
        return self


def _ik_failure_reason(status: int) -> str:
    if status == -9:
        return "ik_joint_limit"
    if status == 0:
        return "ik_residual_above_tolerance"
    return "ik_solver_failure"


def synchronize_bimanual_plan_times(
    plans: dict[str, dict[str, Any]],
) -> list[float]:
    """Put independently safe paths on one per-segment clock by slowing only."""
    if len(plans) < 2:
        return []
    traces = {side: plan["result"]["_trace"] for side, plan in plans.items()}
    segment_count = len(next(iter(traces.values()))["segments"])
    if any(len(trace["segments"]) != segment_count for trace in traces.values()):
        raise ValueError("bimanual plans must have matching model segment counts")
    common_durations = [
        max(float(trace["segments"][index]["duration_s"]) for trace in traces.values())
        for index in range(segment_count)
    ]

    for side, plan in plans.items():
        result = plan["result"]
        trace = traces[side]
        old_times = np.asarray(plan["relative_times_s"], dtype=np.float64)
        motion_count = int(result["motion_waypoints"])
        new_times = np.empty_like(old_times)
        command_index = 0
        old_start = 0.0
        common_start = 0.0
        for report, common_duration in zip(trace["segments"], common_durations):
            samples = int(report["samples"])
            old_duration = float(report["duration_s"])
            stop = command_index + samples
            scale = common_duration / old_duration
            new_times[command_index:stop] = common_start + (
                old_times[command_index:stop] - old_start
            ) * scale
            report["independent_duration_s"] = old_duration
            report["duration_s"] = common_duration
            report["bimanual_time_scale"] = scale
            command_index = stop
            old_start += old_duration
            common_start += common_duration
        if command_index != motion_count:
            raise ValueError(f"{side} segment sample count does not match motion plan")
        new_times[motion_count:] = common_start + (
            old_times[motion_count:] - old_start
        )
        if np.any(np.diff(new_times) <= 0):
            raise ValueError(f"{side} synchronized timestamps are not increasing")
        synchronized = new_times.tolist()
        plan["relative_times_s"] = synchronized
        trace["relative_times_s"] = synchronized
        result["planned_duration_s"] = float(new_times[-1])
        result["bimanual_synchronized"] = True
    return common_durations
