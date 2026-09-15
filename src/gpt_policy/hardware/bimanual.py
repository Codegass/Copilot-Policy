"""Coordination of two arms through their selected hardware adapters."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..motion.coordination import (
    TrajectoryIKError,
    path_check_result,
    synchronize_bimanual_plan_times,
)
from .robot import ArxRobot
from .motion_control import MotionFault
from ..interrupts import defer_interrupts


class BimanualRobot:
    """Coordinate two independently planned arms on shared waypoint times."""

    def __init__(
        self,
        model: str,
        left_interface: str,
        right_interface: str,
        gripper_open_readout: float | None = None,
        trajectory_hz: float = 30.0,
        settings: dict[str, Any] | None = None,
    ) -> None:
        shared = (model, gripper_open_readout, trajectory_hz, settings)
        self.arms = {}
        try:
            self.arms["left"] = ArxRobot(shared[0], left_interface, *shared[1:])
            self.arms["right"] = ArxRobot(shared[0], right_interface, *shared[1:])
        except BaseException:
            self.close()
            raise
        self.interfaces = {"left": left_interface, "right": right_interface}
        motion = (settings or {}).get("motion", {})
        self.start_delay_s = _positive_float(
            motion.get("bimanual_start_delay_s", 0.1),
            "motion.bimanual_start_delay_s",
        )

    @property
    def dof(self) -> int:
        return self.arms["left"].dof

    @classmethod
    def from_arms(cls, arms: dict[str, Any], interfaces: dict[str, str], settings: dict[str, Any]):
        robot = cls.__new__(cls)
        robot.arms = arms
        robot.interfaces = interfaces
        robot.start_delay_s = _positive_float(
            settings.get("motion", {}).get("bimanual_start_delay_s", 0.1),
            "motion.bimanual_start_delay_s",
        )
        return robot

    def close(self) -> None:
        errors = []
        for arm in self.arms.values():
            try:
                arm.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(f"arm cleanup failed: {errors}")

    def cancel(self) -> None:
        errors = []
        for arm in self.arms.values():
            try:
                fault = next((other.motion.fault for other in self.arms.values()
                              if getattr(getattr(other, "motion", None), "fault", None) is not None), None)
                if fault is not None:
                    arm.fault_stop(fault)
                elif hasattr(arm, "cancel"):
                    arm.cancel()
            except Exception as exc:
                errors.append(exc)
        # A fault on either arm must remove position drive from both arms.
        # Do this before joining workers or finalizing video, and preserve the
        # latch even if cancellation on the peer also failed.
        fault = next((arm.motion.fault for arm in self.arms.values()
                      if getattr(getattr(arm, "motion", None), "fault", None) is not None), None)
        if fault is not None:
            for arm in self.arms.values():
                try:
                    arm.fault_stop(fault)
                except Exception as exc:
                    errors.append(exc)
        if errors:
            if fault is not None:
                raise MotionFault(fault) from errors[0]
            raise RuntimeError(f"arm cancellation failed: {errors}")

    def resume(self) -> None:
        for arm in self.arms.values():
            arm.resume()

    def _parallel(self, calls):
        """Cancel both arms before joining workers, including submission errors."""
        pool = ThreadPoolExecutor(max_workers=len(calls))
        try:
            futures = {side: pool.submit(function, *args) for side, (function, args) in calls.items()}
            for future in as_completed(futures.values()):
                future.result()
            return {side: future.result() for side, future in futures.items()}
        except BaseException as exc:
            with defer_interrupts():
                try:
                    self.cancel()
                except Exception as cleanup_error:
                    exc.add_note(f"Peer cancellation also failed: {cleanup_error!r}")
            raise
        finally:
            with defer_interrupts():
                pool.shutdown(wait=True, cancel_futures=True)

    def state(self) -> dict[str, Any]:
        return {
            "arms": {
                side: {**arm.state(), "interface": self.interfaces[side]}
                for side, arm in self.arms.items()
            },
            "interfaces": dict(self.interfaces),
        }

    def return_home(self) -> dict[str, Any]:
        """Return both arms home concurrently and wait for both to finish."""
        # YAM can prepare joint-space home plans without executing them. Finish
        # both before either arm moves, so peer planning failure cannot send one
        # arm home on its own. ARX retains its existing clearance/home sequence.
        if all(hasattr(arm, "plan_return_home") for arm in self.arms.values()):
            plans = {side: arm.plan_return_home() for side, arm in self.arms.items()}
            calls = {side: (arm.return_home, (plans[side],)) for side, arm in self.arms.items()}
        else:
            calls = {side: (arm.return_home, ()) for side, arm in self.arms.items()}
        arms = self._parallel(calls)
        return {
            "source": "bimanual_return_home",
            "concurrent": True,
            "arms": arms,
            "interfaces": dict(self.interfaces),
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name in {"move_to", "move_eef_chunk", "check_path"}:
            return self._execute_bimanual_eef(name, arguments)
        if name == "set_gripper":
            return self._execute_grippers(arguments)
        if name == "state":
            return self.state()
        raise ValueError(f"双臂模式不支持工具: {name}")

    def _execute_grippers(self, arguments: dict[str, Any]) -> dict[str, Any]:
        positions = arguments.get("positions")
        if not isinstance(positions, dict):
            raise ValueError("bimanual set_gripper requires positions.left/right")
        requested = {
            side: float(position)
            for side, position in positions.items()
            if side in self.arms and position is not None
        }
        unknown = set(positions) - set(self.arms)
        if unknown:
            raise ValueError(f"unknown arm: {sorted(unknown)[0]}")
        if not requested:
            raise ValueError("set_gripper must specify at least one arm")
        motions = self._parallel({
            side: (self.arms[side]._set_gripper, (position,))
            for side, position in requested.items()
        })
        result = self.state()
        result["execution_feedback"] = {
            side: self.arms[side]._gripper_feedback(
                requested[side], result["arms"][side], motions[side]
            )
            for side in requested
        }
        return result

    def _execute_bimanual_eef(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        note = arguments.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("movement tool requires a non-empty note")
        requested_by_arm = self._requested_poses(name, arguments)
        plans: dict[str, dict[str, Any]] = {}
        for side, requested in requested_by_arm.items():
            if all(item is None for item in requested):
                continue
            try:
                plans[side] = self.arms[side].plan_eef_trajectory(requested, note)
            except TrajectoryIKError as exc:
                raise exc.add_arm(side)
        if not plans:
            raise ValueError("bimanual trajectory must command left or right")

        common_durations = synchronize_bimanual_plan_times(plans)
        if name == "check_path":
            return path_check_result(plans)
        self._send_plans(plans)
        self._parallel({side: (self.arms[side].wait_reference_complete, ()) for side in plans})

        arm_states = {side: self.arms[side].state() for side in plans}
        feedback = {
            side: self.arms[side]._execution_feedback(plan, arm_states[side])
            for side, plan in plans.items()
        }
        result = self.state()
        result["trajectory"] = {
            "source": "bimanual_eef",
            "tool": name,
            "arms": {side: plan["result"] for side, plan in plans.items()},
            "concurrent": True,
            "segment_synchronized": len(plans) > 1,
            "common_segment_durations_s": common_durations,
            "coordinated_start_delay_s": self.start_delay_s,
            "planned_duration_s": max(
                plan["result"]["planned_duration_s"] for plan in plans.values()
            ),
        }
        result["execution_feedback"] = feedback
        return result

    def _requested_poses(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, list[Any]]:
        if name == "move_to":
            target = arguments.get("target")
            if not isinstance(target, dict):
                raise ValueError("bimanual move_to.target requires left/right")
            return {side: [target.get(side)] for side in self.arms}
        poses = arguments.get("poses")
        if not isinstance(poses, list) or not poses:
            raise ValueError("bimanual move_eef_chunk.poses must be a non-empty point list")
        return {
            side: [point.get(side) if isinstance(point, dict) else None for point in poses]
            for side in self.arms
        }

    def _send_plans(self, plans: dict[str, dict[str, Any]]) -> None:
        if len(plans) == 1:
            side, plan = next(iter(plans.items()))
            self.arms[side].send_eef_trajectory(plan, wait=False)
            return
        clocks = self._parallel({side: (self.arms[side].clock, ()) for side in plans})
        starts = {side: now + self.start_delay_s for side, now in clocks.items()}
        self._parallel({
            side: (self.arms[side].send_eef_trajectory, (plan, False, starts[side]))
            for side, plan in plans.items()
        })


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number
