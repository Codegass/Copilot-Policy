"""Single-arm ARX5 SDK lifecycle and command execution facade."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry.frames import FrameCalibration
from ..geometry.poses import (
    rotation_distance,
    rpy_to_quaternion,
)
from ..motion.ik import ContinuousIK
from ..motion.planner import EefTrajectoryPlanner, copy_sdk_vector
from ..motion.trajectory import MotionLimits
from .gripper import GripperStreamer
from .feedback import motion_progress
from .motion_control import MotionControl, MotionFault


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _positive_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"{name} must be a finite positive scalar or {size}-vector")
    return array


def _load_sdk() -> Any:
    project_root = Path(__file__).resolve().parents[3]
    sdk_python = project_root / "third_party" / "arx5-sdk" / "python"
    sys.path.insert(0, str(sdk_python))
    try:
        import arx5_interface
    except ImportError as exc:
        raise RuntimeError(
            "找不到 ARX SDK Python 模块。请按照 third_party/arx5-sdk/README.md 构建 "
            "或安装 arx5-interface。"
        ) from exc
    return arx5_interface


class ArxRobot:
    """Expose the SDK's joint controller and solver without another robot stack."""

    def __init__(
        self,
        model: str,
        interface: str,
        gripper_open_readout: float | None = None,
        trajectory_hz: float = 30.0,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.motion = MotionControl()
        self.interface = interface
        self.sdk = _load_sdk()
        self.frames = FrameCalibration(settings)
        robot_settings = (settings or {}).get("robot", {})
        robot_config = self.sdk.RobotConfigFactory.get_instance().get_config(model)
        if gripper_open_readout is not None:
            robot_config.gripper_open_readout = float(gripper_open_readout)
        if "gripper_width_m" in robot_settings:
            robot_config.gripper_width = float(robot_settings["gripper_width_m"])
        gripper_velocity_rad_s = float(
            robot_settings.get("gripper_velocity_rad_s", 3.0)
        )
        gripper_scale = abs(float(robot_config.gripper_open_readout))
        if (
            not np.isfinite(gripper_velocity_rad_s)
            or gripper_velocity_rad_s <= 0
            or not np.isfinite(gripper_scale)
            or gripper_scale <= 0
        ):
            raise ValueError("gripper calibration and velocity must be positive")
        robot_config.gripper_vel_max = (
            gripper_velocity_rad_s / gripper_scale
            * float(robot_config.gripper_width)
        )
        controller_config = self.sdk.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot_config.joint_dof
        )
        # Use the same native-radian position-control convention as the
        # Stanford reference: the SDK receives encoder-relative radians
        # directly; there is no degrees/180-degree conversion layer.
        # The vendor joint-controller default is gravity compensation enabled.
        # Keeping it off caused a repeatable 17--28 mm downward TCP error in
        # loaded poses, so follow the SDK default unless explicitly disabled.
        gravity_compensation = robot_settings.get("gravity_compensation", True)
        if not isinstance(gravity_compensation, bool):
            raise ValueError("robot.gravity_compensation must be a boolean")
        controller_config.gravity_compensation = gravity_compensation
        self.gravity_compensation = gravity_compensation
        if int(robot_config.joint_dof) == 6:
            controller_config.default_kp = np.asarray(
                [150.0, 150.0, 150.0, 50.0, 10.0, 40.0], dtype=np.float64
            )
            controller_config.default_kd = np.ones(6, dtype=np.float64)
        controller_config.default_gripper_kp = float(
            robot_settings.get("gripper_kp", 2.0)
        )
        controller_config.default_gripper_kd = float(
            robot_settings.get("gripper_kd", 0.1)
        )
        self.controller = self.sdk.Arx5JointController(
            robot_config, controller_config, interface
        )
        self.config = self.controller.get_robot_config()
        if not np.isfinite(trajectory_hz) or trajectory_hz <= 0:
            raise ValueError("trajectory_hz must be finite and positive")
        self.trajectory_hz = float(trajectory_hz)
        motion_settings = (settings or {}).get("motion", {})
        self.home_time_scale = _positive_float(
            motion_settings.get("home_time_scale", 3.0), "motion.home_time_scale"
        )
        velocity_scale = float(motion_settings.get("joint_velocity_scale", 0.25))
        if not np.isfinite(velocity_scale) or velocity_scale <= 0:
            raise ValueError("motion.joint_velocity_scale must be finite and positive")
        self.motion_limits = MotionLimits(
            output_hz=self.trajectory_hz,
            cartesian_step_m=float(motion_settings.get("cartesian_step_m", 0.005)),
            cartesian_step_rad=float(motion_settings.get("cartesian_step_rad", 0.035)),
            tcp_velocity_m_s=float(motion_settings.get("tcp_velocity_m_s", 0.08)),
            tcp_angular_velocity_rad_s=float(
                motion_settings.get("tcp_angular_velocity_rad_s", 0.5)
            ),
            joint_velocity_rad_s=np.asarray(robot_config.joint_vel_max, dtype=np.float64)
            * velocity_scale,
            joint_acceleration_rad_s2=_positive_vector(
                motion_settings.get("joint_acceleration_rad_s2", 2.0),
                int(robot_config.joint_dof),
                "joint_acceleration_rad_s2",
            ),
            joint_jerk_rad_s3=_positive_vector(
                motion_settings.get("joint_jerk_rad_s3", 12.0),
                int(robot_config.joint_dof),
                "joint_jerk_rad_s3",
            ),
        )
        self.endpoint_hold_s = _positive_float(
            motion_settings.get("endpoint_hold_s", 0.12), "endpoint_hold_s"
        )
        self.settle_position_tolerance_rad = _positive_float(
            motion_settings.get("settle_position_tolerance_rad", 0.03),
            "settle_position_tolerance_rad",
        )
        self.settle_velocity_tolerance_rad_s = _positive_float(
            motion_settings.get("settle_velocity_tolerance_rad_s", 0.05),
            "settle_velocity_tolerance_rad_s",
        )
        self.settle_samples = max(1, int(motion_settings.get("settle_samples", 10)))
        self.settle_timeout_s = _positive_float(
            motion_settings.get("settle_timeout_s", 3.0), "settle_timeout_s"
        )
        self.tracking_error_limit_rad = _positive_float(
            motion_settings.get("tracking_error_limit_rad", 0.12), "tracking_error_limit_rad"
        )
        self.ik_refine_iterations = max(
            0, int(motion_settings.get("ik_refine_iterations", 12))
        )
        self.ik_translation_tolerance_m = _positive_float(
            motion_settings.get("ik_translation_tolerance_m", 1e-4),
            "ik_translation_tolerance_m",
        )
        self.ik_rotation_tolerance_rad = _positive_float(
            motion_settings.get("ik_rotation_tolerance_rad", 5e-4),
            "ik_rotation_tolerance_rad",
        )
        self.ik_execution_translation_tolerance_m = _positive_float(
            motion_settings.get("ik_execution_translation_tolerance_m", 0.002),
            "ik_execution_translation_tolerance_m",
        )
        self.ik_execution_rotation_tolerance_rad = _positive_float(
            motion_settings.get("ik_execution_rotation_tolerance_rad", 0.0174532925),
            "ik_execution_rotation_tolerance_rad",
        )
        if (
            self.ik_execution_translation_tolerance_m
            < self.ik_translation_tolerance_m
            or self.ik_execution_rotation_tolerance_rad
            < self.ik_rotation_tolerance_rad
        ):
            raise ValueError("IK execution tolerances must include numeric tolerances")
        self._arm_target: np.ndarray | None = None
        self._gripper_target = 0.0
        self._reference_end_s = 0.0
        self._last_settle_report: dict[str, Any] = {}
        self._hold_current_pose()
        self.solver = self.sdk.Arx5Solver(
            self.config.urdf_path,
            self.config.joint_dof,
            self.config.joint_pos_min,
            self.config.joint_pos_max,
            self.config.base_link_name,
            self.config.eef_link_name,
            self.config.gravity_vector,
        )
        self.ik = ContinuousIK(
            self.solver,
            self.config.joint_pos_min,
            self.config.joint_pos_max,
            self.ik_refine_iterations,
            self.ik_translation_tolerance_m,
            self.ik_rotation_tolerance_rad,
            self.ik_execution_translation_tolerance_m,
            self.ik_execution_rotation_tolerance_rad,
        )
        self.planner = self._make_trajectory_planner()
        self.gripper = GripperStreamer(
            self.controller,
            self._target_command,
            float(self.config.gripper_width),
            float(self.config.gripper_open_readout),
            gripper_velocity_rad_s,
            float(robot_settings.get("gripper_control_hz", 100.0)),
            float(self.config.gripper_torque_max),
            send_command=self._send_command,
        )

    @property
    def dof(self) -> int:
        return int(self.config.joint_dof)

    def state(self) -> dict[str, Any]:
        joint = self.controller.get_joint_state()
        command = self.controller.get_joint_cmd()
        eef = self.controller.get_eef_state()
        sdk_pose = np.asarray(eef.pose_6d(), dtype=np.float64)
        pose = self.frames.sdk_to_tcp(sdk_pose)
        return {
            "joint_positions_rad": joint.pos().tolist(),
            "joint_velocities_rad_s": joint.vel().tolist(),
            "joint_torques_nm": joint.torque().tolist(),
            "joint_command_positions_rad": command.pos().tolist(),
            "joint_command_velocities_rad_s": command.vel().tolist(),
            "joint_command_torques_nm": command.torque().tolist(),
            "command_timestamp_s": float(command.timestamp),
            "tcp_xyzrpy": pose.tolist(),
            "tcp_xyzquat": [*pose[:3].tolist(), *rpy_to_quaternion(pose[3:])],
            "sdk_eef_xyzrpy": sdk_pose.tolist(),
            "gripper_position_m": float(joint.gripper_pos),
            "gripper_normalized": self._gripper_to_normalized(float(joint.gripper_pos)),
            "gripper_command_normalized": self._gripper_target,
            "gripper_velocity_m_s": float(joint.gripper_vel),
            "gripper_torque_nm": float(joint.gripper_torque),
            "gravity_compensation": self.gravity_compensation,
            "timestamp_s": float(joint.timestamp),
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "check_path":
            from ..motion.coordination import path_check_result
            return path_check_result({"arm": self._move_eef_trajectory(name, arguments)})
        if name == "state":
            return self.state()
        if name == "move_to":
            plan = self._move_eef_trajectory(name, arguments)
        elif name == "move_eef_chunk":
            plan = self._move_eef_trajectory(name, arguments)
        elif name == "move_joints":
            self._move_joints(arguments)
            plan = None
        elif name == "move_eef":
            self._move_eef(arguments)
            plan = None
        elif name == "set_gripper":
            target_gripper = self._gripper_argument(arguments)
            gripper_motion = self._set_gripper(target_gripper)
            plan = None
        elif name == "set_torque":
            command = self._command_copy()
            command.torque()[:] = np.asarray(arguments["torques"], dtype=np.float64)
            self._send_command(command)
            plan = None
        elif name == "set_gain":
            self._set_gain(arguments)
            plan = None
        elif name == "home":
            return self.return_home()
        elif name == "damping":
            self.controller.set_to_damping()
            plan = None
        else:
            raise ValueError(f"未知工具: {name}")
        result = self.state()
        if plan is not None:
            result["trajectory"] = plan["result"]
            result["execution_feedback"] = self._execution_feedback(plan, result)
        elif name == "set_gripper":
            result["execution_feedback"] = self._gripper_feedback(
                target_gripper, result, gripper_motion
            )
        return result

    def return_home(self) -> dict[str, Any]:
        """Follow the two-stage home path, checking feedback throughout both legs."""
        self.motion.check()
        self._check_feedback()
        # reset_to_home() hard-codes its timing and exposes no speed argument.
        # Keep its clearance waypoint and gripper opening, but send timestamped
        # commands and verify each leg before proceeding. Never re-arm damping.
        start = self._command_copy()
        max_error = max(
            float(np.max(np.abs(start.pos()))),
            2.0 * abs(self._gripper_to_normalized(start.gripper_pos) - 1.0),
        )
        approach_s = max(max_error, 0.5) * self.home_time_scale
        final_s = 0.5 * self.home_time_scale
        initial_gain = self.controller.get_gain()
        if np.allclose(np.asarray(initial_gain.kp()), 0.0):
            self._raise_motion_fault({"reason": "controller_in_damping", "phase": "return_home"})

        self._send_command(start)
        target = self.sdk.JointState(self.dof)
        target.pos()[2] = 0.03  # SDK clearance waypoint before the final zero pose.
        target.gripper_pos = float(self.config.gripper_width)
        started = self.controller.get_timestamp()
        target.timestamp = started + approach_s
        self._send_command(target)
        self._arm_target = np.asarray(target.pos(), dtype=np.float64).copy()
        self._reference_end_s = target.timestamp
        self.wait_reference_complete()
        self._require_home_settled()

        final = self.sdk.JointState(self.dof)
        final.gripper_pos = float(self.config.gripper_width)
        final.timestamp = self.controller.get_timestamp() + final_s
        self._send_command(final)
        self._arm_target = np.zeros(self.dof, dtype=np.float64)
        self._gripper_target = 1.0
        self._reference_end_s = final.timestamp
        self.wait_reference_complete()
        self._require_home_settled()
        result = self.state()
        actual = np.asarray(result["joint_positions_rad"], dtype=np.float64)
        result["home"] = {
            "source": "gpt_policy.return_home",
            "time_scale": self.home_time_scale,
            "segment_durations_s": [approach_s, final_s],
            "target_joint_positions_rad": self._arm_target.tolist(),
            "max_joint_residual_rad": float(np.max(np.abs(actual))),
            "gripper_target_normalized": self._gripper_target,
            "settle": dict(self._last_settle_report),
        }
        return result

    def _require_home_settled(self):
        if not self._last_settle_report["settled"]:
            self._raise_motion_fault({"reason": "home_not_settled", **self._last_settle_report})

    def _command_copy(self) -> Any:
        # Feedback effort is not feed-forward effort. Partial position commands
        # start at measured positions with zero velocity and torque.
        source = self.controller.get_joint_state()
        command = self.sdk.JointState(self.dof)
        command.pos()[:] = source.pos()
        command.vel()[:] = 0.0
        command.torque()[:] = 0.0
        command.gripper_pos = source.gripper_pos
        command.gripper_vel = 0.0
        command.gripper_torque = 0.0
        # The SDK command timestamp is consumed by the interpolator. Reusing
        # the old output timestamp makes a fresh partial command stale and can
        # replay an unintended joint trajectory; zero asks the SDK to assign
        # its normal preview timestamp.
        command.timestamp = 0.0
        return command

    def _gripper_argument(self, arguments: dict[str, Any]) -> float:
        if "gripper" in arguments:
            value = float(arguments["gripper"])
        elif "position" in arguments:  # legacy meter-based protocol
            value = self._gripper_to_normalized(float(arguments["position"]))
        else:
            raise ValueError("set_gripper requires gripper in [0, 1]")
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("gripper must be a finite number in [0, 1]")
        return value

    def _normalized_to_meters(self, value: float) -> float:
        return float(value) * float(self.config.gripper_width)

    def _gripper_to_normalized(self, value: float) -> float:
        width = float(self.config.gripper_width)
        return float(value / width) if width else 0.0

    def _execution_feedback(self, plan: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
        commands = plan.get("commands", [])
        if not commands:
            return {"joint_residual_rad": [], "max_joint_residual_rad": 0.0}
        target = np.asarray(commands[-1].pos(), dtype=np.float64)
        actual = np.asarray(measured["joint_positions_rad"], dtype=np.float64)
        residual = np.abs(actual - target)
        target_pose = np.asarray(
            plan["result"]["_trace"]["tcp_samples_xyzrpy"][-1], dtype=np.float64
        )
        # FK uses this exact joint snapshot and its timestamp; state()'s SDK
        # EEF snapshot is acquired separately and can represent a later sample.
        actual_pose = self.frames.sdk_to_tcp(self.solver.forward_kinematics(actual))
        start = plan["result"]["_trace"]["execution_start"]
        # This snapshot was captured before SDK submission. Resolve its TCP
        # here so extra FK work cannot consume a coordinated-start deadline.
        start["tcp_xyzrpy"] = self.frames.sdk_to_tcp(self.solver.forward_kinematics(
            np.asarray(start["joint_positions_rad"], dtype=np.float64)
        )).tolist()
        position_error = target_pose[:3] - actual_pose[:3]
        return {
            "joint_residual_rad": residual.tolist(),
            "max_joint_residual_rad": float(np.max(residual)),
            "target_tcp_xyzrpy": target_pose.tolist(),
            "measured_tcp_xyzrpy": actual_pose.tolist(),
            "tcp_error_xyz_m": position_error.tolist(),
            "tcp_translation_error_m": float(np.linalg.norm(position_error)),
            "tcp_rotation_error_rad": rotation_distance(target_pose[3:], actual_pose[3:]),
            "motion_progress": motion_progress(
                start, target_pose, actual_pose, measured["timestamp_s"],
            ),
            "gripper_measured_normalized": measured["gripper_normalized"],
            "gripper_torque_nm": measured["gripper_torque_nm"],
            "settle": dict(self._last_settle_report),
        }

    def _gripper_feedback(
        self,
        target: float,
        measured: dict[str, Any],
        motion: dict[str, Any],
    ) -> dict[str, Any]:
        residual = abs(float(target) - measured["gripper_normalized"])
        return {
            "gripper_target_normalized": float(target),
            "gripper_active_command_normalized": motion["active_command_normalized"],
            "gripper_measured_normalized": measured["gripper_normalized"],
            "gripper_residual_normalized": residual,
            "gripper_torque_nm": measured["gripper_torque_nm"],
            "motion": motion,
        }

    def _set_gripper(self, position: float) -> dict[str, Any]:
        # A gripper-only command inherits the last arm target accepted by this
        # process. It must not turn a partially reached measured pose into a
        # new arm target and freeze the approach early.
        self.wait_reference_complete()
        # Arm the position loops before streaming the first reference.
        gain = self.controller.get_gain()
        config = self.controller.get_controller_config()
        if np.allclose(np.asarray(gain.kp()), 0.0):
            gain.kp()[:] = config.default_kp
            gain.kd()[:] = config.default_kd
        gain.gripper_kp = float(config.default_gripper_kp)
        gain.gripper_kd = float(config.default_gripper_kd)
        self.motion.send(self.controller.set_gain, gain)
        motion = self.gripper.move(position)
        # Preserve the actual active reference under contact. Replaying the
        # unreachable requested endpoint during the next arm move would build
        # the same position error again.
        self._gripper_target = float(motion["active_command_normalized"])
        return motion

    def _move_eef_trajectory(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "move_to":
            target = arguments.get("target")
            if not isinstance(target, dict):
                raise ValueError("move_to.target must be an object")
            requested = [target]
        else:
            requested = arguments.get("poses")
            if not isinstance(requested, list) or not requested:
                raise ValueError("move_eef_chunk.poses must be a non-empty pose list")
        note = arguments.get("note")
        if not isinstance(note, str) or not note.strip():
            raise ValueError("movement tool requires a non-empty note")
        plan = self.plan_eef_trajectory(requested, note)
        if name != "check_path":
            self.send_eef_trajectory(plan)
        return plan

    def plan_eef_trajectory(self, requested: list[Any], note: str) -> dict[str, Any]:
        """Solve an EEF waypoint list without sending it to CAN."""
        planner = getattr(self, "planner", None) or self._make_trajectory_planner()
        plan = planner.plan(requested, note, self._gripper_target)
        commands = []
        for positions in plan["joint_positions_rad"]:
            command = self.sdk.JointState(self.dof)
            command.pos()[:] = positions
            command.vel()[:] = 0.0
            command.torque()[:] = 0.0
            command.gripper_pos = self._normalized_to_meters(self._gripper_target)
            command.timestamp = 0.0
            commands.append(command)
        plan["commands"] = commands
        return plan

    def clock(self) -> float:
        return float(self.controller.get_timestamp())

    def _send_command(self, command) -> None:
        self._check_feedback()
        self.motion.send(self.controller.set_joint_cmd, command)

    def cancel(self) -> None:
        if self.motion.fault is not None:
            self.fault_stop(self.motion.fault)
            return
        self._check_feedback()
        def hold():
            command = self._command_copy()
            # Zero uses the SDK's preview interval and replaces all queued
            # waypoints; a sampled "now" can become stale before submission.
            command.timestamp = 0.0
            self.controller.set_joint_cmd(command)
            self._arm_target = np.asarray(command.pos(), dtype=np.float64).copy()
            self._gripper_target = self._gripper_to_normalized(command.gripper_pos)
            self._reference_end_s = self.controller.get_timestamp()
        self.motion.cancel(hold)

    def fault_stop(self, details):
        # Latch before taking the send lock, so no worker can submit another
        # target while damping replaces the SDK's complete reference queue.
        self.motion.fault = self.motion.fault or details
        self.motion.cancel(self.controller.set_to_damping)

    def _raise_motion_fault(self, details):
        self.fault_stop({"interface": self.interface, **details})
        raise MotionFault(self.motion.fault)

    def _check_feedback(self):
        if self.motion.fault is not None:
            raise MotionFault(self.motion.fault)
        # Keep both pybind owners alive: their Eigen arrays are borrowed views.
        try:
            measured = self.controller.get_joint_state()
            command = self.controller.get_joint_cmd()
            actual = np.asarray(measured.pos(), dtype=np.float64)
            reference = np.asarray(command.pos(), dtype=np.float64)
            velocity = np.asarray(measured.vel(), dtype=np.float64)
            torque = np.asarray(measured.torque(), dtype=np.float64)
        except Exception as exc:
            self._raise_motion_fault({"reason": "feedback_unavailable", "error": repr(exc)})
        if not all(np.isfinite(v).all() for v in (actual, reference, velocity, torque)):
            self._raise_motion_fault({"reason": "nonfinite_feedback"})
        checks = (
            ("tracking_error", np.abs(actual - reference), np.full(self.dof, self.tracking_error_limit_rad)),
            ("joint_overspeed", np.abs(velocity), np.asarray(self.config.joint_vel_max)),
            ("joint_overload", np.abs(torque), np.asarray(self.config.joint_torque_max)),
        )
        for reason, values, limits in checks:
            if np.any(values > limits):
                joint = int(np.argmax(values / limits))
                self._raise_motion_fault({
                    "reason": reason, "joint": joint, "value": float(values[joint]),
                    "limit": float(limits[joint]), "measured_joint_positions_rad": actual.tolist(),
                    "command_joint_positions_rad": reference.tolist(),
                    "joint_velocities_rad_s": velocity.tolist(), "joint_torques_nm": torque.tolist(),
                })

    def resume(self) -> None:
        self.motion.resume()

    def close(self) -> None:
        # The native controller destructor owns its thread shutdown.
        for name in ("gripper", "planner", "controller"):
            if hasattr(self, name):
                delattr(self, name)

    def _make_trajectory_planner(self) -> EefTrajectoryPlanner:
        return EefTrajectoryPlanner(
            lambda: copy_sdk_vector(self.controller.get_joint_state(), "pos"),
            self.frames,
            self.ik,
            self.trajectory_hz,
            self.motion_limits,
            self.endpoint_hold_s,
        )

    def send_eef_trajectory(
        self,
        plan: dict[str, Any],
        wait: bool = True,
        start_time: float | None = None,
    ) -> None:
        """Send an already solved joint trajectory to this arm's SDK."""
        self._check_feedback()
        measured = self.controller.get_joint_state()
        joints = np.asarray(measured.pos(), dtype=np.float64).copy()
        plan.setdefault("result", {}).setdefault("_trace", {})["execution_start"] = {
            "joint_positions_rad": joints.tolist(),
            "timestamp_s": float(measured.timestamp),
            "timestamp_source": "arx_joint_state_controller_time",
        }
        commands = plan["commands"]
        relative_times = plan["relative_times_s"]
        coordinated_start = start_time is not None
        if start_time is None:
            start_time = self.controller.get_timestamp()
        for relative_time, command in zip(relative_times, commands):
            command.timestamp = start_time + relative_time
        submitted = commands
        if coordinated_start:
            start_hold = self.sdk.JointState(self.dof)
            start_hold.pos()[:] = plan["start_joint_positions_rad"]
            start_hold.vel()[:] = 0.0
            start_hold.torque()[:] = 0.0
            start_hold.gripper_pos = self._normalized_to_meters(
                plan["start_gripper_normalized"]
            )
            start_hold.timestamp = start_time
            submitted = [start_hold, *commands]
        self.motion.send(self.controller.set_joint_traj, submitted)
        if commands:
            self._arm_target = np.asarray(commands[-1].pos(), dtype=np.float64).copy()
            self._reference_end_s = float(commands[-1].timestamp)
        if wait:
            self.wait_reference_complete()

    def wait_reference_complete(self) -> None:
        """Finish the reference with zero velocity, then observe actual settling."""
        self.motion.check()
        while self.controller.get_timestamp() < self._reference_end_s:
            self.motion.check()
            self._check_feedback()
            time.sleep(0.01)

        self._check_feedback()

        # set_joint_traj() estimates endpoint velocity by backward difference.
        # Switching to a position hold explicitly clears any remaining feed-forward
        # velocity, including when an older SDK wheel ignores supplied endpoint qd.
        if self._arm_target is not None:
            self._send_command(self._target_command())

        started = time.monotonic()
        consecutive = 0
        max_position_error = float("inf")
        max_velocity = float("inf")
        while time.monotonic() - started < self.settle_timeout_s:
            self.motion.check()
            self._check_feedback()
            measured = self.controller.get_joint_state()
            actual = np.asarray(measured.pos(), dtype=np.float64)
            velocity = np.asarray(measured.vel(), dtype=np.float64)
            max_position_error = float(np.max(np.abs(actual - self._arm_target)))
            max_velocity = float(np.max(np.abs(velocity)))
            if (
                max_position_error <= self.settle_position_tolerance_rad
                and max_velocity <= self.settle_velocity_tolerance_rad_s
            ):
                consecutive += 1
                if consecutive >= self.settle_samples:
                    break
            else:
                consecutive = 0
            time.sleep(1.0 / self.trajectory_hz)
        self._last_settle_report = {
            "settled": consecutive >= self.settle_samples,
            "observed_s": time.monotonic() - started,
            "consecutive_samples": consecutive,
            "max_position_error_rad": max_position_error,
            "max_velocity_rad_s": max_velocity,
        }

    def _target_command(self) -> Any:
        command = self._command_copy()
        if self._arm_target is not None:
            command.pos()[:] = self._arm_target
            command.vel()[:] = 0.0
        return command

    def _hold_current_pose(self) -> None:
        """Arm the SDK's native position loop at the measured pose."""
        command = self._command_copy()
        self._send_command(command)
        self._arm_target = np.asarray(command.pos(), dtype=np.float64).copy()
        self._gripper_target = self._gripper_to_normalized(float(command.gripper_pos))
        self._reference_end_s = self.controller.get_timestamp()
        config = self.controller.get_controller_config()
        gain = self.sdk.Gain(
            np.asarray(config.default_kp, dtype=np.float64),
            np.asarray(config.default_kd, dtype=np.float64),
            float(config.default_gripper_kp),
            float(config.default_gripper_kd),
        )
        self.motion.send(self.controller.set_gain, gain)

    def _move_joints(self, arguments: dict[str, Any]) -> None:
        command = self._command_copy()
        command.pos()[:] = np.asarray(self._value(arguments, "positions"), dtype=np.float64)
        if arguments.get("velocities") is not None:
            command.vel()[:] = np.asarray(arguments["velocities"], dtype=np.float64)
        if arguments.get("torques") is not None:
            command.torque()[:] = np.asarray(arguments["torques"], dtype=np.float64)
        if arguments.get("gripper_position") is not None:
            command.gripper_pos = float(arguments["gripper_position"])
        self._send_command(command)

    def _move_eef(self, arguments: dict[str, Any]) -> None:
        pose = np.asarray(self._value(arguments, "pose_xyzrpy"), dtype=np.float64)
        current = self.controller.get_joint_state()
        status, positions = self.solver.multi_trial_ik(pose, current.pos(), 5)
        if status != 0:
            name = self.solver.get_ik_status_name(status)
            raise RuntimeError(f"逆运动学失败: {name} ({status})")
        command = self._command_copy()
        command.pos()[:] = positions
        if arguments.get("gripper_position") is not None:
            command.gripper_pos = float(arguments["gripper_position"])
        self._send_command(command)

    def _set_gain(self, arguments: dict[str, Any]) -> None:
        gain = self.sdk.Gain(
            np.asarray(self._value(arguments, "kp"), dtype=np.float64),
            np.asarray(self._value(arguments, "kd"), dtype=np.float64),
            float(arguments.get("gripper_kp") or 0.0),
            float(arguments.get("gripper_kd") or 0.0),
        )
        self.motion.send(self.controller.set_gain, gain)

    @staticmethod
    def _value(arguments: dict[str, Any], key: str) -> Any:
        value = arguments.get(key)
        if value is None:
            raise ValueError(f"动作缺少字段: {key}")
        return value
