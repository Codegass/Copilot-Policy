"""Thin YAM adapter: official I2RT control, shared TCP planning and recording."""

from __future__ import annotations

import time
import threading
import traceback
from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor

import numpy as np
from ruckig import InputParameter, Result, Ruckig, Trajectory

from ..geometry.frames import FrameCalibration
from ..geometry.poses import rotation_distance, rpy_to_quaternion
from ..motion.planner import EefTrajectoryPlanner
from ..motion.trajectory import MotionLimits
from .yam_ik import YamIK
from .yam_driver import close_driver
from .feedback import motion_progress
from .motion_control import MotionControl


_THREAD_FAILURES = {}
_THREAD_FAILURES_LOCK = threading.Lock()
_THREAD_HOOK_INSTALLED = False
_PREVIOUS_THREAD_HOOK = None


def _install_thread_failure_hook():
    global _THREAD_HOOK_INSTALLED, _PREVIOUS_THREAD_HOOK
    if _THREAD_HOOK_INSTALLED:
        return
    _PREVIOUS_THREAD_HOOK = threading.excepthook

    def hook(args):
        thread = args.thread
        with _THREAD_FAILURES_LOCK:
            _THREAD_FAILURES[getattr(thread, "ident", None)] = {
                "thread": getattr(thread, "name", None),
                "timestamp": time.monotonic(),
                "exception": "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)).strip(),
            }
            if len(_THREAD_FAILURES) > 32:
                oldest = min(_THREAD_FAILURES, key=lambda key: _THREAD_FAILURES[key]["timestamp"])
                del _THREAD_FAILURES[oldest]
        _PREVIOUS_THREAD_HOOK(args)

    threading.excepthook = hook
    _THREAD_HOOK_INSTALLED = True


class YamRobot:
    dof = 6

    def __init__(self, interface, settings, side="left", *, driver=None, ik=None):
        self.interface, self.side = interface, side
        self._created_at = time.monotonic()
        robot = settings.get("robot", {})
        motion = settings.get("motion", {})
        self.frames = FrameCalibration(settings)
        self.ik = ik or YamIK(motion, robot.get("gripper_type", "LINEAR_4310"))
        self.hz = float(settings.get("runtime", {}).get("trajectory_hz", 100))
        self.width = float(robot.get("gripper_width_m", 0.088))
        self.gripper_speed = float(robot.get("gripper_speed_normalized_s", 0.5))
        self.temperature_limit = float(robot.get("motor_temp_limit_c", 80))
        self.settle_timeout = float(motion.get("settle_timeout_s", 3))
        self.settle_position = float(motion.get("settle_position_tolerance_rad", 0.03))
        self.settle_velocity = float(motion.get("settle_velocity_tolerance_rad_s", 0.05))
        self.settle_samples = int(motion.get("settle_samples", 10))
        self.settle_window = float(motion.get("settle_window_s", 0.3))
        self.settle_position_span = float(motion.get("settle_position_span_rad", 0.002))
        self.max_lateness = float(motion.get("max_command_lateness_s", 0.25))
        self.home_time_scale = float(motion.get("home_time_scale", 3.0))
        positive = (self.hz, self.width, self.gripper_speed, self.temperature_limit,
                    self.settle_timeout, self.settle_position, self.settle_velocity,
                    self.settle_window, self.settle_position_span, self.max_lateness, self.home_time_scale)
        if not np.isfinite(positive).all() or min(positive) <= 0 or self.settle_samples < 1:
            raise ValueError("YAM control and settling settings must be finite and positive")
        limits = MotionLimits(
            output_hz=self.hz,
            cartesian_step_m=float(motion.get("cartesian_step_m", 0.005)),
            cartesian_step_rad=float(motion.get("cartesian_step_rad", 0.035)),
            tcp_velocity_m_s=float(motion.get("tcp_velocity_m_s", 0.08)),
            tcp_angular_velocity_rad_s=float(motion.get("tcp_angular_velocity_rad_s", 0.5)),
            joint_velocity_rad_s=np.broadcast_to(motion.get("joint_velocity_rad_s", 0.6), (6,)).copy(),
            joint_acceleration_rad_s2=np.broadcast_to(motion.get("joint_acceleration_rad_s2", 2.0), (6,)).copy(),
            joint_jerk_rad_s3=np.broadcast_to(motion.get("joint_jerk_rad_s3", 12.0), (6,)).copy(),
        )
        self.planner = EefTrajectoryPlanner(
            self._positions, self.frames, self.ik, self.hz, limits,
            float(motion.get("endpoint_hold_s", 0.12)),
        )
        self.motion = MotionControl()
        self._stop = self.motion.stopped
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"yam-{side}")
        self._pending = None
        self._closed = False
        self._driver_closed = False
        self._last_settle_report = {}
        self.driver = driver
        try:
            if self.driver is None:
                from i2rt.robots.get_robot import get_yam_robot, GripperType

                _install_thread_failure_hook()
                gripper_limits = robot.get("gripper_limits", {}).get(side)
                self.driver = get_yam_robot(
                    channel=interface, gripper_type=GripperType[robot.get("gripper_type", "LINEAR_4310")],
                    zero_gravity_mode=False,
                    gripper_limits_override=None if gripper_limits is None else np.asarray(gripper_limits),
                    enable_auto_recovery=False,
                )
            initial = self._feedback()
            if hasattr(self.ik, "configure_joint_limits"):
                bounds = np.asarray(self.driver.get_robot_info()["joint_limits"])
                self.ik.configure_joint_limits(bounds[:6])
            self._server_thread_id = getattr(self.driver._server_thread, "ident", None)
            self._home = initial["q"][:6].copy()
            self._target = self._home.copy()
            self._gripper_target = float(initial["q"][6])
        except BaseException:
            self.close()
            raise

    def clock(self):
        return time.monotonic()

    def _feedback(self):
        if not self.driver.motor_chain.running or not self.driver._server_thread.is_alive():
            running = bool(self.driver.motor_chain.running)
            server_alive = bool(self.driver._server_thread.is_alive())
            with _THREAD_FAILURES_LOCK:
                failures = list(_THREAD_FAILURES.items())
            own_server_id = getattr(self, "_server_thread_id", None)
            matching = [
                failure for thread_id, failure in failures
                if failure.get("timestamp", 0) >= self._created_at
                and (thread_id == own_server_id or f"channel={self.interface}" in failure["exception"])
            ]
            motor_failures = [
                failure for failure in matching
                if "Motor error detected:" in failure["exception"]
                or "motor errors detected:" in failure["exception"]
                or "fail to communicate with the motor" in failure["exception"]
            ]
            failure = max(motor_failures or matching, key=lambda item: item["timestamp"], default=None)
            detail = ""
            if failure is not None:
                summary = failure["exception"].splitlines()[-1].strip()
                detail = f"; control_exception={summary}"
            raise RuntimeError(
                f"I2RT control loop stopped on {self.side} arm "
                f"(motor_chain.running={running}, server_thread_alive={server_alive}){detail}"
            )
        # Public get_observations() omits the source timestamp. Copy the pinned
        # SDK's state under its own lock so q/dq/effort/time are one sample.
        with self.driver._state_lock:
            s = self.driver._joint_state
            result = {"q": np.asarray(s.pos).copy(), "dq": np.asarray(s.vel).copy(),
                      "effort": np.asarray(s.eff).copy(), "timestamp": float(s.timestamp),
                      "temp_mos": np.asarray(s.temp_mos).copy(), "temp_rotor": np.asarray(s.temp_rotor).copy()}
        if any(result[k].shape != (7,) or not np.isfinite(result[k]).all() for k in ("q", "dq", "effort")):
            raise RuntimeError("Invalid YAM feedback; expected six arm joints and one normalized gripper")
        if not np.isfinite(result["timestamp"]) or time.time() - result["timestamp"] > 1.0:
            raise RuntimeError("I2RT feedback has not updated for over one second")
        for key in ("temp_mos", "temp_rotor"):
            if result[key].shape != (7,) or not np.isfinite(result[key]).all():
                raise RuntimeError(f"Invalid YAM temperature feedback: {key}")
        over_limit = {}
        for key in ("temp_mos", "temp_rotor"):
            hot = np.flatnonzero(result[key] >= self.temperature_limit)
            if hot.size:
                over_limit[key] = [
                    {"motor_index": int(index), "value_c": float(result[key][index])}
                    for index in hot
                ]
        result["temperature_over_limit"] = {
            "over_limit": over_limit,
            "limit_c": self.temperature_limit,
        } if over_limit else None
        return result

    def _positions(self):
        return self._feedback()["q"][:6]

    def state(self):
        s = self._feedback()
        tcp = self.frames.sdk_to_tcp(self.ik.forward_kinematics(s["q"][:6]))
        return {
            "joint_positions_rad": s["q"][:6].tolist(),
            "joint_velocities_rad_s": s["dq"][:6].tolist(),
            "joint_torques_nm": s["effort"][:6].tolist(),
            "tcp_xyzrpy": tcp.tolist(), "tcp_xyzquat": [*tcp[:3], *rpy_to_quaternion(tcp[3:])],
            "gripper_normalized": float(s["q"][6]), "gripper_position_m": float(s["q"][6]) * self.width,
            "gripper_command_normalized": self._gripper_target,
            "gripper_velocity_m_s": float(s["dq"][6]) * self.width,
            "gripper_torque_nm": float(s["effort"][6]),
            "gravity_compensation": True,
            "timestamp_s": s["timestamp"], "timestamp_source": "i2rt_motor_state_snapshot_host_time",
            "temperature_mos_c": s["temp_mos"].tolist(), "temperature_rotor_c": s["temp_rotor"].tolist(),
            "temperature_limit_c": self.temperature_limit,
            "temperature_over_limit": s["temperature_over_limit"],
        }

    def plan_eef_trajectory(self, requested, note):
        return self.planner.plan(requested, note, self._gripper_target)

    def send_eef_trajectory(self, plan, wait=True, start_time=None):
        self.motion.check()
        if self._pending is not None and not self._pending.done():
            raise RuntimeError("Previous YAM trajectory is still running")
        start = self.clock() + 0.02 if start_time is None else start_time
        self._pending = self._pool.submit(self._stream, plan, start)
        if wait:
            self.wait_reference_complete()

    def _stream(self, plan, start):
        times = np.r_[0.0, plan["relative_times_s"]]
        positions = np.vstack([plan["start_joint_positions_rad"], plan["joint_positions_rad"]])
        duration = float(times[-1])
        gripper = float(plan["start_gripper_normalized"])
        # Submit at a fixed cadence. Shared retiming can have sparse waypoints;
        # writing each knot alone would make the I2RT command buffer jump.
        submitted = 0
        lateness = 0.0
        try:
            for t in np.r_[np.arange(0, duration, 1 / self.hz), duration]:
                if self._stop.wait(max(0.0, start + t - self.clock())):
                    raise InterruptedError("YAM trajectory cancelled")
                late = self.clock() - (start + t)
                measured = self._feedback()
                if submitted == 0:
                    # Use the first actual execution sample, including for a
                    # scheduled start; the planning seed may already be old.
                    # Defer FK until after the stream so its lock cannot delay
                    # a command in the fixed-cadence loop.
                    plan["result"].setdefault("_trace", {})["execution_start"] = {
                        "joint_positions_rad": measured["q"][:6].tolist(),
                        "timestamp_s": measured["timestamp"],
                        "timestamp_source": "i2rt_motor_state_snapshot_host_time",
                    }
                q = np.array([np.interp(t, times, positions[:, j]) for j in range(6)])
                self.motion.send(self.driver.command_joint_pos, np.r_[q, gripper])
                submitted += 1
                lateness = max(lateness, late)
            self._target = positions[-1].copy()
            plan["result"]["submission"] = {
                "samples": submitted,
                "max_lateness_s": lateness,
                "lateness_limit_s": self.max_lateness,
                "lateness_policy": "record_only",
                "clock": "host_monotonic",
                "receipt": "SDK command buffer; measured settling reported separately",
            }
            snapshot = plan["result"]["_trace"]["execution_start"]
            snapshot["tcp_xyzrpy"] = self.frames.sdk_to_tcp(self.ik.forward_kinematics(
                np.asarray(snapshot["joint_positions_rad"])
            )).tolist()
        except BaseException:
            # Stop advancing on cancellation/fault; avoid continuing a partially
            # submitted path in the background after the caller sees an error.
            try:
                with self.motion.lock:
                    self._hold()
            except Exception:
                pass
            raise

    def wait_reference_complete(self):
        self.motion.check()
        if self._pending is not None:
            self._pending.result()
        deadline = self.clock() + self.settle_timeout
        samples = deque()
        required_samples = max(2, self.settle_samples)
        required_window = max(self.settle_window, (required_samples - 1) / self.hz)
        last_stamp = None
        started = self.clock()
        while True:
            s = self._feedback()
            error = float(np.max(np.abs(s["q"][:6] - self._target)))
            velocity = float(np.max(np.abs(s["dq"][:6])))
            # I2RT exposes independently reported motor velocity, whose wrist
            # noise can stay above the speed limit even with stationary encoder
            # positions. Settle from a continuous encoder-position window;
            # keep raw/RMS motor velocity as diagnostics, not a second veto.
            if last_stamp is not None and s["timestamp"] < last_stamp:
                raise RuntimeError("I2RT feedback timestamp moved backwards")
            if s["timestamp"] != last_stamp:
                if last_stamp is not None and s["timestamp"] - last_stamp > max(0.1, 3 / self.hz):
                    samples.clear()  # A gap cannot count as stable observation.
                samples.append(s)
                last_stamp = s["timestamp"]
                # Keep one sample at/before the window boundary and enough
                # distinct packets. Repeated snapshots never advance it.
                while len(samples) > required_samples and samples[1]["timestamp"] <= last_stamp - required_window:
                    samples.popleft()
            duration = samples[-1]["timestamp"] - samples[0]["timestamp"]
            positions = np.array([row["q"][:6] for row in samples])
            speeds = np.array([row["dq"][:6] for row in samples])
            rms_speed = float(np.max(np.sqrt(np.mean(speeds ** 2, axis=0))))
            span = float(np.max(np.ptp(positions, axis=0)))
            window_error = float(np.max(np.abs(positions - self._target)))
            window_ready = len(samples) >= required_samples and duration >= required_window
            settled = (window_ready and window_error <= self.settle_position
                       and span <= self.settle_position_span
                       and span <= self.settle_velocity * duration)
            if settled or self.clock() >= deadline:
                self._last_settle_report = {
                    "settled": bool(settled), "max_joint_residual_rad": error,
                    "max_joint_velocity_rad_s": velocity,
                    "method": "encoder_position_window",
                    "samples": len(samples), "required_samples": required_samples,
                    "required_window_s": required_window,
                    "window_s": duration, "elapsed_s": self.clock() - started,
                    "target_joint_positions_rad": self._target.tolist(),
                    "position_tolerance_rad": self.settle_position,
                    "position_span_tolerance_rad": self.settle_position_span,
                    "max_window_joint_residual_rad": window_error,
                    "max_joint_velocity_rms_rad_s": rms_speed,
                    "max_joint_position_span_rad": span,
                    "max_encoder_span_rate_rad_s": span / duration if duration > 0 else None,
                }
                return
            if self._stop.wait(1 / self.hz):
                raise InterruptedError("YAM settling cancelled")

    def _execution_feedback(self, plan, measured):
        target = np.asarray(plan["result"]["_trace"]["tcp_samples_xyzrpy"][-1])
        actual = np.asarray(measured["tcp_xyzrpy"])
        return {"target_tcp_xyzrpy": target.tolist(), "measured_tcp_xyzrpy": actual.tolist(),
                "max_joint_residual_rad": float(np.max(np.abs(plan["joint_positions_rad"][-1] - measured["joint_positions_rad"]))),
                "tcp_translation_error_m": float(np.linalg.norm(target[:3] - actual[:3])),
                "tcp_rotation_error_rad": rotation_distance(target[3:], actual[3:]),
                "motion_progress": motion_progress(
                    plan["result"]["_trace"]["execution_start"], target, actual, measured["timestamp_s"]
                ),
                "settle": dict(self._last_settle_report)}

    def _set_gripper(self, position):
        if not np.isfinite(position) or not 0 <= position <= 1:
            raise ValueError("gripper must be in [0, 1]")
        self.wait_reference_complete()
        start = self._gripper_target
        duration = abs(position - start) / self.gripper_speed
        count = max(1, int(np.ceil(duration * self.hz)))
        for value in np.linspace(start, position, count + 1)[1:]:
            self._feedback()
            self.motion.send(self.driver.command_joint_pos, np.r_[self._target, value])
            self._gripper_target = float(value)
            if self._stop.wait(duration / count):
                raise InterruptedError("YAM gripper cancelled")
        return {"active_command_normalized": position, "duration_s": duration, "force_limiter": "official I2RT"}

    def _gripper_feedback(self, target, measured, motion):
        return {"gripper_target_normalized": target, "gripper_measured_normalized": measured["gripper_normalized"],
                "gripper_residual_normalized": abs(target - measured["gripper_normalized"]),
                "gripper_torque_nm": measured["gripper_torque_nm"], "motion": motion}

    def plan_return_home(self):
        """Return to known startup joints without inventing a Cartesian IK path."""
        self.motion.check()
        if self._pending is not None and not self._pending.done():
            raise RuntimeError("Previous YAM trajectory is still running")
        start, home = self._positions().copy(), self._home.copy()
        endpoints = np.asarray([start, home])
        if (endpoints.shape != (2, 6) or not np.isfinite(endpoints).all()
                or np.any(endpoints < self.ik.lower - 1e-7)
                or np.any(endpoints > self.ik.upper + 1e-7)):
            raise ValueError("YAM home endpoints exceed configured joint limits")
        # Interpolate actual encoder angles, without modulo wrapping. A straight
        # joint segment stays within the joint bounds of its two endpoints.
        count = max(2, int(np.ceil(np.max(np.abs(home - start)) / .01)))
        fractions = np.linspace(0, 1, count + 1)
        joints = start + fractions[:, None] * (home - start)
        joints[0], joints[-1] = start, home
        poses = np.asarray([self.frames.sdk_to_tcp(self.ik.forward_kinematics(q)) for q in joints])
        # Bound the sampled TCP rate along this curved path, rather than using
        # endpoint distance, which can miss a large intermediate sweep.
        translation = float(np.max(np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) / np.diff(fractions)))
        rotation = max(rotation_distance(a[3:], b[3:]) / ds
                       for a, b, ds in zip(poses, poses[1:], np.diff(fractions)))
        limits = self.planner.limits
        delta = np.abs(home - start)
        active = delta > 1e-9
        if np.any(active):
            velocity = [float(np.min(limits.joint_velocity_rad_s[active] / delta[active]))]
            if translation > 1e-9:
                velocity.append(limits.tcp_velocity_m_s / (translation * 1.02))
            if rotation > 1e-9:
                velocity.append(limits.tcp_angular_velocity_rad_s / (rotation * 1.02))
            inp = InputParameter(1)
            inp.current_position, inp.target_position = [0.0], [1.0]
            inp.current_velocity = inp.target_velocity = [0.0]
            inp.current_acceleration = inp.target_acceleration = [0.0]
            inp.max_velocity = [min(velocity)]
            inp.max_acceleration = [float(np.min(limits.joint_acceleration_rad_s2[active] / delta[active]))]
            inp.max_jerk = [float(np.min(limits.joint_jerk_rad_s3[active] / delta[active]))]
            curve = Trajectory(1)
            if Ruckig(1, 1 / self.hz).calculate(inp, curve) != Result.Working:
                raise RuntimeError("YAM home time parameterization failed")
            scale = max(1.0, self.home_time_scale)
            duration = curve.duration * scale
            times = np.r_[np.arange(1 / self.hz, duration, 1 / self.hz), duration]
            progress = np.array([curve.at_time(t / scale)[0][0] for t in times])
            commands = start + progress[:, None] * (home - start)
            commands[-1] = home
        else:
            times, commands = np.array([1 / self.hz]), home[None, :].copy()
        hold = max(2, int(np.ceil(self.planner.endpoint_hold_s * self.hz)))
        commands = np.vstack([commands, np.repeat(home[None, :], hold, axis=0)])
        times = np.r_[times, times[-1] + np.arange(1, hold + 1) / self.hz]
        return {"start_joint_positions_rad": start, "joint_positions_rad": commands,
                "relative_times_s": times.tolist(), "start_gripper_normalized": self._gripper_target,
                "result": {"source": "run_start_joints", "timing": "ruckig_scalar_path_parameterization",
                           "planned_duration_s": float(times[-1]), "gripper_during_motion": "held",
                           "target_joint_positions_rad": home.tolist(),
                           "_trace": {"joint_waypoints_rad": commands.tolist(), "relative_times_s": times.tolist()}}}

    def return_home(self, plan=None):
        plan = self.plan_return_home() if plan is None else plan
        self.send_eef_trajectory(plan)
        return {**self.state(), "home": {"source": "run_start_joints", "settle": self._last_settle_report,
                                        "trajectory": plan["result"]}}

    def execute(self, name, arguments):
        if name == "state":
            return self.state()
        if name == "set_gripper":
            motion = self._set_gripper(float(arguments["gripper"]))
            state = self.state()
            return {**state, "execution_feedback": self._gripper_feedback(arguments["gripper"], state, motion)}
        if name not in {"move_to", "move_eef_chunk", "check_path"}:
            raise ValueError(f"Unknown YAM tool: {name}")
        poses = [arguments["target"]] if name == "move_to" else arguments["poses"]
        plan = self.plan_eef_trajectory(poses, arguments["note"])
        if name == "check_path":
            from ..motion.coordination import path_check_result
            return path_check_result({"arm": plan})
        self.send_eef_trajectory(plan)
        state = self.state()
        return {**state, "trajectory": plan["result"], "execution_feedback": self._execution_feedback(plan, state)}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._pool.shutdown(wait=True, cancel_futures=True)
        if self.driver is not None:
            close_driver(self.driver)

    def cancel(self):
        self.motion.cancel(self._hold)

    def _hold(self):
        q = self._feedback()["q"]
        self.driver.command_joint_pos(q)
        self._target = q[:6].copy()
        self._gripper_target = float(q[6])

    def resume(self):
        # Drain the cancelled stream before allowing the home trajectory.
        if self._pending is not None:
            try:
                self._pending.result()
            except (InterruptedError, CancelledError):
                pass
            self._pending = None
        self.motion.resume()
