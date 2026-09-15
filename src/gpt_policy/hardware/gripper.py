"""Measured-position streaming for the ARX gripper."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any


class GripperStreamer:
    """Advance a gripper goal without accumulating error against an object."""

    def __init__(
        self,
        controller: Any,
        command_factory: Callable[[], Any],
        width_m: float,
        open_readout_rad: float,
        velocity_rad_s: float,
        control_hz: float,
        sdk_torque_max_nm: float,
        send_command: Callable[[Any], None] | None = None,
    ) -> None:
        values = (
            width_m,
            abs(open_readout_rad),
            velocity_rad_s,
            control_hz,
            sdk_torque_max_nm,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("gripper calibration and streaming rate must be positive")
        self.controller = controller
        self.command_factory = command_factory
        self.send_command = send_command or controller.set_joint_cmd
        self.width_m = float(width_m)
        self.open_readout_rad = float(open_readout_rad)
        self.velocity_rad_s = float(velocity_rad_s)
        self.control_hz = float(control_hz)
        self.sdk_torque_max_nm = float(sdk_torque_max_nm)
        self.step_m = (
            self.velocity_rad_s / abs(self.open_readout_rad)
            * self.width_m / self.control_hz
        )

    @property
    def velocity_m_s(self) -> float:
        return self.step_m * self.control_hz

    def move(self, requested_normalized: float) -> dict[str, float | int | str | bool]:
        """Stream a finite reference and rebase it when the SDK reports blockage."""
        goal_m = float(requested_normalized) * self.width_m
        start_m = float(self.controller.get_joint_state().gripper_pos)
        delta_m = goal_m - start_m
        direction = 1.0 if delta_m >= 0 else -1.0
        samples = max(1, math.ceil(abs(delta_m) / self.step_m))
        active_m = start_m
        sent_samples = 0
        sdk_contact = False
        next_tick = time.monotonic()

        for index in range(samples):
            state = self.controller.get_joint_state()
            measured_m = float(state.gripper_pos)
            remaining_m = goal_m - measured_m
            if direction * remaining_m <= 0:
                active_m = goal_m
            elif self._sdk_reports_blocked(float(state.gripper_torque), direction):
                # Remove the accumulated unreachable reference while retaining
                # one streamed step of preload against the contacted object.
                active_m = measured_m + direction * min(
                    self.step_m, abs(remaining_m)
                )
                sdk_contact = True
            else:
                # Advance from the planned reference, not from delayed encoder
                # feedback. Otherwise a real gripper never gets beyond its
                # first small step.
                active_m = start_m + direction * min(
                    (index + 1) * self.step_m, abs(delta_m)
                )
            self._send(active_m)
            sent_samples += 1
            if sdk_contact:
                break

            next_tick += 1.0 / self.control_hz
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)

        measured_m = float(self.controller.get_joint_state().gripper_pos)
        return {
            "mode": "measured_position_stream",
            "requested_normalized": float(requested_normalized),
            "active_command_normalized": active_m / self.width_m,
            "measured_normalized": measured_m / self.width_m,
            "planned_samples": samples,
            "sent_samples": sent_samples,
            "sdk_contact": sdk_contact,
            "control_hz": self.control_hz,
            "velocity_native_rad_s": self.velocity_rad_s,
            "velocity_m_s": self.velocity_m_s,
        }

    def _sdk_reports_blocked(self, torque_nm: float, direction: float) -> bool:
        """Mirror the vendor SDK's existing blocked-direction decision."""
        if abs(torque_nm) <= self.sdk_torque_max_nm / 2:
            return False
        torque_direction = 1.0 if torque_nm * self.open_readout_rad > 0 else -1.0
        return direction * torque_direction > 0

    def _send(self, position_m: float) -> None:
        command = self.command_factory()
        command.gripper_pos = position_m
        command.gripper_vel = 0.0
        command.gripper_torque = 0.0
        command.timestamp = 0.0
        self.send_command(command)
