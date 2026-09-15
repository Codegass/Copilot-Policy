"""Measured motion diagnostics shared by the hardware adapters."""

from __future__ import annotations

from typing import Any

import numpy as np


def motion_progress(
    start: dict[str, Any],
    target_tcp: np.ndarray,
    measured_tcp: np.ndarray,
    end_timestamp_s: float,
) -> dict[str, Any]:
    """Report net translation to the final target, never chunk path length.

    Start and end are measured snapshots in one arm's timestamp domain. These
    numbers do not classify contact, stability, successful insertion or arrival.
    """
    origin = np.asarray(start["tcp_xyzrpy"], dtype=np.float64)[:3]
    target = np.asarray(target_tcp, dtype=np.float64)[:3]
    actual = np.asarray(measured_tcp, dtype=np.float64)[:3]
    return {
        "requested_delta_xyz_m": (target - origin).tolist(),
        "achieved_delta_xyz_m": (actual - origin).tolist(),
        "remaining_delta_xyz_m": (target - actual).tolist(),
        "start_timestamp_s": float(start["timestamp_s"]),
        "end_timestamp_s": float(end_timestamp_s),
        "timestamp_source": start["timestamp_source"],
    }
