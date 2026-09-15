"""Project-local robot TCP and camera frame calibration."""

from __future__ import annotations

from typing import Any

import numpy as np

from .poses import matrix_to_pose, pose_to_matrix


class FrameCalibration:
    """Convert SDK eef_link poses to the calibrated fingertip TCP frame."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        values = (settings or {}).get("calibration", {})
        link6_from_sdk = _transform(values.get("link6_from_sdk_eef"), "link6_from_sdk_eef")
        link6_from_tcp = _transform(values.get("link6_from_tcp"), "link6_from_tcp")
        self.sdk_eef_from_tcp = np.linalg.inv(link6_from_sdk) @ link6_from_tcp
        cameras = values.get("link6_from_camera", {})
        self.tcp_from_camera = {
            name: np.linalg.inv(link6_from_tcp) @ _transform(matrix, f"{name} camera")
            for name, matrix in cameras.items()
            if matrix is not None
        }
        fixed = values.get("base_from_camera", {})
        self.fixed_camera_transforms = {
            camera: {
                arm: _transform(matrix, f"{arm} base from {camera} camera")
                for arm, matrix in transforms.items()
            }
            for camera, transforms in fixed.items()
            if isinstance(transforms, dict)
        }

    def sdk_to_tcp(self, sdk_pose_xyzrpy: object) -> np.ndarray:
        return matrix_to_pose(pose_to_matrix(sdk_pose_xyzrpy) @ self.sdk_eef_from_tcp)

    def tcp_to_sdk(self, tcp_pose_xyzrpy: object) -> np.ndarray:
        return matrix_to_pose(pose_to_matrix(tcp_pose_xyzrpy) @ np.linalg.inv(self.sdk_eef_from_tcp))

    def base_from_camera(self, camera: str, tcp_pose_xyzrpy: object) -> np.ndarray | None:
        transform = self.tcp_from_camera.get(camera)
        return None if transform is None else pose_to_matrix(tcp_pose_xyzrpy) @ transform

    def fixed_base_from_camera(self, camera: str, arm: str) -> np.ndarray | None:
        transform = self.fixed_camera_transforms.get(camera, {}).get(arm)
        return None if transform is None else transform.copy()

    def fixed_camera_arms(self, camera: str) -> tuple[str, ...]:
        return tuple(self.fixed_camera_transforms.get(camera, {}))


def _transform(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"calibration {name} must be a finite 4x4 matrix")
    return matrix
