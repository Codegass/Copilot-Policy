"""Calibrated RGB ray projection and two-view triangulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry.frames import FrameCalibration


class PixelLocalizer:
    def __init__(self, settings: dict[str, Any]) -> None:
        vision = settings.get("vision", {})
        self.intrinsics = {
            name: np.asarray(matrix, dtype=np.float64)
            for name, matrix in vision.get("camera_intrinsics", {}).items()
        }
        self.distortion = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in vision.get("distortion_coefficients", {}).items()
        }
        self.frames = FrameCalibration(settings)
        self.min_triangulation_angle_deg = float(
            vision.get("min_triangulation_angle_deg", 5.0)
        )
        self.max_triangulation_condition = float(
            vision.get("max_triangulation_condition", 500.0)
        )
        self.max_triangulation_residual_m = float(
            vision.get("max_triangulation_residual_m", 0.01)
        )
        quality_limits = (
            self.min_triangulation_angle_deg,
            self.max_triangulation_condition,
            self.max_triangulation_residual_m,
        )
        if not all(np.isfinite(value) and value > 0 for value in quality_limits):
            raise ValueError("triangulation quality limits must be finite and positive")

    def locate(
        self,
        arguments: dict[str, Any],
        state: dict[str, Any],
        history: dict[int, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        name = str(arguments.get("camera", ""))
        pixel = np.asarray(arguments.get("pixel_xy", []), dtype=np.float64)
        ray = self._camera_ray(name, pixel)
        result: dict[str, Any] = {
            "camera": name,
            "pixel_xy": pixel.tolist(),
            "ray_camera_xyz": ray.tolist(),
            "metric_position_available": False,
            "explanation": "单张 RGB 图像只确定相机射线。",
        }
        fixed_rays = self._fixed_base_rays(name, ray)
        if fixed_rays:
            result.update({
                "base_frame_rays": fixed_rays,
                "explanation": "已使用顶部相机固定外参转换到左右机械臂 base 坐标系。",
            })
            return result
        base_ray = self._base_ray(name, pixel, state)
        if base_ray is None:
            return result
        camera_origin, direction = base_ray
        result.update({
            "ray_origin_base_xyz": camera_origin.tolist(),
            "ray_direction_base_xyz": direction.tolist(),
            "explanation": "已使用项目内标定转换到机械臂 base 坐标系。",
        })
        reference_step = arguments.get("reference_step")
        reference_pixel = arguments.get("reference_pixel_xy")
        if reference_step is None or reference_pixel is None or history is None:
            return result
        reference_state = history.get(int(reference_step))
        if reference_state is None:
            raise ValueError(f"找不到 step {reference_step} 的历史状态")
        reference_ray = self._base_ray(name, np.asarray(reference_pixel), reference_state)
        if reference_ray is None:
            return result
        point, residual, quality = _triangulate(
            camera_origin, direction, *reference_ray
        )
        reasons = []
        if quality["parallax_angle_deg"] < self.min_triangulation_angle_deg:
            reasons.append("parallax_angle_too_small")
        if quality["condition_number"] > self.max_triangulation_condition:
            reasons.append("triangulation_ill_conditioned")
        if residual > self.max_triangulation_residual_m:
            reasons.append("ray_residual_too_large")
        if any(depth <= 0 for depth in quality["ray_depths_m"]):
            reasons.append("intersection_behind_camera")
        valid = not reasons
        result.update({
            "triangulation_candidate_base_xyz": point.tolist(),
            "triangulation_residual_m": residual,
            "triangulation_valid": valid,
            "triangulation_rejection_reasons": reasons,
            **quality,
            "reference_step": int(reference_step),
        })
        if valid:
            result.update({
                "metric_position_base_xyz": point.tolist(),
                "metric_position_available": True,
                "explanation": "两次腕部 RGB 视线交会通过视差和数值条件检查。",
            })
        else:
            result["explanation"] = (
                "两次腕部 RGB 视线交会退化；候选点仅供诊断，不作为米制定位。"
            )
        return result

    def _camera_ray(self, name: str, pixel: np.ndarray) -> np.ndarray:
        matrix = self.intrinsics.get(name)
        if matrix is None or matrix.shape != (3, 3):
            raise ValueError(f"没有 {name} 相机的有效内参")
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("pixel_xy 必须是有限的 [x, y]")
        distorted = np.array([
            (pixel[0] - matrix[0, 2]) / matrix[0, 0],
            (pixel[1] - matrix[1, 2]) / matrix[1, 1],
        ])
        normalized = _undistort(distorted, self.distortion.get(name))
        ray = np.r_[normalized, 1.0]
        return ray / np.linalg.norm(ray)

    def _base_ray(
        self, name: str, pixel: np.ndarray, state: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        arm = "left" if name == "left" else "right" if name == "right" else None
        if arm is None:
            return None
        arm_state = state["arms"][arm] if "arms" in state else state
        transform = self.frames.base_from_camera(name, arm_state["tcp_xyzrpy"])
        if transform is None:
            return None
        direction = transform[:3, :3] @ self._camera_ray(name, np.asarray(pixel))
        return transform[:3, 3], direction / np.linalg.norm(direction)

    def _fixed_base_rays(
        self, name: str, camera_ray: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        result = {}
        for arm in self.frames.fixed_camera_arms(name):
            transform = self.frames.fixed_base_from_camera(name, arm)
            if transform is None:
                continue
            direction = transform[:3, :3] @ camera_ray
            result[arm] = {
                "frame": f"{arm}_base_link",
                "ray_origin_base_xyz": transform[:3, 3].tolist(),
                "ray_direction_base_xyz": (
                    direction / np.linalg.norm(direction)
                ).tolist(),
            }
        return result


def _undistort(point: np.ndarray, coefficients: np.ndarray | None) -> np.ndarray:
    if coefficients is None or coefficients.shape != (5,):
        return point
    k1, k2, p1, p2, k3 = coefficients
    x, y = point
    for _ in range(8):
        radius = x * x + y * y
        radial = 1 + k1 * radius + k2 * radius**2 + k3 * radius**3
        dx = 2 * p1 * x * y + p2 * (radius + 2 * x * x)
        dy = p1 * (radius + 2 * y * y) + 2 * p2 * x * y
        x, y = (point[0] - dx) / radial, (point[1] - dy) / radial
    return np.array([x, y])


def _triangulate(
    origin_a: np.ndarray,
    direction_a: np.ndarray,
    origin_b: np.ndarray,
    direction_b: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    directions = [direction_a / np.linalg.norm(direction_a), direction_b / np.linalg.norm(direction_b)]
    matrix = sum(np.eye(3) - np.outer(direction, direction) for direction in directions)
    vector = sum(
        (np.eye(3) - np.outer(direction, direction)) @ origin
        for direction, origin in zip(directions, (origin_a, origin_b))
    )
    point = np.linalg.lstsq(matrix, vector, rcond=None)[0]
    residual = max(
        np.linalg.norm(np.cross(point - origin, direction))
        for origin, direction in zip((origin_a, origin_b), directions)
    )
    cosine = float(np.clip(np.dot(directions[0], directions[1]), -1.0, 1.0))
    quality = {
        "camera_baseline_m": float(np.linalg.norm(origin_a - origin_b)),
        "parallax_angle_deg": float(np.degrees(np.arccos(cosine))),
        "condition_number": float(np.linalg.cond(matrix)),
        "ray_depths_m": [
            float(np.dot(point - origin, direction))
            for origin, direction in zip((origin_a, origin_b), directions)
        ],
    }
    return point, float(residual), quality
