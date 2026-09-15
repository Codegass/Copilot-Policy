"""Official I2RT/Mink IK with an explicit six-arm-joint model mapping."""

from __future__ import annotations

import threading

import numpy as np

from ..geometry.poses import matrix_to_pose, pose_to_matrix, rotation_distance
from ..motion.coordination import TrajectoryIKError


class YamIK:
    name = "i2rt.Kinematics.ik (Mink/quadprog), verified by FK"

    def __init__(self, motion=None, gripper_type="LINEAR_4310"):
        from i2rt.robots.kinematics import Kinematics
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
        import mink

        motion = motion or {}
        self.kinematics = Kinematics(
            combine_arm_and_gripper_xml(ArmType.YAM, GripperType[gripper_type]), "grasp_site",
        )
        model = self.kinematics._configuration.model
        self.indices = np.array([model.joint(f"joint{i}").qposadr[0] for i in range(1, 7)])
        self.joint_ids = np.array([model.joint(f"joint{i}").id for i in range(1, 7)])
        if len(set(self.indices)) != 6 or model.nq != model.nv:
            raise ValueError("Unsupported YAM joint mapping")
        self.lower, self.upper = model.jnt_range[self.joint_ids].T.copy()
        # The model has 8 coordinates; the physical command has 6 arm joints
        # and ONE normalized gripper. Neither slider coordinate is a motor.
        self.template = model.qpos0.copy()
        self.extra_indices = sorted(set(range(model.nq)) - set(self.indices))
        for j in range(model.njnt):
            index = model.jnt_qposadr[j]
            if index in self.extra_indices and model.jnt_limited[j]:
                self.template[index] = model.jnt_range[j].mean()
        self.limits = [mink.ConfigurationLimit(model)]
        self.execution_translation_tolerance_m = float(motion.get("ik_execution_translation_tolerance_m", 0.002))
        self.execution_rotation_tolerance_rad = float(motion.get("ik_execution_rotation_tolerance_rad", 0.0174532925))
        self.numeric_position = float(motion.get("ik_translation_tolerance_m", 1e-4))
        self.numeric_rotation = float(motion.get("ik_rotation_tolerance_rad", 5e-4))
        self.lock = threading.RLock()
        self.solver = self  # The common planner's FK interface.

    def configure_joint_limits(self, bounds):
        """Use the initialized SDK's actual arm limits, including its encoder margin."""
        import mink
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (6, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Expected six valid I2RT arm joint limits")
        with self.lock:
            model = self.kinematics._configuration.model
            model.jnt_range[self.joint_ids] = bounds
            self.lower, self.upper = bounds.T.copy()
            self.limits = [mink.ConfigurationLimit(model)]

    def _full(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("YAM arm joints must be a finite six-vector")
        full = self.template.copy()
        full[self.indices] = joints
        return full

    def forward_kinematics(self, joints):
        with self.lock:
            return matrix_to_pose(self.kinematics.fk(self._full(joints)).copy())

    def error(self, target, joints):
        actual = self.forward_kinematics(joints)
        return float(np.linalg.norm(target[:3] - actual[:3])), rotation_distance(target[3:], actual[3:])

    def solve(self, target, seed, segment, sample):
        with self.lock:
            success, full = self.kinematics.ik(
                pose_to_matrix(target), "grasp_site", init_q=self._full(seed),
                limits=self.limits, pos_threshold=self.numeric_position,
                ori_threshold=self.numeric_rotation, max_iters=200,
            )
            joints = np.asarray(full)[self.indices].copy()
            finite = np.isfinite(joints).all()
            translation, rotation = self.error(target, joints) if finite else (1e30, 1e30)
            within_limits = finite and np.all(joints >= self.lower - 1e-7) and np.all(joints <= self.upper + 1e-7)
            if within_limits and translation <= self.execution_translation_tolerance_m and rotation <= self.execution_rotation_tolerance_rad:
                return joints
            raise TrajectoryIKError(
                segment=segment, sample=sample, status=0 if success else -1,
                status_name="MINK_CONVERGED" if success else "MINK_NOT_CONVERGED",
                translation_error_m=translation, rotation_error_rad=rotation,
                reason="ik_residual_above_tolerance" if within_limits else "ik_joint_limit",
            )
