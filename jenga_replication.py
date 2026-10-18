"""PyBullet replication of the Jenga pipeline described in the AR523 presentation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data


@dataclass
class SimulationConfig:
    layers: int = 18
    blocks_per_layer: int = 3
    block_length: float = 0.075
    block_width: float = 0.025
    block_height: float = 0.015
    block_mass_kg: float = 0.0196
    block_friction_range: tuple[float, float] = (0.4, 0.4)
    tower_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_base: tuple[float, float, float] = (0.55, 0.0, 0.0)
    time_step: float = 1.0 / 240.0
    camera_width: int = 640
    camera_height: int = 480
    camera_fov: float = 60.0
    camera_lateral_offset: float = 0.06
    first_force_threshold: float = 0.32
    second_force_threshold: float = 0.18
    force_setpoint_n: float = 0.23
    force_control_gain_m_s_n: float = 0.75
    force_control_contact_n: float = 0.02
    force_control_max_correction_m_s: float = 0.06
    max_push_speed_m_s: float = 0.005
    force_control_hz: float = 240.0
    sensor_hz: float = 20.0
    servo_hz: float = 9.0
    position_tolerance: float = 0.007
    probe_width: float = 0.011
    approach_distance: float = 0.10
    probe_clearance: float = 0.025
    push_distance: float = 0.10
    max_servo_seconds: float = 8.0
    max_attempts_per_game: int = 60


@dataclass
class Block:
    block_id: int
    layer: int
    slot: int
    long_axis: str
    body_id: int
    status: str = "present"
    peak_force: float = 0.0
    attempts: int = 0


@dataclass
class Detection:
    pixel: tuple[int, int]
    world_point: tuple[float, float, float]
    bbox: tuple[int, int, int, int]
    pnp_pose: Optional[tuple[tuple[float, ...], tuple[float, ...]]] = None
    pnp_world_position: Optional[tuple[float, float, float]] = None


@dataclass
class AttemptResult:
    trial: int
    block_id: int
    layer: int
    detector: str
    controller: str
    threshold_n: float
    outcome: str
    peak_force_n: float
    alignment_error_mm: Optional[float]
    servo_time_s: float
    extracted: bool
    tower_collapsed: bool
    axial_displacement_mm: Optional[float] = None
    vertical_drop_mm: Optional[float] = None
    vision_tracking_misses: int = 0
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)
    force_profile: list[tuple[float, float]] = field(default_factory=list)


def trapezoid_profile(steps: int, acceleration_fraction: float = 0.2) -> np.ndarray:
    """Return a normalized trapezoidal velocity profile integrated over `steps`."""
    if steps < 2:
        return np.array([0.0, 1.0])
    accel_steps = max(1, int(steps * acceleration_fraction))
    velocities = np.ones(steps, dtype=float)
    velocities[:accel_steps] = np.linspace(0.0, 1.0, accel_steps, endpoint=False)
    velocities[-accel_steps:] = np.linspace(1.0, 0.0, accel_steps, endpoint=False)
    velocities[-1] = 0.0
    positions = np.cumsum(velocities)
    positions -= positions[0]
    if positions[-1] <= 0:
        return np.linspace(0.0, 1.0, steps)
    return positions / positions[-1]


def force_feedback_correction(
    measured_force_n: float,
    target_force_n: float,
    gain_m_s_n: float,
    max_correction_m_s: float,
) -> float:
    return float(np.clip(
        gain_m_s_n * (target_force_n - measured_force_n),
        -max_correction_m_s,
        max_correction_m_s,
    ))


class BlockSelectionPolicy:
    """Two-subspace policy with per-block memory and one extraction per layer."""

    def __init__(self, blocks: Sequence[Block], layers: int, seed: int = 0):
        self.blocks = list(blocks)
        self.layers = layers
        self.rng = random.Random(seed)
        self.split_layer = layers // 2
        self.phase = self.rng.randrange(2)
        self.preferred_layer: Optional[int] = None
        self.first_layer = 4
        self.deferred_blocks: set[int] = set()

    @property
    def force_threshold(self) -> float:
        return 0.32 if self.phase == 0 else 0.18

    def _eligible(
        self,
        phase: int,
        exclude_deferred: bool = True,
        visible_block_ids: Optional[set[int]] = None,
    ) -> list[Block]:
        eligible = [
            block for block in self.blocks
            if block.status == "present"
            and (
                visible_block_ids is None
                or block.block_id in visible_block_ids
            )
            and (not exclude_deferred or block.block_id not in self.deferred_blocks)
            and self.first_layer <= block.layer < self.layers - 2
            and (block.layer < self.split_layer) == (phase == 0)
            and not any(
                other.layer == block.layer and other.status == "extracted"
                for other in self.blocks
            )
        ]
        return eligible

    def next_block(
        self,
        visible_block_ids: Optional[set[int]] = None,
    ) -> Optional[Block]:
        eligible = self._eligible(
            self.phase, visible_block_ids=visible_block_ids
        )
        if not eligible:
            other_phase = 1 - self.phase
            eligible = self._eligible(
                other_phase, visible_block_ids=visible_block_ids
            )
            if eligible:
                self.phase = other_phase
                self.preferred_layer = None
        if not eligible:
            self.deferred_blocks.clear()
            eligible = self._eligible(
                self.phase,
                exclude_deferred=False,
                visible_block_ids=visible_block_ids,
            )
            if not eligible:
                other_phase = 1 - self.phase
                eligible = self._eligible(
                    other_phase,
                    exclude_deferred=False,
                    visible_block_ids=visible_block_ids,
                )
                if eligible:
                    self.phase = other_phase
                    self.preferred_layer = None
        if not eligible:
            return None
        preferred = [
            block for block in eligible if block.layer == self.preferred_layer
        ]
        if preferred:
            eligible = preferred
        layers = sorted({block.layer for block in eligible})
        chosen_layer = self.rng.choice(layers)
        in_layer = [block for block in eligible if block.layer == chosen_layer]
        in_layer.sort(key=lambda block: (abs(block.slot - 1), block.slot))
        self.preferred_layer = chosen_layer
        return in_layer[0]

    def observe(self, block: Block, outcome: str) -> None:
        if outcome == "extracted":
            block.status = "extracted"
            self.preferred_layer = None
        elif outcome == "immobile":
            block.status = "tested"
            self.preferred_layer = block.layer
        elif outcome == "error":
            block.attempts += 1
            self.deferred_blocks.add(block.block_id)
            self.preferred_layer = None
        else:
            raise ValueError(f"Unknown attempt outcome: {outcome}")


class Tower:
    def __init__(self, config: SimulationConfig, seed: int):
        self.config = config
        self.rng = random.Random(seed)
        self.blocks: list[Block] = []
        self._create()

    def _create(self) -> None:
        cfg = self.config
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf")
        p.changeDynamics(plane, -1, lateralFriction=0.7, spinningFriction=0.01)
        self.board_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.18, 0.18, 0.008]
            ),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.18, 0.18, 0.008],
                rgbaColor=[0.16, 0.17, 0.19, 1],
            ),
            basePosition=[cfg.tower_origin[0], cfg.tower_origin[1], -0.008],
        )
        half = [cfg.block_length / 2, cfg.block_width / 2, cfg.block_height / 2]
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        for layer in range(cfg.layers):
            long_axis = "x" if layer % 2 == 0 else "y"
            z = cfg.tower_origin[2] + cfg.block_height * (layer + 0.5)
            for slot in range(cfg.blocks_per_layer):
                if long_axis == "x":
                    offset = (slot - (cfg.blocks_per_layer - 1) / 2) * cfg.block_width
                    pos = [cfg.tower_origin[0], cfg.tower_origin[1] + offset, z]
                    orn = p.getQuaternionFromEuler([0, 0, 0])
                else:
                    offset = (slot - (cfg.blocks_per_layer - 1) / 2) * cfg.block_width
                    pos = [cfg.tower_origin[0] + offset, cfg.tower_origin[1], z]
                    orn = p.getQuaternionFromEuler([0, 0, math.pi / 2])
                shade = self.rng.uniform(0.55, 0.88)
                color = [
                    min(1.0, shade),
                    min(1.0, shade * self.rng.uniform(0.57, 0.78)),
                    min(1.0, shade * self.rng.uniform(0.30, 0.48)),
                    1.0,
                ]
                visual = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=half, rgbaColor=color
                )
                body = p.createMultiBody(
                    baseMass=cfg.block_mass_kg,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=pos,
                    baseOrientation=orn,
                )
                p.changeDynamics(
                    body, -1,
                    lateralFriction=self.rng.uniform(*cfg.block_friction_range),
                    spinningFriction=0.01,
                    rollingFriction=0.001,
                    restitution=0.0,
                    linearDamping=0.04,
                    angularDamping=0.04,
                )
                self.blocks.append(
                    Block(len(self.blocks), layer, slot, long_axis, body)
                )
        self.settle()

    def settle(self, steps: int = 240) -> None:
        for _ in range(steps):
            p.stepSimulation()

    def block_center(self, block: Block) -> np.ndarray:
        return np.asarray(p.getBasePositionAndOrientation(block.body_id)[0])

    def face_center_and_push(self, block: Block) -> tuple[np.ndarray, np.ndarray]:
        center = self.block_center(block)
        if block.long_axis == "x":
            push = np.array([-1.0, 0.0, 0.0])
            face = center + np.array([self.config.block_length / 2, 0.0, 0.0])
        else:
            push = np.array([0.0, 1.0, 0.0])
            face = center + np.array([0.0, -self.config.block_length / 2, 0.0])
        return face, push

    def has_collapsed(self) -> bool:
        cfg = self.config
        for block in self.blocks:
            if block.status == "extracted":
                continue
            pos, orn = p.getBasePositionAndOrientation(block.body_id)
            initial = np.array([
                cfg.tower_origin[0] + (
                    (block.slot - (cfg.blocks_per_layer - 1) / 2) * cfg.block_width
                    if block.long_axis == "y" else 0.0
                ),
                cfg.tower_origin[1] + (
                    (block.slot - (cfg.blocks_per_layer - 1) / 2) * cfg.block_width
                    if block.long_axis == "x" else 0.0
                ),
                cfg.tower_origin[2] + cfg.block_height * (block.layer + 0.5),
            ])
            if np.linalg.norm(np.asarray(pos)[:2] - initial[:2]) > 0.09:
                return True
            if pos[2] < cfg.tower_origin[2] + cfg.block_height * block.layer - 0.02:
                return True
            up = np.asarray(p.rotateVector(orn, [0, 0, 1]))
            if up[2] < 0.75:
                return True
        return False


class Vision:
    def __init__(self, config: SimulationConfig, tower: Tower, robot_id: int, eef_link: int):
        self.config = config
        self.tower = tower
        self.robot_id = robot_id
        self.eef_link = eef_link
        self.focal = config.camera_height / (
            2 * math.tan(math.radians(config.camera_fov) / 2)
        )
        self.camera_matrix = np.array([
            [self.focal, 0, config.camera_width / 2],
            [0, self.focal, config.camera_height / 2],
            [0, 0, 1],
        ], dtype=np.float64)
        self.distortion = np.zeros((5, 1), dtype=np.float64)

    def capture(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[float], Sequence[float]]:
        cfg = self.config
        state = p.getLinkState(
            self.robot_id, self.eef_link, computeForwardKinematics=True
        )
        ee_pos = np.asarray(state[4], dtype=float)
        ee_orn = state[5]
        forward = np.asarray(p.rotateVector(ee_orn, [0, 0, 1]), dtype=float)
        lateral = np.cross(forward, np.array([0.0, 0.0, 1.0]))
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        eye = ee_pos - forward * 0.085 + lateral * cfg.camera_lateral_offset
        eye += np.array([0.0, 0.0, 0.08])
        target = eye + forward * 0.6 - np.array([0.0, 0.0, 0.08])
        view = p.computeViewMatrix(eye, target, [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(
            cfg.camera_fov,
            cfg.camera_width / cfg.camera_height,
            0.01,
            2.0,
        )
        image = p.getCameraImage(
            cfg.camera_width,
            cfg.camera_height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        )
        rgba = np.reshape(image[2], (cfg.camera_height, cfg.camera_width, 4))
        depth = np.reshape(image[3], (cfg.camera_height, cfg.camera_width))
        segmentation = np.reshape(image[4], (cfg.camera_height, cfg.camera_width))
        return rgba[:, :, :3].copy(), depth, segmentation, view, projection

    def project(self, point: Sequence[float], view: Sequence[float],
                projection: Sequence[float]) -> Optional[tuple[int, int]]:
        view_matrix = np.asarray(view).reshape((4, 4), order="F")
        proj_matrix = np.asarray(projection).reshape((4, 4), order="F")
        clip = proj_matrix @ view_matrix @ np.array([*point, 1.0])
        if clip[3] <= 0:
            return None
        ndc = clip[:3] / clip[3]
        px = int(round((ndc[0] + 1) * self.config.camera_width / 2))
        py = int(round((1 - ndc[1]) * self.config.camera_height / 2))
        if not (0 <= px < self.config.camera_width and 0 <= py < self.config.camera_height):
            return None
        return px, py

    def unproject(self, px: int, py: int, depth: float,
                  view: Sequence[float], projection: Sequence[float]) -> np.ndarray:
        view_matrix = np.asarray(view).reshape((4, 4), order="F")
        proj_matrix = np.asarray(projection).reshape((4, 4), order="F")
        clip = np.array([
            2 * px / self.config.camera_width - 1,
            1 - 2 * py / self.config.camera_height,
            2 * depth - 1,
            1.0,
        ])
        world = np.linalg.inv(proj_matrix @ view_matrix) @ clip
        if abs(world[3]) < 1e-12:
            raise ValueError("Cannot unproject pixel at infinity")
        return world[:3] / world[3]

    def _pnp_world_position(
        self,
        pose: Optional[tuple[tuple[float, ...], tuple[float, ...]]],
        view: Sequence[float],
    ) -> Optional[tuple[float, float, float]]:
        if pose is None:
            return None
        translation = np.asarray(pose[1], dtype=float)
        view_matrix = np.asarray(view).reshape((4, 4), order="F")
        camera_to_world = np.linalg.inv(view_matrix)
        camera_point_gl = np.array([
            translation[0], -translation[1], -translation[2], 1.0
        ])
        world_point = camera_to_world @ camera_point_gl
        if abs(world_point[3]) < 1e-12:
            return None
        return tuple(world_point[:3] / world_point[3])

    def target_detection(
        self, block: Block, mode: str
    ) -> Optional[Detection]:
        image, depth, segmentation, view, projection = self.capture()
        face, _ = self.tower.face_center_and_push(block)
        expected = self.project(face, view, projection)
        if expected is None:
            return None
        if mode == "ground_truth":
            encoded_id = segmentation.astype(np.int64) & ((1 << 24) - 1)
            mask = encoded_id == block.body_id
            ys, xs = np.where(mask)
            if len(xs) < 4:
                return None
            distances = np.hypot(xs - expected[0], ys - expected[1])
            nearest = int(np.argmin(distances))
            if distances[nearest] > 30:
                return None
            px, py = int(xs[nearest]), int(ys[nearest])
            d = float(depth[py, px])
            x, y, w, h = cv2.boundingRect(np.column_stack((xs, ys)).astype(np.int32))
            pnp = self._pnp_from_bbox(x, y, w, h, block)
            world_point = self.unproject(px, py, d, view, projection)
            pnp_world = self._pnp_world_position(pnp, view)
            if pnp_world is not None and np.linalg.norm(
                np.asarray(pnp_world) - world_point
            ) > 0.03:
                pnp_world = None
            return Detection(
                (px, py), tuple(world_point),
                (x, y, w, h), pnp, pnp_world,
            )

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([3, 35, 35], dtype=np.uint8),
            np.array([32, 255, 255], dtype=np.uint8),
        )
        radius = 24
        left = max(0, expected[0] - radius)
        right = min(self.config.camera_width, expected[0] + radius + 1)
        top = max(0, expected[1] - radius)
        bottom = min(self.config.camera_height, expected[1] + radius + 1)
        roi = mask[top:bottom, left:right]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 8:
                continue
            distance = cv2.pointPolygonTest(
                contour, (float(expected[0] - left), float(expected[1] - top)), True
            )
            if distance < 0:
                points = contour.reshape(-1, 2)
                distance = -float(np.min(np.linalg.norm(
                    points - (np.asarray(expected) - np.array([left, top])), axis=1
                )))
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((distance, x, y, w, h, contour))
        if not candidates:
            return None
        distance, x, y, w, h, contour = max(candidates, key=lambda item: item[0])
        if distance < -radius:
            return None
        contour_mask = np.zeros(roi.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        ys, xs = np.where(contour_mask != 0)
        if not len(xs):
            return None
        local_expected = np.asarray([expected[0] - left, expected[1] - top])
        nearest = int(np.argmin(np.hypot(xs - local_expected[0], ys - local_expected[1])))
        px, py = int(xs[nearest] + left), int(ys[nearest] + top)
        d = float(depth[py, px])
        if d >= 1.0:
            return None
        x += left
        y += top
        pnp = self._pnp_from_bbox(x, y, w, h, block)
        world_point = self.unproject(px, py, d, view, projection)
        pnp_world = self._pnp_world_position(pnp, view)
        if pnp_world is not None and np.linalg.norm(
            np.asarray(pnp_world) - world_point
        ) > 0.03:
            pnp_world = None
        return Detection(
            (px, py), tuple(world_point),
            (x, y, w, h), pnp, pnp_world,
        )

    def visible_block_ids(self, blocks: Sequence[Block], mode: str) -> set[int]:
        image, depth, segmentation, view, projection = self.capture()
        visible: set[int] = set()
        encoded_ids = segmentation.astype(np.int64) & ((1 << 24) - 1)
        hsv = (
            cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            if mode == "opencv"
            else None
        )
        for block in blocks:
            if block.status != "present":
                continue
            face, _ = self.tower.face_center_and_push(block)
            pixel = self.project(face, view, projection)
            if pixel is None:
                continue
            x, y = pixel
            if mode == "ground_truth":
                if encoded_ids[y, x] == block.body_id:
                    visible.add(block.block_id)
                continue
            if mode != "opencv":
                raise ValueError(f"Unknown detector mode: {mode}")
            window = hsv[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            if window.size == 0 or depth[y, x] >= 0.999:
                continue
            wood_pixels = (
                (window[:, :, 0] >= 3)
                & (window[:, :, 0] <= 32)
                & (window[:, :, 1] >= 35)
                & (window[:, :, 2] >= 35)
            )
            if np.mean(wood_pixels) < 0.3:
                continue
            observed_point = self.unproject(
                x, y, float(depth[y, x]), view, projection
            )
            if np.linalg.norm(observed_point - face) <= 0.03:
                visible.add(block.block_id)
        return visible

    def _pnp_from_bbox(self, x: int, y: int, w: int, h: int,
                       block: Block) -> Optional[tuple[tuple[float, ...], tuple[float, ...]]]:
        if w < 2 or h < 2:
            return None
        image_points = np.array([
            [x, y], [x + w - 1, y],
            [x + w - 1, y + h - 1], [x, y + h - 1],
        ], dtype=np.float64)
        horizontal = self.config.block_width
        vertical = self.config.block_height
        object_points = np.array([
            [-horizontal / 2, -vertical / 2, 0],
            [horizontal / 2, -vertical / 2, 0],
            [horizontal / 2, vertical / 2, 0],
            [-horizontal / 2, vertical / 2, 0],
        ], dtype=np.float64)
        try:
            ok, rotation, translation = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.distortion,
                flags=cv2.SOLVEPNP_IPPE,
            )
        except cv2.error:
            return None
        if not ok:
            return None
        return tuple(rotation.ravel()), tuple(translation.ravel())


class PandaArm:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = random.Random(0)
        base_orn = p.getQuaternionFromEuler([0, 0, math.pi])
        self.robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            config.robot_base,
            base_orn,
            useFixedBase=True,
        )
        self.arm_joints = list(range(7))
        self.eef_link = 11
        self.target_orientation = p.getQuaternionFromEuler([0, -math.pi / 2, 0])
        self.probe_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[
                config.probe_width / 2,
                config.probe_width / 2,
                config.probe_width * 0.75,
            ],
        )
        self.probe_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[
                config.probe_width / 2,
                config.probe_width / 2,
                config.probe_width * 0.75,
            ],
            rgbaColor=[0.1, 0.75, 0.9, 1],
        )
        self.probe_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.probe_shape,
            baseVisualShapeIndex=self.probe_visual,
            basePosition=[0.4, 0, 0.4],
        )
        for link in range(-1, p.getNumJoints(self.robot_id)):
            p.setCollisionFilterPair(
                self.robot_id, self.probe_id, link, -1, enableCollision=0
            )
        self.reset()

    def reset(self) -> None:
        ready = [0, -0.4, 0, -2.2, 0, 1.8, 0.8]
        for joint, value in zip(self.arm_joints, ready):
            p.resetJointState(self.robot_id, joint, value)
        self._hold_current()
        self.update_probe()
        for _ in range(80):
            p.stepSimulation()
            self.update_probe()

    def _hold_current(self) -> None:
        positions = [p.getJointState(self.robot_id, i)[0] for i in self.arm_joints]
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joints,
            p.POSITION_CONTROL,
            targetPositions=positions,
            forces=[180] * len(self.arm_joints),
            positionGains=[0.2] * len(self.arm_joints),
            velocityGains=[1.0] * len(self.arm_joints),
        )

    def joint_positions(self) -> list[float]:
        return [p.getJointState(self.robot_id, i)[0] for i in self.arm_joints]

    def end_effector_position(self) -> np.ndarray:
        return np.asarray(
            p.getLinkState(self.robot_id, self.eef_link, computeForwardKinematics=True)[4]
        )

    def update_probe(self) -> None:
        state = p.getLinkState(
            self.robot_id, self.eef_link, computeForwardKinematics=True
        )
        pos, orn = state[4], state[5]
        offset = p.rotateVector(orn, [0, 0, self.config.probe_width * 0.75])
        probe_pos = [pos[i] + offset[i] for i in range(3)]
        p.resetBasePositionAndOrientation(self.probe_id, probe_pos, orn)

    def _ik(self, position: Sequence[float], orientation: Sequence[float]) -> list[float]:
        lower = [p.getJointInfo(self.robot_id, joint)[8] for joint in self.arm_joints]
        upper = [p.getJointInfo(self.robot_id, joint)[9] for joint in self.arm_joints]
        current_states = [p.getJointState(self.robot_id, joint) for joint in self.arm_joints]
        current = [state[0] for state in current_states]
        seeds = [current]
        seeds.extend(
            [self.rng.uniform(low, high) for low, high in zip(lower, upper)]
            for _ in range(64)
        )
        best: Optional[tuple[float, list[float]]] = None
        try:
            for seed in seeds:
                for joint, value in zip(self.arm_joints, seed):
                    p.resetJointState(self.robot_id, joint, value)
                result = p.calculateInverseKinematics(
                    self.robot_id,
                    self.eef_link,
                    position,
                    orientation,
                    maxNumIterations=300,
                    residualThreshold=1e-5,
                )
                candidate = list(result[:7])
                if any(
                    value < low - 1e-5 or value > high + 1e-5
                    for value, low, high in zip(candidate, lower, upper)
                ):
                    continue
                for joint, value in zip(self.arm_joints, candidate):
                    p.resetJointState(self.robot_id, joint, value)
                state = p.getLinkState(
                    self.robot_id, self.eef_link, computeForwardKinematics=True
                )
                position_error = float(np.linalg.norm(
                    np.asarray(state[4]) - np.asarray(position)
                ))
                orientation_error = 1.0 - abs(float(np.dot(state[5], orientation)))
                if position_error > 0.003 or orientation_error > 0.002:
                    continue
                range_scale = np.maximum(np.asarray(upper) - np.asarray(lower), 1e-6)
                motion_cost = float(np.linalg.norm(
                    (np.asarray(candidate) - np.asarray(current)) / range_scale
                ))
                score = position_error + orientation_error + motion_cost * 1e-4
                if best is None or score < best[0]:
                    best = (score, candidate)
        finally:
            for joint, state in zip(self.arm_joints, current_states):
                p.resetJointState(
                    self.robot_id, joint, state[0], targetVelocity=state[1]
                )
        if best is None:
            raise RuntimeError(
                f"No Panda IK solution within joint limits for position {list(position)}"
            )
        return best[1]

    def move_cartesian(self, position: Sequence[float], orientation: Sequence[float],
                       duration: float = 1.5) -> list[tuple[float, float, float]]:
        target = self._ik(position, orientation)
        start = np.asarray(self.joint_positions())
        delta = np.asarray(target) - start
        steps = max(2, int(duration / self.config.time_step))
        profile = trapezoid_profile(steps)
        trajectory = []
        for alpha in profile:
            desired = start + alpha * delta
            p.setJointMotorControlArray(
                self.robot_id,
                self.arm_joints,
                p.POSITION_CONTROL,
                targetPositions=desired.tolist(),
                forces=[180] * 7,
                positionGains=[0.18] * 7,
                velocityGains=[1.0] * 7,
            )
            self.update_probe()
            p.stepSimulation()
            trajectory.append(tuple(self.end_effector_position()))
        return trajectory

    def jacobian_velocity(
        self,
        cartesian_velocity: Sequence[float],
        target_orientation: Sequence[float],
    ) -> np.ndarray:
        movable = list(range(7)) + [9, 10]
        q = [p.getJointState(self.robot_id, i)[0] for i in movable]
        qdot = [p.getJointState(self.robot_id, i)[1] for i in movable]
        zeros = [0.0] * len(movable)
        linear, angular = p.calculateJacobian(
            self.robot_id,
            self.eef_link,
            [0, 0, 0],
            q,
            qdot,
            zeros,
        )
        base_orientation = p.getBasePositionAndOrientation(self.robot_id)[1]
        base_rotation = np.asarray(
            p.getMatrixFromQuaternion(base_orientation), dtype=float
        ).reshape((3, 3))
        linear_world = base_rotation @ np.asarray(linear, dtype=float)
        angular_world = base_rotation @ np.asarray(angular, dtype=float)
        jacobian = np.vstack((linear_world, angular_world))[:, :7]
        state = p.getLinkState(
            self.robot_id, self.eef_link, computeForwardKinematics=True
        )
        inverse_current = p.invertTransform([0, 0, 0], state[5])[1]
        error_quaternion = p.multiplyTransforms(
            [0, 0, 0], target_orientation, [0, 0, 0], inverse_current
        )[1]
        axis, angle = p.getAxisAngleFromQuaternion(error_quaternion)
        angular_velocity = np.clip(
            np.asarray(axis) * angle * 2.0, -0.35, 0.35
        )
        twist = np.concatenate((np.asarray(cartesian_velocity), angular_velocity))
        return np.linalg.pinv(jacobian) @ twist

    def velocity_step(
        self,
        cartesian_velocity: Sequence[float],
        seconds: float,
        target_orientation: Sequence[float],
    ) -> None:
        qdot = np.clip(
            self.jacobian_velocity(cartesian_velocity, target_orientation),
            -0.7,
            0.7,
        )
        lower = [p.getJointInfo(self.robot_id, joint)[8] for joint in self.arm_joints]
        upper = [p.getJointInfo(self.robot_id, joint)[9] for joint in self.arm_joints]
        for index, joint in enumerate(self.arm_joints):
            current = p.getJointState(self.robot_id, joint)[0]
            if (current <= lower[index] + 0.03 and qdot[index] < 0) or (
                current >= upper[index] - 0.03 and qdot[index] > 0
            ):
                qdot[index] = 0.0
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joints,
            p.VELOCITY_CONTROL,
            targetVelocities=qdot.tolist(),
            forces=[100] * 7,
        )
        steps = max(1, int(seconds / self.config.time_step))
        for _ in range(steps):
            self.update_probe()
            p.stepSimulation()


class JengaSimulation:
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        detector: str = "ground_truth",
        controller: str = "closed_loop",
        seed: int = 0,
        gui: bool = False,
    ):
        self.config = config or SimulationConfig()
        if not math.isfinite(self.config.block_mass_kg) or self.config.block_mass_kg <= 0:
            raise ValueError("block_mass_kg must be positive")
        friction_low, friction_high = self.config.block_friction_range
        if (
            not math.isfinite(friction_low)
            or not math.isfinite(friction_high)
            or friction_low < 0
            or friction_high < friction_low
        ):
            raise ValueError("block_friction_range must satisfy 0 <= low <= high")
        force_parameters = (
            self.config.force_setpoint_n,
            self.config.force_control_gain_m_s_n,
            self.config.force_control_contact_n,
            self.config.force_control_max_correction_m_s,
            self.config.max_push_speed_m_s,
            self.config.force_control_hz,
        )
        if (
            not all(math.isfinite(value) for value in force_parameters)
            or self.config.force_setpoint_n <= 0
            or self.config.force_control_gain_m_s_n < 0
            or self.config.force_control_contact_n < 0
            or self.config.force_control_max_correction_m_s <= 0
            or self.config.max_push_speed_m_s <= 0
            or self.config.force_control_hz <= 0
        ):
            raise ValueError("force-control settings must be finite and non-negative")
        if detector not in {"ground_truth", "opencv"}:
            raise ValueError("detector must be 'ground_truth' or 'opencv'")
        if controller not in {"closed_loop", "open_loop"}:
            raise ValueError("controller must be 'closed_loop' or 'open_loop'")
        self.detector_mode = detector
        self.controller = controller
        self.seed = seed
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self.gui = gui
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(self.config.time_step)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.config.time_step,
            numSolverIterations=100,
            deterministicOverlappingPairs=1,
        )
        self.tower = Tower(self.config, seed)
        self.arm = PandaArm(self.config)
        for block in self.tower.blocks:
            for link in range(-1, p.getNumJoints(self.arm.robot_id)):
                p.setCollisionFilterPair(
                    self.arm.robot_id, block.body_id, link, -1, enableCollision=0
                )
            p.setCollisionFilterPair(
                self.arm.probe_id, block.body_id, -1, -1, enableCollision=0
            )
        self.vision = Vision(
            self.config, self.tower, self.arm.robot_id, self.arm.eef_link
        )
        self.policy = BlockSelectionPolicy(self.tower.blocks, self.config.layers, seed)
        self.attempts: list[AttemptResult] = []
        self.trial_index = 0

    def _tool_orientation(self, push: np.ndarray) -> tuple[float, float, float, float]:
        if np.allclose(push, [-1, 0, 0]):
            return p.getQuaternionFromEuler([0, -math.pi / 2, 0])
        if np.allclose(push, [0, 1, 0]):
            return p.getQuaternionFromEuler([-math.pi / 2, 0, 0])
        return p.getQuaternionFromEuler([0, math.pi / 2, 0])

    def _move_to_overview(self) -> list[tuple[float, float, float]]:
        eye = np.asarray([0.35, -0.40, 0.50], dtype=float)
        look_at = np.asarray([
            self.config.tower_origin[0],
            self.config.tower_origin[1],
            self.config.tower_origin[2] + self.config.layers * self.config.block_height / 2,
        ])
        world_up = np.asarray([0.0, 0.0, 1.0])
        forward = look_at - eye + np.asarray([0.0, 0.0, 0.08])
        forward /= np.linalg.norm(forward)
        lateral = np.cross(forward, world_up)
        lateral /= np.linalg.norm(lateral)
        end_effector_position = (
            eye
            + forward * 0.085
            - lateral * self.config.camera_lateral_offset
            - np.asarray([0.0, 0.0, 0.08])
        )
        axis = np.cross(world_up, forward)
        angle = math.acos(float(np.clip(np.dot(world_up, forward), -1.0, 1.0)))
        orientation = p.getQuaternionFromAxisAngle(axis, angle)
        return self.arm.move_cartesian(
            end_effector_position,
            orientation,
            duration=1.0,
        )

    def _set_probe_tower_collisions(self, enabled: bool) -> None:
        for block in self.tower.blocks:
            p.setCollisionFilterPair(
                self.arm.probe_id,
                block.body_id,
                -1,
                -1,
                enableCollision=int(enabled),
            )

    def _align(self, block: Block, face: np.ndarray, push: np.ndarray) -> tuple[Optional[Detection], float, list[tuple[float, float, float]]]:
        orientation = self._tool_orientation(push)
        outward = -push
        pre_push = face + outward * (
            self.config.approach_distance + self.config.probe_clearance
        )
        staging = pre_push + outward * 0.06
        staging[2] = pre_push[2]
        current_orientation = p.getLinkState(
            self.arm.robot_id, self.arm.eef_link, computeForwardKinematics=True
        )[5]
        trajectory = self.arm.move_cartesian(
            staging, current_orientation, duration=1.5
        )
        trajectory.extend(
            self.arm.move_cartesian(pre_push, orientation, duration=1.0)
        )
        detection = self.vision.target_detection(block, self.detector_mode)
        if detection is None:
            return None, 0.0, trajectory
        servo_period = 1.0 / self.config.servo_hz
        fixed_face = np.asarray(
            detection.pnp_world_position or detection.world_point
        )
        if self.controller == "open_loop":
            target = fixed_face + outward * self.config.probe_clearance
            trajectory.extend(
                self.arm.move_cartesian(target, orientation, duration=1.2)
            )
            return detection, 1.2, trajectory
        for iteration in range(int(self.config.max_servo_seconds * self.config.servo_hz)):
            if iteration > 0:
                detection = self.vision.target_detection(block, self.detector_mode)
                if detection is None:
                    return None, (iteration + 1) * servo_period, trajectory
                fixed_face = np.asarray(detection.world_point)
            target = fixed_face + outward * self.config.probe_clearance
            position = self.arm.end_effector_position()
            error = target - position
            if np.linalg.norm(error) <= self.config.position_tolerance:
                return detection, iteration * servo_period, trajectory
            velocity = np.clip(2.5 * error, -0.16, 0.16)
            self.arm.velocity_step(velocity, servo_period, orientation)
            trajectory.append(tuple(self.arm.end_effector_position()))
        return None, self.config.max_servo_seconds, trajectory

    def _contact_force(self) -> float:
        contacts = p.getContactPoints(bodyA=self.arm.probe_id)
        return sum(float(contact[9]) for contact in contacts)

    def _push(
        self,
        block: Block,
        push: np.ndarray,
        threshold: float,
        initial_face: Sequence[float],
    ) -> tuple[str, float, list[tuple[float, float]], float, float, int]:
        start = self.tower.block_center(block)
        total_steps = int(
            self.config.push_distance
            / (self.config.max_push_speed_m_s * self.config.time_step)
        )
        sensor_steps = max(1, int(1.0 / self.config.sensor_hz / self.config.time_step))
        servo_steps = max(1, int(1.0 / self.config.servo_hz / self.config.time_step))
        control_steps = max(
            1,
            int(round(1.0 / self.config.force_control_hz / self.config.time_step)),
        )
        force_profile: list[tuple[float, float]] = []
        peak = 0.0
        observed_face = np.asarray(initial_face, dtype=float)
        push_height = float(observed_face[2])
        tracking_misses = 0
        measured_force = 0.0
        force_limit = max(
            self.config.first_force_threshold,
            self.config.second_force_threshold,
        )
        for step in range(0, total_steps, control_steps):
            measured_force = self._contact_force()
            peak = max(peak, measured_force)
            block.peak_force = peak
            if step % sensor_steps == 0:
                force_profile.append((step * self.config.time_step, measured_force))
            if measured_force >= force_limit:
                break
            if self.controller == "closed_loop":
                if step > 0 and step % servo_steps == 0:
                    detection = self.vision.target_detection(block, self.detector_mode)
                    if detection is None:
                        tracking_misses += 1
                    else:
                        observed_face = np.asarray(detection.world_point)
                        observed_face[2] = push_height
                target = observed_face - push * self.config.probe_clearance
                error = target + push * (
                    self.config.push_distance * step / max(1, total_steps)
                ) - self.arm.end_effector_position()
                velocity = np.clip(2.0 * error, -0.08, 0.08)
                axial_velocity = float(np.dot(velocity, push))
                bounded_axial_velocity = float(np.clip(
                    axial_velocity,
                    -self.config.max_push_speed_m_s,
                    self.config.max_push_speed_m_s,
                ))
                velocity += push * (bounded_axial_velocity - axial_velocity)
                if measured_force >= self.config.force_control_contact_n:
                    correction = force_feedback_correction(
                        measured_force,
                        self.config.force_setpoint_n,
                        self.config.force_control_gain_m_s_n,
                        self.config.force_control_max_correction_m_s,
                    )
                    commanded_axial_velocity = float(np.clip(
                        bounded_axial_velocity + correction,
                        -self.config.force_control_max_correction_m_s,
                        2 * self.config.max_push_speed_m_s,
                    ))
                    velocity += push * (
                        commanded_axial_velocity - bounded_axial_velocity
                    )
                self.arm.velocity_step(
                    velocity,
                    min(control_steps, total_steps - step) * self.config.time_step,
                    self._tool_orientation(push),
                )
            else:
                target = np.asarray(initial_face) - push * self.config.probe_clearance
                error = target + push * (
                    self.config.push_distance * step / max(1, total_steps)
                ) - self.arm.end_effector_position()
                self.arm.velocity_step(
                    np.clip(2.0 * error, -0.08, 0.08),
                    min(control_steps, total_steps - step) * self.config.time_step,
                    self._tool_orientation(push),
                )
            block_position = self.tower.block_center(block)
            axial_displacement = float(np.dot(block_position - start, push))
            vertical_drop = float(start[2] - block_position[2])
            if (
                axial_displacement >= self.config.block_length
                or (
                    axial_displacement >= 0.9 * self.config.block_length
                    and vertical_drop >= self.config.block_height
                )
            ):
                break
        end = self.tower.block_center(block)
        axial_displacement = float(np.dot(end - start, push))
        vertical_drop = float(start[2] - end[2])
        cleared_layer = (
            axial_displacement >= self.config.block_length
            or (
                axial_displacement >= 0.9 * self.config.block_length
                and vertical_drop >= self.config.block_height
            )
        )
        if cleared_layer:
            outcome = "extracted"
        elif peak >= threshold:
            outcome = "immobile"
        else:
            outcome = "error"
        return (
            outcome,
            peak,
            force_profile,
            axial_displacement * 1000.0,
            vertical_drop * 1000.0,
            tracking_misses,
        )

    def _retract(self, face: np.ndarray, push: np.ndarray) -> list[tuple[float, float, float]]:
        orientation = self._tool_orientation(push)
        target = face - push * (
            self.config.approach_distance + self.config.probe_clearance
        )
        return self.arm.move_cartesian(target, orientation, duration=1.2)

    def run_attempt(self) -> Optional[AttemptResult]:
        self._move_to_overview()
        visible_block_ids = self.vision.visible_block_ids(
            self.tower.blocks,
            self.detector_mode,
        )
        block = self.policy.next_block(visible_block_ids)
        if block is None:
            return None
        block.attempts += 1
        face, push = self.tower.face_center_and_push(block)
        threshold = (
            self.config.first_force_threshold
            if self.policy.phase == 0
            else self.config.second_force_threshold
        )
        detection, servo_time, trajectory = self._align(block, face, push)
        alignment = None
        if detection is None:
            outcome, peak, force_profile = "error", 0.0, []
            axial_displacement_mm = None
            vertical_drop_mm = None
            tracking_misses = 0
        else:
            ground_truth_face, push = self.tower.face_center_and_push(block)
            contact_point = (
                self.arm.end_effector_position()
                + push * self.config.probe_width * 1.5
            )
            tangential_error = contact_point - ground_truth_face
            tangential_error -= push * float(np.dot(tangential_error, push))
            alignment = float(np.linalg.norm(
                tangential_error
            ) * 1000)
            push_face = (
                detection.pnp_world_position or detection.world_point
                if self.controller == "open_loop"
                else detection.world_point
            )
            self._set_probe_tower_collisions(True)
            try:
                (
                    outcome,
                    peak,
                    force_profile,
                    axial_displacement_mm,
                    vertical_drop_mm,
                    tracking_misses,
                ) = self._push(
                    block, push, threshold, push_face
                )
            finally:
                self._set_probe_tower_collisions(False)
        self.policy.observe(block, outcome)
        if outcome in {"error", "immobile", "extracted"}:
            self._retract(face, push)
        collapsed = self.tower.has_collapsed()
        extracted = outcome == "extracted"
        if collapsed:
            outcome = "tower_collapse"
        result = AttemptResult(
            trial=self.trial_index,
            block_id=block.block_id,
            layer=block.layer,
            detector=self.detector_mode,
            controller=self.controller,
            threshold_n=threshold,
            outcome=outcome,
            peak_force_n=peak,
            alignment_error_mm=alignment,
            servo_time_s=servo_time,
            extracted=extracted,
            tower_collapsed=collapsed,
            axial_displacement_mm=axial_displacement_mm,
            vertical_drop_mm=vertical_drop_mm,
            vision_tracking_misses=tracking_misses,
            trajectory=trajectory,
            force_profile=force_profile,
        )
        self.attempts.append(result)
        if self.gui:
            p.stepSimulation()
            time.sleep(self.config.time_step)
        return result

    def run_game(self, trial_index: int = 0, max_attempts: Optional[int] = None) -> list[AttemptResult]:
        self.trial_index = trial_index
        limit = max_attempts or self.config.max_attempts_per_game
        start = len(self.attempts)
        for _ in range(limit):
            if self.tower.has_collapsed():
                break
            attempt = self.run_attempt()
            if attempt is None or attempt.tower_collapsed:
                break
        return self.attempts[start:]

    def close(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    def __enter__(self) -> "JengaSimulation":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def aggregate(results: Sequence[AttemptResult]) -> dict[str, Any]:
    total = len(results)
    extracted = sum(result.outcome == "extracted" for result in results)
    physically_extracted = sum(result.extracted for result in results)
    immobile = sum(result.outcome == "immobile" for result in results)
    errors = sum(result.outcome not in {"extracted", "immobile"} for result in results)
    successful = extracted + immobile
    alignment = [
        result.alignment_error_mm for result in results
        if result.alignment_error_mm is not None
    ]
    convergence = [result.servo_time_s for result in results if result.servo_time_s > 0]
    trial_groups: dict[int, list[AttemptResult]] = {}
    for result in results:
        trial_groups.setdefault(result.trial, []).append(result)
    trial_successes: list[int] = []
    trial_extractions: list[int] = []
    trial_errors: list[int] = []
    trial_longest_streaks: list[int] = []
    for trial_results in trial_groups.values():
        trial_successes.append(sum(
            result.outcome in {"extracted", "immobile"} for result in trial_results
        ))
        trial_extractions.append(sum(
            result.outcome == "extracted" for result in trial_results
        ))
        trial_errors.append(sum(
            result.outcome not in {"extracted", "immobile"} for result in trial_results
        ))
        longest = 0
        streak = 0
        for result in trial_results:
            if result.outcome in {"extracted", "immobile"}:
                streak += 1
                longest = max(longest, streak)
            else:
                streak = 0
        trial_longest_streaks.append(longest)

    def trial_summary(values: Sequence[int]) -> dict[str, Optional[float]]:
        if not values:
            return {"mean": None, "std": None, "ci95": None}
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else None
        return {"mean": mean, "std": std, "ci95": ci95}

    return {
        "trials": len(trial_groups),
        "attempts": total,
        "extracted": extracted,
        "blocks_physically_extracted": physically_extracted,
        "immobile_classified": immobile,
        "correctly_classified_attempts": successful,
        "errors": errors,
        "extraction_rate": extracted / total if total else 0.0,
        "immobile_classification_rate": immobile / total if total else 0.0,
        "correctly_classified_attempt_rate": successful / total if total else 0.0,
        "error_rate": errors / total if total else 0.0,
        "alignment_error_mm_mean": float(np.mean(alignment)) if alignment else None,
        "alignment_error_mm_std": float(np.std(alignment)) if alignment else None,
        "servo_time_s_mean": float(np.mean(convergence)) if convergence else None,
        "servo_time_s_std": float(np.std(convergence)) if convergence else None,
        "peak_force_n_mean": float(np.mean([r.peak_force_n for r in results])) if total else None,
        "maximum_extractions": max(trial_extractions, default=0),
        "successful_attempts_per_trial": trial_summary(trial_successes),
        "extractions_per_trial": trial_summary(trial_extractions),
        "errors_per_trial": trial_summary(trial_errors),
        "longest_success_streak_per_trial": trial_summary(trial_longest_streaks),
    }


def _save_results(
    results: Sequence[AttemptResult],
    output_dir: str,
    config: SimulationConfig,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for result in results:
        row = asdict(result)
        row.pop("trajectory", None)
        row.pop("force_profile", None)
        rows.append(row)
    csv_path = os.path.join(output_dir, "attempts.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    combinations: dict[str, list[AttemptResult]] = {}
    for result in results:
        key = f"{result.detector}_{result.controller}"
        combinations.setdefault(key, []).append(result)
    summary: dict[str, Any] = {
        "assumptions": {
            "block_mass_kg": config.block_mass_kg,
            "block_friction_range": config.block_friction_range,
            "physical_parameters_are_calibrated": False,
            "note": (
                "Mass and static friction are literature baselines from the "
                "supplied mechanics paper, not measurements of the user's set; "
                "PyBullet's friction model is approximate."
            ),
        },
        "configuration": asdict(config),
        "comparisons": {
            key: aggregate(group) for key, group in sorted(combinations.items())
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    _plot_results(results, output_dir, combinations)
    return summary


def _plot_results(
    results: Sequence[AttemptResult],
    output_dir: str,
    combinations: dict[str, list[AttemptResult]],
) -> None:
    if not results:
        return
    labels = ["extracted", "immobile", "error", "tower_collapse"]
    fig, ax = plt.subplots(figsize=(7, 4))
    group_names = sorted(combinations)
    width = 0.8 / len(group_names)
    x_positions = np.arange(len(labels))
    colors = ["#4c956c", "#e9c46a", "#e76f51", "#9b2226"]
    for index, group_name in enumerate(group_names):
        counts = [
            sum(result.outcome == label for result in combinations[group_name])
            for label in labels
        ]
        ax.bar(
            x_positions + index * width,
            counts,
            width,
            label=group_name,
            color=colors[index % len(colors)],
        )
    ax.set_xticks(x_positions + width * (len(group_names) - 1) / 2, labels)
    ax.set_ylabel("Attempts")
    ax.set_title("Jenga attempt outcomes")
    if len(group_names) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "outcomes.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for result in results:
        if result.force_profile:
            profile = np.asarray(result.force_profile)
            ax.plot(profile[:, 0], profile[:, 1], alpha=0.55)
    ax.axhline(
        SimulationConfig().force_setpoint_n,
        color="#6a4c93",
        linestyle="-.",
        label=f"{SimulationConfig().force_setpoint_n:.2f} N force target",
    )
    ax.axhline(0.32, color="#d1495b", linestyle="--", label="0.32 N")
    ax.axhline(0.18, color="#00798c", linestyle=":", label="0.18 N")
    ax.set_xlabel("Push time (s)")
    ax.set_ylabel("Reaction force (N)")
    ax.legend()
    ax.set_title("Force profiles")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "force_profiles.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for result in results:
        if result.trajectory:
            path = np.asarray(result.trajectory)
            ax.plot(path[:, 0], path[:, 2], alpha=0.6)
    ax.set_xlabel("End-effector x (m)")
    ax.set_ylabel("End-effector z (m)")
    ax.set_title("End-effector trajectories")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectories.png"), dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=["ground_truth", "opencv"], default="ground_truth")
    parser.add_argument("--controller", choices=["closed_loop", "open_loop"], default="closed_loop")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all ground-truth/OpenCV and open-loop/closed-loop combinations.",
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--block-mass-kg",
        type=float,
        default=SimulationConfig.block_mass_kg,
        help="Simulated block mass in kg; default is a literature average, not calibrated to your set.",
    )
    parser.add_argument(
        "--friction-min",
        type=float,
        default=SimulationConfig.block_friction_range[0],
        help="Lower bound for simulated PyBullet friction; literature-based default is fixed at 0.4.",
    )
    parser.add_argument(
        "--friction-max",
        type=float,
        default=SimulationConfig.block_friction_range[1],
        help="Upper bound for simulated PyBullet friction; literature-based default is fixed at 0.4.",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    try:
        config = SimulationConfig(
            block_mass_kg=args.block_mass_kg,
            block_friction_range=(args.friction_min, args.friction_max),
        )
    except ValueError as error:
        parser.error(str(error))
    combinations = (
        [
            (detector, controller)
            for detector in ("ground_truth", "opencv")
            for controller in ("open_loop", "closed_loop")
        ]
        if args.compare
        else [(args.detector, args.controller)]
    )
    results: list[AttemptResult] = []
    for detector, controller in combinations:
        for trial in range(args.trials):
            with JengaSimulation(
                config=config,
                detector=detector,
                controller=controller,
                seed=args.seed + trial,
                gui=args.gui,
            ) as simulation:
                results.extend(simulation.run_game(trial, args.max_attempts))
    summary = _save_results(results, args.output, config)
    print(json.dumps(summary["comparisons"], indent=2))
    print(f"Results written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
