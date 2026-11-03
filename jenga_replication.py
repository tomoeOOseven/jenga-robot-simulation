"""Simulated replication of Marchionna et al. (2023) Jenga pipeline in PyBullet.

Pipeline (AR523 presentation, four stages):
  1. Block detection: simulator ground truth (perfect instance masks and poses)
     or a classical OpenCV pipeline (colour threshold, contours, PnP, depth
     refinement), associated with a registered tower map.
  2. Block selection: the paper's two-sub-space policy with a memory buffer,
     one extraction per level and phase-dependent force thresholds
     (0.32 N in the first sub-space, 0.18 N in the second).
  3. Approach: Panda inverse kinematics and trapezoidal joint trajectories.
  4. Alignment and push: 2 1/2-D visual servoing v = -lambda L+ e with
     s = (x, y, log(Z/Z*), theta u), mapped to joints with the Jacobian
     pseudo-inverse, then an open-loop push monitored by a 20 Hz 1-D force
     sensor (a spring-loaded fingertip) that aborts above the threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

DETECTORS = ("ground_truth", "opencv")
CONTROLLERS = ("open_loop", "closed_loop")
OUTCOMES = ("extracted", "immobile", "error")
WORLD_UP = np.array([0.0, 0.0, 1.0])
MAXIMAL_COORDINATES = True  # blocks are free rigid bodies
SEAM_INSET = 0.0012   # width of the dark seam around every block face (m)


# --------------------------------------------------------------------------
# Configuration and records
# --------------------------------------------------------------------------

@dataclass
class SimulationConfig:
    # Tower (presentation / paper block size; mechanics-paper mass and friction)
    layers: int = 18
    blocks_per_layer: int = 3
    block_length: float = 0.075
    block_width: float = 0.025
    block_height: float = 0.015
    block_mass_kg: float = 0.0196
    block_friction_range: tuple[float, float] = (0.4, 0.4)
    block_height_tolerance_m: float = 1.0e-4   # std of per-block height error
    block_placement_jitter_m: float = 5.0e-4   # +/- shift along the long axis
    block_gap_m: float = 2.0e-4                # clearance between blocks in a layer
    stand_height: float = 0.10
    tower_origin: tuple[float, float] = (0.0, 0.0)

    # Robot and tool
    robot_base: tuple[float, float, float] = (0.55, -0.55, 0.0)
    finger_length: float = 0.17       # flange to fingertip
    fingertip_length: float = 0.02
    probe_width: float = 0.011        # paper: 11 mm fingertip
    probe_height: float = 0.009
    camera_up_offset: float = 0.08    # camera above the finger axis
    camera_back_offset: float = 0.15  # camera behind the fingertip
    joint_bias_std_rad: float = 0.003  # encoder offsets = kinematic calibration error

    # Force sensor (1-D, on the fingertip; paper: 5 N full scale, 12 bit)
    sensor_stiffness_n_m: float = 100.0  # 0.32 N = 3.2 mm of tip travel
    sensor_damping_ratio: float = 0.7
    fingertip_mass_kg: float = 0.02
    sensor_full_scale_n: float = 5.0
    sensor_bits: int = 12
    sensor_hz: float = 20.0
    first_force_threshold: float = 0.32
    second_force_threshold: float = 0.18
    classification_travel: float = 0.015   # block travel that proves it is mobile
    push_safety_force: float = 0.64        # abort limit once the block is moving

    # Camera (eye-in-hand RGB-D)
    camera_width: int = 640
    camera_height: int = 480
    camera_fov: float = 60.0
    camera_near: float = 0.02
    camera_far: float = 2.0
    depth_noise_m: float = 5.0e-4
    depth_scale_error: float = 0.01    # std of the per-camera systematic depth scale error
    overview_distance: float = 0.30

    # Motion and control
    time_step: float = 1.0 / 240.0
    solver_iterations: int = 300       # friction in a tall stack needs many PGS iterations
    servo_hz: float = 9.0
    servo_gain: float = 1.5
    servo_max_linear_m_s: float = 0.04
    servo_max_angular_rad_s: float = 0.4
    servo_tolerance_m: float = 5.0e-4
    servo_tolerance_rad: float = 0.005
    max_servo_seconds: float = 15.0
    max_tracking_misses: int = 4
    position_tolerance: float = 0.007  # paper: max error from the face centre
    staging_distance: float = 0.20
    approach_distance: float = 0.10
    probe_clearance: float = 0.025
    push_speed_m_s: float = 0.010   # stiffness x speed = 1 N/s force rate
    push_overtravel: float = 0.008
    approach_speed_m_s: float = 0.05
    max_joint_speed_rad_s: float = 1.0

    # Game
    min_level: int = 2
    max_level: int = 15                # 14 levels -> paper's minimum of 42 attempts
    max_attempts_per_block: int = 2
    max_attempts_per_game: int = 60

    def __post_init__(self) -> None:
        if not math.isfinite(self.block_mass_kg) or self.block_mass_kg <= 0:
            raise ValueError("block_mass_kg must be positive")
        low, high = self.block_friction_range
        if not (math.isfinite(low) and math.isfinite(high)) or low < 0 or high < low:
            raise ValueError("block_friction_range must satisfy 0 <= low <= high")
        if self.block_height_tolerance_m < 0 or self.joint_bias_std_rad < 0:
            raise ValueError("tolerances must be non-negative")
        if self.first_force_threshold <= 0 or self.second_force_threshold <= 0:
            raise ValueError("force thresholds must be positive")
        if not 0 <= self.min_level <= self.max_level < self.layers - 1:
            raise ValueError("levels must satisfy 0 <= min_level <= max_level < layers - 1")
        if self.sensor_stiffness_n_m <= 0 or self.push_speed_m_s <= 0:
            raise ValueError("sensor stiffness and push speed must be positive")

    @property
    def tower_base_z(self) -> float:
        return self.stand_height

    @property
    def split_level(self) -> int:
        """First level of the upper sub-space."""
        return self.min_level + (self.max_level - self.min_level + 1) // 2


@dataclass
class Block:
    block_id: int
    layer: int
    slot: int
    long_axis: str
    body_id: int
    status: str = "present"        # present | tested | extracted | skipped
    peak_force: float = 0.0
    attempts: int = 0
    threshold: Optional[float] = None


@dataclass
class FaceObservation:
    """A block end face seen by the camera, in camera (OpenCV) coordinates."""
    center_cam: np.ndarray
    rotation_cam: np.ndarray        # columns: face x (right), y (up), z (outward)
    pixel: tuple[float, float]
    area_px: int
    block_id: Optional[int] = None  # only known to the ground-truth detector
    world_center: Optional[np.ndarray] = None
    world_normal: Optional[np.ndarray] = None


@dataclass
class AttemptResult:
    trial: int
    seed: int
    detector: str
    controller: str
    attempt: int
    block_id: int
    layer: int
    slot: int
    phase: int
    threshold_n: float
    outcome: str
    success: bool
    failure_reason: str
    peak_force_n: float
    alignment_error_mm: Optional[float]
    alignment_error_h_mm: Optional[float]
    alignment_error_v_mm: Optional[float]
    servo_time_s: Optional[float]
    servo_iterations: int
    axial_displacement_mm: Optional[float]
    tower_perturbation_mm: float
    tower_collapsed: bool
    sim_time_s: float
    trajectory: list[tuple[float, float, float]] = field(default_factory=list)
    force_profile: list[tuple[float, float]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Small maths helpers
# --------------------------------------------------------------------------

def trapezoid_profile(steps: int, acceleration_fraction: float = 0.2) -> np.ndarray:
    """Normalised position samples (0 -> 1) of a trapezoidal velocity profile."""
    if steps < 2:
        return np.array([0.0, 1.0])
    ta = float(np.clip(acceleration_fraction, 1e-3, 0.5))
    s = np.linspace(0.0, 1.0, steps)
    vmax = 1.0 / (1.0 - ta)
    out = np.where(
        s < ta,
        0.5 * vmax / ta * s ** 2,
        np.where(
            s <= 1.0 - ta,
            vmax * (s - 0.5 * ta),
            1.0 - 0.5 * vmax / ta * (1.0 - s) ** 2,
        ),
    )
    out[0], out[-1] = 0.0, 1.0
    return out


def matrix_from_quat(quaternion: Sequence[float]) -> np.ndarray:
    return np.asarray(p.getMatrixFromQuaternion(quaternion), dtype=float).reshape(3, 3)


def quat_from_matrix(rotation: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(rotation, dtype=float)
    trace = np.trace(r)
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w, x = 0.25 * s, (r[2, 1] - r[1, 2]) / s
        y, z = (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w, x = (r[2, 1] - r[1, 2]) / s, 0.25 * s
        y, z = (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w, x = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s
        y, z = 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w, x = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s
        y, z = (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Logarithm map: theta * u of a rotation matrix."""
    r = np.asarray(rotation, dtype=float)
    cos_theta = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    if math.pi - theta < 1e-4:
        axis = np.sqrt(np.maximum((np.diag(r) + 1.0) / 2.0, 0.0))
        k = int(np.argmax(axis))
        axis[(k + 1) % 3] = math.copysign(axis[(k + 1) % 3], r[k, (k + 1) % 3])
        axis[(k + 2) % 3] = math.copysign(axis[(k + 2) % 3], r[k, (k + 2) % 3])
        return axis / np.linalg.norm(axis) * theta
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return vee * theta / (2.0 * math.sin(theta))


def make_pose(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def invert_pose(pose: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inverse


def face_frame(normal: np.ndarray) -> np.ndarray:
    """Face axes: z outward normal, y vertical up, x = y x z (viewer's right)."""
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    y = WORLD_UP - z * float(np.dot(WORLD_UP, z))
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack((x, y, z))


def tool_in_face(clearance: float) -> np.ndarray:
    """Tool pose in the face frame for a fingertip `clearance` in front of the
    face centre, pushing along -z_face with tool x pointing down."""
    rotation = np.column_stack(((0.0, -1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
    return make_pose(rotation, (0.0, 0.0, clearance))


def interaction_matrix(x: float, y: float, depth: float) -> np.ndarray:
    """2 1/2-D interaction matrix for s = (x, y, log Z, theta u) with L_w ~= I."""
    inv_z = 1.0 / depth
    return np.array([
        [-inv_z, 0.0, x * inv_z, x * y, -(1.0 + x * x), y],
        [0.0, -inv_z, y * inv_z, 1.0 + y * y, -x * y, -x],
        [0.0, 0.0, -inv_z, -y, x, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ])


def visual_servo_twist(
    current: np.ndarray,
    desired: np.ndarray,
    gain: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper eq. (1): v = -lambda L+ e for the face pose seen by the camera.

    `current` and `desired` are 4x4 poses of the face in the camera frame.
    Returns the camera twist (6,) and the feature error (6,).
    """
    point, point_star = current[:3, 3], desired[:3, 3]
    x, y = point[0] / point[2], point[1] / point[2]
    x_star, y_star = point_star[0] / point_star[2], point_star[1] / point_star[2]
    # theta u of the rotation between desired and current camera frames,
    # expressed in the current camera frame (c*R_c).
    c_star_r_c = desired[:3, :3] @ current[:3, :3].T
    error = np.concatenate((
        [x - x_star, y - y_star, math.log(point[2] / point_star[2])],
        rotation_vector(c_star_r_c),
    ))
    matrix = interaction_matrix(x, y, point[2])
    twist = -gain * np.linalg.pinv(matrix) @ error
    return twist, error


def damped_pinv(jacobian: np.ndarray, damping: float = 0.01) -> np.ndarray:
    jjt = jacobian @ jacobian.T
    return jacobian.T @ np.linalg.inv(jjt + damping ** 2 * np.eye(jjt.shape[0]))


def t_critical_95(dof: int) -> float:
    table = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262,
             2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101,
             2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052,
             2.048, 2.045, 2.042]
    return table[dof - 1] if 1 <= dof <= len(table) else 1.96


def mean_std_ci(values: Sequence[float]) -> dict[str, Optional[float]]:
    data = [float(v) for v in values if v is not None]
    if not data:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = float(np.mean(data))
    if len(data) < 2:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95": None}
    std = float(np.std(data, ddof=1))
    return {
        "n": len(data),
        "mean": mean,
        "std": std,
        "ci95": t_critical_95(len(data) - 1) * std / math.sqrt(len(data)),
    }


def quantize_force(force: float, config: SimulationConfig) -> float:
    resolution = config.sensor_full_scale_n / (2 ** config.sensor_bits)
    clipped = float(np.clip(force, 0.0, config.sensor_full_scale_n))
    return round(clipped / resolution) * resolution


# --------------------------------------------------------------------------
# Block selection policy (paper section 3.2)
# --------------------------------------------------------------------------

class BlockSelectionPolicy:
    """Two sub-spaces (lower / upper levels), random start, memory buffer.

    The first sub-space uses the aggressive threshold (0.32 N). Once all its
    levels have been tested the policy switches sub-space and threshold
    (0.18 N). Only one block per level may be extracted; after an immobile
    block the policy keeps trying the remaining blocks of the same level.
    """

    def __init__(self, blocks: Sequence[Block], config: SimulationConfig, seed: int = 0):
        self.blocks = list(blocks)
        self.config = config
        self.rng = random.Random(seed)
        self.subspaces = (
            range(config.min_level, config.split_level),
            range(config.split_level, config.max_level + 1),
        )
        self.current_subspace = self.rng.randrange(2)
        self.phase = 0
        self.current_level: Optional[int] = None

    @property
    def force_threshold(self) -> float:
        return (
            self.config.first_force_threshold
            if self.phase == 0
            else self.config.second_force_threshold
        )

    def _level_open(self, level: int) -> bool:
        return not any(
            b.layer == level and b.status == "extracted" for b in self.blocks
        )

    def candidates(
        self, subspace: int, visible: Optional[set[int]] = None
    ) -> list[Block]:
        levels = self.subspaces[subspace]
        return [
            b for b in self.blocks
            if b.layer in levels
            and b.status == "present"
            and self._level_open(b.layer)
            and (visible is None or b.block_id in visible)
        ]

    def next_block(self, visible: Optional[set[int]] = None) -> Optional[Block]:
        eligible = self.candidates(self.current_subspace, visible)
        if not eligible and self.phase == 0:
            self.phase = 1
            self.current_subspace = 1 - self.current_subspace
            self.current_level = None
            eligible = self.candidates(self.current_subspace, visible)
        if not eligible:
            return None
        in_level = [b for b in eligible if b.layer == self.current_level]
        if not in_level:
            self.current_level = self.rng.choice(sorted({b.layer for b in eligible}))
            in_level = [b for b in eligible if b.layer == self.current_level]
        # Centre block first: its extraction cannot torque the level above
        # (mechanics paper, section 2.2); then the side blocks in random order.
        in_level.sort(key=lambda b: (b.slot != 1, self.rng.random()))
        return in_level[0]

    def observe(self, block: Block, outcome: str, peak_force: float = 0.0) -> None:
        block.attempts += 1
        block.peak_force = max(block.peak_force, peak_force)
        block.threshold = self.force_threshold
        if outcome == "extracted":
            block.status = "extracted"
            self.current_level = None
        elif outcome == "immobile":
            block.status = "tested"
            self.current_level = block.layer
        elif outcome == "error":
            # Paper: errors abort the attempt without updating the policy;
            # the attempt counter stops endless retries of an unreachable block.
            if block.attempts >= self.config.max_attempts_per_block:
                block.status = "skipped"
        else:
            raise ValueError(f"Unknown attempt outcome: {outcome}")


# --------------------------------------------------------------------------
# Tower (ground truth) and tower map (the robot's belief)
# --------------------------------------------------------------------------

def _wood_colour(rng: np.random.Generator) -> list[float]:
    hsv = np.array([[[rng.uniform(11, 19), rng.uniform(110, 170), rng.uniform(170, 235)]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0] / 255.0
    return [float(c) for c in rgb] + [1.0]


def _block_visual(extents: Sequence[float]) -> int:
    """Dark core plus inset wood plates on five faces: the dark seam framing
    each face makes touching blocks separable in the image, like the chamfered
    edges of real blocks."""
    half_extents = [[e - 2e-4 for e in extents]]
    positions = [[0.0, 0.0, 0.0]]
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1)):
        plate = [e - SEAM_INSET for e in extents]
        plate[axis] = 1e-4
        center = [0.0, 0.0, 0.0]
        center[axis] = sign * (extents[axis] - 5e-5)
        half_extents.append(plate)
        positions.append(center)
    return p.createVisualShapeArray(
        shapeTypes=[p.GEOM_BOX] * len(half_extents),
        halfExtents=half_extents,
        visualFramePositions=positions,
    )


class Tower:
    def __init__(self, config: SimulationConfig, seed: int, asset_dir: str):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.blocks: list[Block] = []
        self.initial_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._bin_slots = 0
        self._create(asset_dir)

    def _create(self, asset_dir: str) -> None:
        cfg = self.config
        plane = p.loadURDF("plane.urdf")
        p.changeDynamics(plane, -1, lateralFriction=0.6)
        half = [0.10, 0.10, cfg.stand_height / 2]
        self.stand_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half),
            baseVisualShapeIndex=p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=[0.22, 0.23, 0.26, 1]
            ),
            basePosition=[cfg.tower_origin[0], cfg.tower_origin[1], cfg.stand_height / 2],
        )
        p.changeDynamics(self.stand_id, -1, lateralFriction=0.5)
        sigma = cfg.block_height_tolerance_m
        z = cfg.tower_base_z
        pitch = cfg.block_width + cfg.block_gap_m
        for layer in range(cfg.layers):
            long_axis = "x" if layer % 2 == 0 else "y"
            heights = cfg.block_height + np.clip(
                self.rng.normal(0.0, sigma, cfg.blocks_per_layer), -2 * sigma, 2 * sigma
            ) if sigma > 0 else np.full(cfg.blocks_per_layer, cfg.block_height)
            for slot in range(cfg.blocks_per_layer):
                h = float(heights[slot])
                offset = (slot - (cfg.blocks_per_layer - 1) / 2) * pitch
                jitter = self.rng.uniform(-1, 1) * cfg.block_placement_jitter_m
                if long_axis == "x":
                    pos = [cfg.tower_origin[0] + jitter, cfg.tower_origin[1] + offset, z + h / 2]
                    orn = p.getQuaternionFromEuler([0, 0, 0])
                else:
                    pos = [cfg.tower_origin[0] + offset, cfg.tower_origin[1] + jitter, z + h / 2]
                    orn = p.getQuaternionFromEuler([0, 0, math.pi / 2])
                extents = [cfg.block_length / 2, cfg.block_width / 2, h / 2]
                body = p.createMultiBody(
                    baseMass=cfg.block_mass_kg,
                    baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=extents),
                    baseVisualShapeIndex=_block_visual(extents),
                    basePosition=pos,
                    baseOrientation=orn,
                    useMaximalCoordinates=MAXIMAL_COORDINATES,
                )
                wood = _wood_colour(self.rng)
                p.changeVisualShape(body, -1, shapeIndex=0, rgbaColor=[0.16, 0.10, 0.06, 1])
                for shape in range(1, 6):
                    p.changeVisualShape(body, -1, shapeIndex=shape, rgbaColor=wood)
                p.changeDynamics(
                    body, -1,
                    lateralFriction=float(self.rng.uniform(*cfg.block_friction_range)),
                    spinningFriction=0.001,
                    rollingFriction=0.0001,
                    restitution=0.0,
                    linearDamping=0.04,
                    angularDamping=0.04,
                    # Resting blocks sleep, so the solver only works while
                    # the tower is being touched.
                    activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING,
                    sleepThreshold=0.005,
                    # Anchored friction stops the slow creep of stacked boxes.
                    frictionAnchor=1,
                )
                self.blocks.append(Block(len(self.blocks), layer, slot, long_axis, body))
            z += float(np.max(heights))

    def settle(self, steps: int = 720) -> None:
        for _ in range(steps):
            p.stepSimulation()
        for block in self.blocks:
            self.initial_poses[block.block_id] = self.pose(block)

    def pose(self, block: Block) -> tuple[np.ndarray, np.ndarray]:
        position, orientation = p.getBasePositionAndOrientation(block.body_id)
        return np.asarray(position), matrix_from_quat(orientation)

    def face(self, block: Block) -> tuple[np.ndarray, np.ndarray]:
        """True centre and outward normal of the block's push face."""
        position, rotation = self.pose(block)
        long_axis = rotation[:, 0]
        outward = np.array([1.0, 0.0, 0.0]) if block.long_axis == "x" else np.array([0.0, -1.0, 0.0])
        if np.dot(long_axis, outward) < 0:
            long_axis = -long_axis
        return position + long_axis * self.config.block_length / 2, long_axis

    def remove(self, block: Block) -> None:
        """An extracted block falls behind the tower, outside the workspace."""
        block.status = "extracted"
        self._bin_slots += 1
        p.resetBasePositionAndOrientation(
            block.body_id, [2.0 + 0.1 * self._bin_slots, 2.0, 0.02], [0, 0, 0, 1]
        )
        p.resetBaseVelocity(block.body_id, [0, 0, 0], [0, 0, 0])

    def snapshot(self) -> dict[int, np.ndarray]:
        return {
            b.block_id: self.pose(b)[0]
            for b in self.blocks if b.status != "extracted"
        }

    def has_fallen(self, block: Block) -> bool:
        """True if the block dropped or tipped over since its baseline pose."""
        position, rotation = self.pose(block)
        start, start_rotation = self.initial_poses[block.block_id]
        if start[2] - position[2] > self.config.block_height * 0.5:
            return True
        return float(np.dot(rotation[:, 2], start_rotation[:, 2])) < math.cos(math.radians(20))

    def has_collapsed(self, exclude: Optional[int] = None) -> bool:
        """True if any remaining block (other than `exclude`) fell from its place.

        A block that is only dragged a little and stays on its layer is a
        disturbance (reported as tower perturbation), not a collapse.
        """
        cfg = self.config
        for block in self.blocks:
            if block.status == "extracted" or block.block_id == exclude:
                continue
            position, _ = self.pose(block)
            start, _ = self.initial_poses[block.block_id]
            if np.linalg.norm(position[:2] - start[:2]) > cfg.block_length * 0.5:
                return True
            if self.has_fallen(block):
                return True
        return False


class TowerMap:
    """The robot's model of the tower: a nominal grid registered from vision."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.origin = np.array([config.tower_origin[0], config.tower_origin[1], config.tower_base_z])

    def nominal_face(self, layer: int, slot: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        pitch = cfg.block_width + cfg.block_gap_m
        offset = (slot - (cfg.blocks_per_layer - 1) / 2) * pitch
        z = self.origin[2] + cfg.block_height * (layer + 0.5)
        if layer % 2 == 0:
            normal = np.array([1.0, 0.0, 0.0])
            center = np.array([self.origin[0] + cfg.block_length / 2, self.origin[1] + offset, z])
        else:
            normal = np.array([0.0, -1.0, 0.0])
            center = np.array([self.origin[0] + offset, self.origin[1] - cfg.block_length / 2, z])
        return center, normal

    def associate(
        self, center: np.ndarray, normal: np.ndarray, tolerance: float = 0.010
    ) -> Optional[tuple[int, int]]:
        cfg = self.config
        if normal[0] > 0.85:
            parity = 0
        elif normal[1] < -0.85:
            parity = 1
        else:
            return None
        layer = int(round((center[2] - self.origin[2]) / cfg.block_height - 0.5))
        if layer % 2 != parity or not 0 <= layer < cfg.layers:
            return None
        pitch = cfg.block_width + cfg.block_gap_m
        lateral = center[1] - self.origin[1] if parity == 0 else center[0] - self.origin[0]
        slot = int(round(lateral / pitch + (cfg.blocks_per_layer - 1) / 2))
        if not 0 <= slot < cfg.blocks_per_layer:
            return None
        nominal, nominal_normal = self.nominal_face(layer, slot)
        residual = center - nominal
        depth = float(np.dot(residual, nominal_normal))
        # Faces are told apart in-plane; depth varies with per-block jitter,
        # half-pushed blocks and depth-sensor bias, so it is gated loosely.
        if np.linalg.norm(residual - depth * nominal_normal) > tolerance or abs(depth) > 0.03:
            return None
        return layer, slot

    def _coarse_align(self, observations: Sequence[FaceObservation]) -> None:
        """Snap the grid to the faces before fine registration.

        Camera calibration error can shift every face by more than half a
        layer, which would make association pick the wrong layer. Each face is
        compared with the nearest layer of its own orientation (every second
        layer, so +-15 mm is unambiguous) and the nearest slot, and the grid is
        moved by the median offset.
        """
        cfg = self.config
        pitch = cfg.block_width + cfg.block_gap_m
        offsets: list[list[float]] = [[], [], []]
        for obs in observations:
            if obs.world_center is None or obs.world_normal is None:
                continue
            if obs.world_normal[0] > 0.85:
                parity, lateral_axis = 0, 1
            elif obs.world_normal[1] < -0.85:
                parity, lateral_axis = 1, 0
            else:
                continue
            height = obs.world_center[2] - self.origin[2] - cfg.block_height * (parity + 0.5)
            period = 2 * cfg.block_height
            offsets[2].append(height - period * round(height / period))
            lateral = obs.world_center[lateral_axis] - self.origin[lateral_axis]
            offsets[lateral_axis].append(lateral - pitch * round(lateral / pitch))
        for axis, values in enumerate(offsets):
            if len(values) >= 3:
                self.origin[axis] += float(np.median(values))

    def register(self, observations: Sequence[FaceObservation]) -> None:
        """Refine the tower origin from the in-plane residuals of associated faces.

        Each face only measures the origin across its own plane: x-facing
        faces give y and z, y-facing faces give x and z. Depth is ignored
        because it depends on per-block jitter.
        """
        self._coarse_align(observations)
        for _ in range(2):
            per_axis: list[list[float]] = [[], [], []]
            for obs in observations:
                if obs.world_center is None:
                    continue
                key = self.associate(obs.world_center, obs.world_normal, tolerance=0.015)
                if key is None:
                    continue
                nominal, normal = self.nominal_face(*key)
                residual = obs.world_center - nominal
                for axis in range(3):
                    if abs(normal[axis]) < 0.5:
                        per_axis[axis].append(float(residual[axis]))
            if sum(len(values) for values in per_axis) < 8:
                return
            for axis, values in enumerate(per_axis):
                if len(values) >= 3:
                    self.origin[axis] += float(np.median(values))


# --------------------------------------------------------------------------
# Robot: Panda with finger, 1-D force sensor and eye-in-hand camera
# --------------------------------------------------------------------------

def _box_inertia(mass: float, size: Sequence[float]) -> str:
    x, y, z = size
    return (
        f'<inertia ixx="{mass * (y * y + z * z) / 12:.3e}" ixy="0" ixz="0" '
        f'iyy="{mass * (x * x + z * z) / 12:.3e}" iyz="0" izz="{mass * (x * x + y * y) / 12:.3e}"/>'
    )


def camera_in_tool(config: SimulationConfig) -> np.ndarray:
    """Camera pose (OpenCV axes) in the tool frame at the fingertip."""
    angle = math.atan2(config.camera_up_offset, config.camera_back_offset + config.probe_clearance)
    a, b = math.sin(angle), math.cos(angle)
    rotation = np.column_stack(((0.0, -1.0, 0.0), (b, 0.0, -a), (a, 0.0, b)))
    return make_pose(rotation, (-config.camera_up_offset, 0.0, -config.camera_back_offset))


def build_robot_urdf(config: SimulationConfig, directory: str) -> str:
    source_dir = os.path.join(pybullet_data.getDataPath(), "franka_panda")
    with open(os.path.join(source_dir, "panda.urdf"), encoding="utf-8") as stream:
        text = stream.read()
    head = text[: text.index('<joint name="panda_hand_joint"')]
    head = head.replace("package://meshes/", source_dir.replace("\\", "/") + "/meshes/")
    rod = config.finger_length - config.fingertip_length
    w, h, tip = config.probe_width, config.probe_height, config.fingertip_length
    cam = camera_in_tool(config)
    cam_xyz = cam[:3, 3] + np.array([0.0, 0.0, config.finger_length])
    cam_rpy = p.getEulerFromQuaternion(quat_from_matrix(cam[:3, :3]))
    tiny = '<inertial><origin xyz="0 0 0"/><mass value="0.0001"/><inertia ixx="1e-8" ixy="0" ixz="0" iyy="1e-8" iyz="0" izz="1e-8"/></inertial>'
    tail = f"""
  <link name="finger_mount">
    <inertial><origin xyz="0 0 {rod / 2}"/><mass value="0.05"/>{_box_inertia(0.05, (h, w, rod))}</inertial>
    <visual><origin xyz="0 0 {rod / 2}"/><geometry><box size="{h} {w} {rod}"/></geometry>
      <material name="finger"><color rgba="0.55 0.6 0.68 1"/></material></visual>
    <collision><origin xyz="0 0 {rod / 2}"/><geometry><box size="{h} {w} {rod}"/></geometry></collision>
    <visual><origin xyz="{cam_xyz[0] - 0.012} 0 {cam_xyz[2] - 0.02}"/><geometry><box size="0.025 0.09 0.025"/></geometry>
      <material name="camera"><color rgba="0.08 0.08 0.1 1"/></material></visual>
  </link>
  <joint name="finger_mount_joint" type="fixed">
    <parent link="panda_link8"/><child link="finger_mount"/><origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <link name="fingertip">
    <inertial><origin xyz="0 0 {tip / 2}"/><mass value="{config.fingertip_mass_kg}"/>{_box_inertia(config.fingertip_mass_kg, (h, w, tip))}</inertial>
    <visual><origin xyz="0 0 {tip / 2}"/><geometry><box size="{h} {w} {tip}"/></geometry>
      <material name="tip"><color rgba="0.1 0.7 0.9 1"/></material></visual>
    <collision><origin xyz="0 0 {tip / 2}"/><geometry><box size="{h} {w} {tip}"/></geometry></collision>
  </link>
  <joint name="force_sensor_joint" type="prismatic">
    <parent link="finger_mount"/><child link="fingertip"/>
    <origin xyz="0 0 {rod}" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-0.008" upper="0.0005" effort="50" velocity="1.0"/>
  </joint>
  <link name="tool_frame">{tiny}</link>
  <joint name="tool_joint" type="fixed">
    <parent link="finger_mount"/><child link="tool_frame"/><origin xyz="0 0 {config.finger_length}" rpy="0 0 0"/>
  </joint>
  <link name="camera_link">{tiny}</link>
  <joint name="camera_joint" type="fixed">
    <parent link="finger_mount"/><child link="camera_link"/>
    <origin xyz="{cam_xyz[0]} {cam_xyz[1]} {cam_xyz[2]}" rpy="{cam_rpy[0]} {cam_rpy[1]} {cam_rpy[2]}"/>
  </joint>
</robot>
"""
    path = os.path.join(directory, "panda_jenga_finger.urdf")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(head + tail)
    return path


class PandaArm:
    """The simulated (true) Panda plus a kinematic model used by the controller.

    The controller reads joint encoders with a constant per-joint offset
    (kinematic calibration error), so its forward kinematics, IK and camera
    pose differ slightly from reality. The camera image is always rendered
    from the true camera pose.
    """

    MODEL_OFFSET = np.array([0.0, 0.0, -5.0])
    READY = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.8]

    def __init__(self, config: SimulationConfig, urdf_path: str, rng: np.random.Generator):
        self.config = config
        base = np.asarray(config.robot_base, dtype=float)
        yaw = math.atan2(config.tower_origin[1] - base[1], config.tower_origin[0] - base[0])
        orientation = p.getQuaternionFromEuler([0, 0, yaw])
        self.base_rotation = matrix_from_quat(orientation)
        self.robot_id = p.loadURDF(
            urdf_path, base.tolist(), orientation, useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )
        self.model_id = p.loadURDF(
            urdf_path, (base + self.MODEL_OFFSET).tolist(), orientation, useFixedBase=True,
        )
        links = {
            p.getJointInfo(self.robot_id, j)[12].decode(): j
            for j in range(p.getNumJoints(self.robot_id))
        }
        self.arm_joints = list(range(7))
        self.tool_link = links["tool_frame"]
        self.camera_link = links["camera_link"]
        self.tip_link = links["fingertip"]
        self.sensor_joint = links["fingertip"]  # joint index == child link index
        self.movable = [
            j for j in range(p.getNumJoints(self.robot_id))
            if p.getJointInfo(self.robot_id, j)[2] != p.JOINT_FIXED
        ]
        info = [p.getJointInfo(self.robot_id, j) for j in self.arm_joints]
        self.lower = np.array([i[8] for i in info])
        self.upper = np.array([i[9] for i in info])
        self.efforts = [float(i[10]) for i in info]
        self.max_velocity = np.array([i[11] for i in info])
        self.joint_axes = [np.asarray(i[13], dtype=float) for i in info]
        self.posture = np.asarray(self.READY, dtype=float)  # null-space target
        self.sensor_offset = 0.0
        self.bias = rng.normal(0.0, config.joint_bias_std_rad, 7)
        self.rng = rng
        for link in range(-1, p.getNumJoints(self.model_id)):
            p.setCollisionFilterGroupMask(self.model_id, link, 0, 0)
            p.changeVisualShape(self.model_id, link, rgbaColor=[0, 0, 0, 0])
        for body in (self.robot_id, self.model_id):
            p.setJointMotorControl2(body, self.sensor_joint, p.VELOCITY_CONTROL, force=0)
        p.changeDynamics(self.robot_id, self.tip_link, lateralFriction=0.3)
        mass = config.fingertip_mass_kg
        self.damping = 2.0 * config.sensor_damping_ratio * math.sqrt(config.sensor_stiffness_n_m * mass)
        for joint, value in zip(self.arm_joints, self.READY):
            p.resetJointState(self.robot_id, joint, value - self.bias[joint])
        self.command_positions(np.asarray(self.READY))

    # ----- true robot -------------------------------------------------------
    def q_true(self) -> np.ndarray:
        return np.array([s[0] for s in p.getJointStates(self.robot_id, self.arm_joints)])

    def q_meas(self) -> np.ndarray:
        return self.q_true() + self.bias

    def true_link_pose(self, link: int) -> np.ndarray:
        state = p.getLinkState(self.robot_id, link, computeForwardKinematics=True)
        return make_pose(matrix_from_quat(state[5]), state[4])

    def pre_step(self) -> None:
        """Spring-damper of the 1-D force sensor (applied every physics step)."""
        position, velocity = p.getJointState(self.robot_id, self.sensor_joint)[:2]
        force = -self.config.sensor_stiffness_n_m * position - self.damping * velocity
        p.setJointMotorControl2(self.robot_id, self.sensor_joint, p.TORQUE_CONTROL, force=force)

    def raw_sensor_force(self) -> float:
        position = p.getJointState(self.robot_id, self.sensor_joint)[0]
        return -self.config.sensor_stiffness_n_m * position

    def tare(self) -> None:
        """Zero the sensor (removes the fingertip weight) while not in contact."""
        self.sensor_offset = self.raw_sensor_force()

    def sensor_force(self) -> float:
        """Sensor reading: tared spring force, 12-bit quantised, 5 N full scale."""
        return quantize_force(self.raw_sensor_force() - self.sensor_offset, self.config)

    def contact_force(self) -> float:
        return float(sum(
            c[9] for c in p.getContactPoints(bodyA=self.robot_id, linkIndexA=self.tip_link)
        ))

    # ----- controller-side kinematic model ---------------------------------
    def _set_model(self, q: Sequence[float]) -> None:
        for joint, value in zip(self.arm_joints, q):
            p.resetJointState(self.model_id, joint, float(value))

    def model_link_pose(self, link: int, q: Optional[Sequence[float]] = None) -> np.ndarray:
        self._set_model(self.q_meas() if q is None else q)
        state = p.getLinkState(self.model_id, link, computeForwardKinematics=True)
        return make_pose(matrix_from_quat(state[5]), np.asarray(state[4]) - self.MODEL_OFFSET)

    def tool_pose(self, q: Optional[Sequence[float]] = None) -> np.ndarray:
        return self.model_link_pose(self.tool_link, q)

    def camera_pose(self, q: Optional[Sequence[float]] = None) -> np.ndarray:
        return self.model_link_pose(self.camera_link, q)

    def pose_and_jacobian(self, q: Optional[Sequence[float]] = None) -> tuple[np.ndarray, np.ndarray]:
        """Tool pose and geometric Jacobian (world frame, 6 x 7) of the model.

        Column i = [z_i x (p_tool - p_i); z_i] with z_i the world axis of
        revolute joint i, read from the model's link frames.
        """
        q = self.q_meas() if q is None else np.asarray(q)
        self._set_model(q)
        states = p.getLinkStates(self.model_id, self.arm_joints + [self.tool_link],
                                 computeForwardKinematics=True)
        tool_position = np.asarray(states[-1][4])
        jacobian = np.zeros((6, 7))
        for index in range(7):
            axis = matrix_from_quat(states[index][5]) @ self.joint_axes[index]
            jacobian[:3, index] = np.cross(axis, tool_position - np.asarray(states[index][4]))
            jacobian[3:, index] = axis
        pose = make_pose(matrix_from_quat(states[-1][5]), tool_position - self.MODEL_OFFSET)
        return pose, jacobian

    def jacobian(self, q: Optional[Sequence[float]] = None) -> np.ndarray:
        return self.pose_and_jacobian(q)[1]

    def ik(self, target: np.ndarray, attempts: int = 12) -> Optional[np.ndarray]:
        """Damped least-squares IK on the controller's model, within joint limits.

        Several seeds are tried; the valid solution closest to the current
        configuration is returned (shortest joint-space motion).
        """
        current = self.q_meas()
        seeds = [current, np.asarray(self.READY, dtype=float)]
        # PyBullet's unconstrained IK from random starts gives diverse seeds;
        # the damped least-squares loop below enforces the joint limits.
        position = (target[:3, 3] + self.MODEL_OFFSET).tolist()
        orientation = quat_from_matrix(target[:3, :3])
        for _ in range(attempts):
            self._set_model(self.rng.uniform(self.lower, self.upper))
            solution = p.calculateInverseKinematics(
                self.model_id, self.tool_link, position, orientation,
                maxNumIterations=100, residualThreshold=1e-6,
            )
            seeds.append((np.asarray(solution[:7]) + math.pi) % (2 * math.pi) - math.pi)
        margin = 0.02
        best: Optional[tuple[float, np.ndarray]] = None
        for index, seed in enumerate(seeds):
            q = np.clip(np.asarray(seed, dtype=float), self.lower + margin, self.upper - margin)
            previous = math.inf
            for iteration in range(100):
                pose, jacobian = self.pose_and_jacobian(q)
                error = np.concatenate((
                    target[:3, 3] - pose[:3, 3],
                    rotation_vector(target[:3, :3] @ pose[:3, :3].T),
                ))
                size = float(np.linalg.norm(error[:3]) + 0.1 * np.linalg.norm(error[3:]))
                if size < 1e-5:
                    break
                if iteration % 20 == 19:
                    if size > 0.9 * previous:
                        break  # stalled against a joint limit
                    previous = size
                pinv = damped_pinv(jacobian, 0.02)
                mid = (self.lower + self.upper) / 2
                null = np.eye(7) - np.linalg.pinv(jacobian) @ jacobian
                posture = null @ (0.05 * (mid - q))
                step = pinv @ error + posture
                step *= min(1.0, 0.3 / max(float(np.max(np.abs(step))), 1e-12))
                q = np.clip(q + step, self.lower + margin, self.upper - margin)
            pose = self.tool_pose(q)
            position_error = float(np.linalg.norm(pose[:3, 3] - target[:3, 3]))
            angle_error = float(np.linalg.norm(rotation_vector(target[:3, :3] @ pose[:3, :3].T)))
            if position_error < 2e-4 and angle_error < 2e-3:
                cost = float(np.linalg.norm(q - current))
                if best is None or cost < best[0]:
                    best = (cost, q)
                if index < 2:
                    break  # a solution near the current or ready pose is preferred
        return None if best is None else best[1]

    def command_positions(self, q_meas: np.ndarray, qd: Optional[np.ndarray] = None) -> None:
        p.setJointMotorControlArray(
            self.robot_id, self.arm_joints, p.POSITION_CONTROL,
            targetPositions=(np.asarray(q_meas) - self.bias).tolist(),
            targetVelocities=(np.zeros(7) if qd is None else np.asarray(qd)).tolist(),
            forces=self.efforts,
            positionGains=[0.1] * 7,
            velocityGains=[1.0] * 7,
        )

    def command_velocities(self, qd: np.ndarray) -> None:
        p.setJointMotorControlArray(
            self.robot_id, self.arm_joints, p.VELOCITY_CONTROL,
            targetVelocities=np.asarray(qd).tolist(), forces=self.efforts,
        )

    def twist_to_joint_velocity(self, twist: np.ndarray, speed_scale: float = 0.5) -> np.ndarray:
        """Resolved-rate control: q_dot = J+ v plus a null-space posture term."""
        q = self.q_meas()
        jacobian = self.jacobian(q)
        pinv = damped_pinv(jacobian)
        null = (np.eye(7) - np.linalg.pinv(jacobian) @ jacobian) @ (0.3 * (self.posture - q))
        qd = pinv @ twist + null
        limit = self.max_velocity * speed_scale
        scale = float(np.max(np.abs(qd) / limit))
        if scale > 1.0:
            qd /= scale
        near_lower = (q <= self.lower + 0.02) & (qd < 0)
        near_upper = (q >= self.upper - 0.02) & (qd > 0)
        qd[near_lower | near_upper] = 0.0
        return qd


# --------------------------------------------------------------------------
# Eye-in-hand RGB-D camera and block detectors
# --------------------------------------------------------------------------

class Vision:
    def __init__(self, config: SimulationConfig, tower: Tower, rng: np.random.Generator):
        self.config = config
        self.tower = tower
        self.rng = rng
        self.focal = config.camera_height / (2 * math.tan(math.radians(config.camera_fov) / 2))
        self.camera_matrix = np.array([
            [self.focal, 0, config.camera_width / 2],
            [0, self.focal, config.camera_height / 2],
            [0, 0, 1],
        ])
        self.projection = p.computeProjectionMatrixFOV(
            config.camera_fov, config.camera_width / config.camera_height,
            config.camera_near, config.camera_far,
        )
        self.renders = 0
        self.depth_scale = 1.0 + float(rng.normal(0.0, config.depth_scale_error))

    def render(self, camera_pose: np.ndarray, segmentation: bool) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        cfg = self.config
        eye = camera_pose[:3, 3]
        view = p.computeViewMatrix(
            eye.tolist(), (eye + camera_pose[:3, 2]).tolist(), (-camera_pose[:3, 1]).tolist()
        )
        image = p.getCameraImage(
            cfg.camera_width, cfg.camera_height, view, self.projection,
            renderer=p.ER_TINY_RENDERER,
            lightDirection=[0.6, -0.5, 1.0], lightAmbientCoeff=0.6,
            lightDiffuseCoeff=0.45, lightSpecularCoeff=0.0, shadow=0,
            flags=0 if segmentation else p.ER_NO_SEGMENTATION_MASK,
        )
        self.renders += 1
        shape = (cfg.camera_height, cfg.camera_width)
        rgb = np.reshape(np.asarray(image[2], dtype=np.uint8), shape + (4,))[:, :, :3].copy()
        buffer = np.reshape(np.asarray(image[3], dtype=float), shape)
        near, far = cfg.camera_near, cfg.camera_far
        depth = far * near / (far - (far - near) * buffer)
        background = buffer >= 0.9999
        depth = depth * self.depth_scale
        if cfg.depth_noise_m > 0:
            depth = depth + self.rng.normal(0.0, 1.0, shape) * cfg.depth_noise_m * (1 + depth ** 2)
        depth[background] = np.inf
        seg = np.reshape(np.asarray(image[4]), shape) if segmentation else None
        return rgb, depth, seg

    def project(self, point_cam: np.ndarray) -> Optional[tuple[float, float]]:
        if point_cam[2] <= 1e-6:
            return None
        u = self.focal * point_cam[0] / point_cam[2] + self.config.camera_width / 2
        v = self.focal * point_cam[1] / point_cam[2] + self.config.camera_height / 2
        return float(u), float(v)

    def back_project(self, us: np.ndarray, vs: np.ndarray, depth: np.ndarray) -> np.ndarray:
        x = (us + 0.5 - self.config.camera_width / 2) * depth / self.focal
        y = (vs + 0.5 - self.config.camera_height / 2) * depth / self.focal
        return np.column_stack((x, y, depth))

    # ----- ground truth ---------------------------------------------------
    def detect_ground_truth(self, true_camera: np.ndarray, roi: Optional[tuple[float, float, float]] = None) -> list[FaceObservation]:
        _, depth, seg = self.render(true_camera, segmentation=True)
        world_to_cam = invert_pose(true_camera)
        observations = []
        ids, counts = np.unique(seg[seg >= 0], return_counts=True)
        by_body = {b.body_id: b for b in self.tower.blocks if b.status != "extracted"}
        for body, count in zip(ids, counts):
            block = by_body.get(int(body))
            if block is None or count < 15:
                continue
            center, normal = self.tower.face(block)
            center_cam = world_to_cam[:3, :3] @ center + world_to_cam[:3, 3]
            normal_cam = world_to_cam[:3, :3] @ normal
            if np.dot(normal_cam, center_cam) >= 0:
                continue
            pixel = self.project(center_cam)
            if pixel is None:
                continue
            u, v = int(pixel[0]), int(pixel[1])
            if not (0 <= u < self.config.camera_width and 0 <= v < self.config.camera_height):
                continue
            if roi is not None and math.hypot(pixel[0] - roi[0], pixel[1] - roi[1]) > roi[2]:
                continue
            # Visible only if the face centre is not occluded.
            if seg[v, u] != block.body_id:
                continue
            rotation = world_to_cam[:3, :3] @ face_frame(normal)
            observations.append(FaceObservation(
                center_cam, rotation, pixel, int(count), block_id=block.block_id,
            ))
        return observations

    # ----- classical OpenCV pipeline ----------------------------------------
    def detect_opencv(self, true_camera: np.ndarray, model_camera: np.ndarray,
                      roi: Optional[tuple[float, float, float]] = None) -> list[FaceObservation]:
        cfg = self.config
        rgb, depth, _ = self.render(true_camera, segmentation=False)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([5, 70, 95], np.uint8), np.array([30, 255, 255], np.uint8))
        finite = np.isfinite(depth)
        safe_depth = np.where(finite, depth, 10.0).astype(np.float32)
        grad = np.hypot(cv2.Sobel(safe_depth, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(safe_depth, cv2.CV_32F, 0, 1, ksize=3))
        mask[(grad > 0.02) | ~finite] = 0
        mask = cv2.erode(mask, np.ones((2, 2), np.uint8))
        if roi is not None:
            window = np.zeros_like(mask)
            cv2.circle(window, (int(roi[0]), int(roi[1])), int(roi[2]), 255, -1)
            mask &= window
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        up_cam = model_camera[:3, :3].T @ WORLD_UP
        observations = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 25:
                continue
            x0, y0, w, h = stats[label, :4]
            local = labels[y0:y0 + h, x0:x0 + w] == label
            vs, us = np.nonzero(local)
            us, vs = us + x0, vs + y0
            observation = self._estimate_face(us, vs, depth[vs, us], up_cam)
            if observation is not None:
                observations.append(observation)
        return observations

    def _estimate_face(self, us: np.ndarray, vs: np.ndarray, z: np.ndarray, up_cam: np.ndarray) -> Optional[FaceObservation]:
        """PnP on the face corners (initial pose), refined with the depth points."""
        cfg = self.config
        # The dark seam trims the visible wood on every side.
        visible_w = cfg.block_width - 2 * SEAM_INSET
        visible_h = cfg.block_height - 2 * SEAM_INSET
        pnp = self._pnp(us, vs, visible_w, visible_h)
        valid = np.isfinite(z)
        points = self.back_project(us[valid].astype(float), vs[valid].astype(float), z[valid])
        refined = self._refine_with_depth(points, up_cam) if len(points) >= 20 else None
        if refined is not None:
            center, rotation, width, height = refined
        elif pnp is not None:
            rotation, center = pnp
            width, height = visible_w, visible_h
        else:
            return None
        # End faces are 25 x 15 mm; long side faces (75 mm) are rejected.
        if not (0.55 * visible_w <= width <= 1.3 * visible_w):
            return None
        if not (0.5 * visible_h <= height <= 1.35 * visible_h):
            return None
        pixel = self.project(center)
        if pixel is None:
            return None
        return FaceObservation(center, rotation, pixel, int(len(us)))

    @staticmethod
    def _refine_with_depth(points: np.ndarray, up_cam: np.ndarray):
        centroid = points.mean(axis=0)
        normal = np.linalg.svd(points - centroid, full_matrices=False)[2][2]
        inliers = np.abs((points - centroid) @ normal) < 0.0015
        if inliers.sum() < 15:
            return None
        points = points[inliers]
        centroid = points.mean(axis=0)
        normal = np.linalg.svd(points - centroid, full_matrices=False)[2][2]
        if np.dot(normal, centroid) > 0:
            normal = -normal
        y_axis = up_cam - normal * float(np.dot(up_cam, normal))
        if np.linalg.norm(y_axis) < 0.5:
            return None
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, normal)
        local = points - centroid
        x_lo, x_hi = np.percentile(local @ x_axis, [1, 99])
        y_lo, y_hi = np.percentile(local @ y_axis, [1, 99])
        center = centroid + x_axis * (x_lo + x_hi) / 2 + y_axis * (y_lo + y_hi) / 2
        return center, np.column_stack((x_axis, y_axis, normal)), x_hi - x_lo, y_hi - y_lo

    def _pnp(self, us: np.ndarray, vs: np.ndarray, width: float, height: float) -> Optional[tuple[np.ndarray, np.ndarray]]:
        hull = cv2.convexHull(np.column_stack((us, vs)).astype(np.float32))
        corners = cv2.boxPoints(cv2.minAreaRect(hull))
        order = np.argsort(corners[:, 1])
        top = corners[order[:2]][np.argsort(corners[order[:2], 0])]
        bottom = corners[order[2:]][np.argsort(corners[order[2:], 0])]
        image_points = np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)
        object_points = np.array([
            [-width / 2, height / 2, 0], [width / 2, height / 2, 0],
            [width / 2, -height / 2, 0], [-width / 2, -height / 2, 0],
        ])
        try:
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix,
                                          np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            return None
        if not ok:
            return None
        return cv2.Rodrigues(rvec)[0], tvec.ravel()

    def detect(self, mode: str, true_camera: np.ndarray, model_camera: np.ndarray,
               roi: Optional[tuple[float, float, float]] = None) -> list[FaceObservation]:
        if mode == "ground_truth":
            observations = self.detect_ground_truth(true_camera, roi)
        elif mode == "opencv":
            observations = self.detect_opencv(true_camera, model_camera, roi)
        else:
            raise ValueError(f"Unknown detector mode: {mode}")
        for obs in observations:
            obs.world_center = model_camera[:3, :3] @ obs.center_cam + model_camera[:3, 3]
            obs.world_normal = model_camera[:3, :3] @ obs.rotation_cam[:, 2]
        return observations


# --------------------------------------------------------------------------
# Simulation / game
# --------------------------------------------------------------------------

class JengaSimulation:
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        detector: str = "ground_truth",
        controller: str = "closed_loop",
        seed: int = 0,
        gui: bool = False,
    ):
        if detector not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}")
        if controller not in CONTROLLERS:
            raise ValueError(f"controller must be one of {CONTROLLERS}")
        self.config = config or SimulationConfig()
        self.detector_mode = detector
        self.controller = controller
        self.seed = seed
        self.gui = gui
        if p.isConnected():
            raise RuntimeError("Only one JengaSimulation may be open per process")
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self.asset_dir = tempfile.mkdtemp(prefix="jenga_")
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.config.time_step,
            numSolverIterations=self.config.solver_iterations,
            deterministicOverlappingPairs=1,
            # Motions are slow, so contacts are only created within 3 mm; this
            # also keeps the resting tower asleep while the finger hovers.
            contactBreakingThreshold=0.003,
        )
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(0.75, 45, -20, [0.1, -0.1, 0.25])
        # Separate streams so detector/controller choices do not change the
        # tower or the robot calibration error of a given seed.
        self.tower = Tower(self.config, seed, self.asset_dir)
        urdf = build_robot_urdf(self.config, self.asset_dir)
        self.arm = PandaArm(self.config, urdf, np.random.default_rng(seed + 10_000))
        self.tower.settle()
        self.vision = Vision(self.config, self.tower, np.random.default_rng(seed + 20_000))
        self.map = TowerMap(self.config)
        self.policy = BlockSelectionPolicy(self.tower.blocks, self.config, seed)
        self.attempts: list[AttemptResult] = []
        self.trial_index = 0
        self.sim_time = 0.0
        self.camera_in_tool = camera_in_tool(self.config)
        self.tool_in_camera = invert_pose(self.camera_in_tool)
        self.sensor_steps = max(1, int(round(1.0 / (self.config.sensor_hz * self.config.time_step))))
        self.servo_steps = max(1, int(round(1.0 / (self.config.servo_hz * self.config.time_step))))
        self._overview_q: Optional[np.ndarray] = None

    # ----- low level --------------------------------------------------------
    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.arm.pre_step()
            p.stepSimulation()
            self.sim_time += self.config.time_step
            if self.gui:
                time.sleep(self.config.time_step)

    def _tool_target(self, center: np.ndarray, normal: np.ndarray, clearance: float) -> np.ndarray:
        return make_pose(face_frame(normal), center) @ tool_in_face(clearance)

    def _move_joints(self, q_goal: np.ndarray, trajectory: list) -> None:
        """Trapezoidal joint-space trajectory (stage 3)."""
        q_start = self.arm.q_meas()
        delta = q_goal - q_start
        duration = max(0.4, 1.25 * float(np.max(np.abs(delta))) / self.config.max_joint_speed_rad_s)
        steps = max(2, int(duration / self.config.time_step))
        profile = trapezoid_profile(steps)
        velocity = np.gradient(profile) / self.config.time_step
        for index, (s, ds) in enumerate(zip(profile, velocity)):
            self.arm.command_positions(q_start + s * delta, ds * delta)
            self.step()
            if index % 12 == 0:
                trajectory.append(tuple(self.arm.true_link_pose(self.arm.tool_link)[:3, 3]))
        self.arm.command_positions(q_goal)
        self.step(30)

    def _move_linear(self, goal: np.ndarray, speed: float, trajectory: list,
                     monitor=None) -> Optional[str]:
        """Straight-line Cartesian motion with resolved-rate control.

        `monitor(step_index)` may return a string to stop the motion early.
        """
        start = self.arm.tool_pose()
        self.arm.posture = self.arm.q_meas()
        path = goal[:3, 3] - start[:3, 3]
        distance = float(np.linalg.norm(path))
        direction = path / distance if distance > 1e-9 else np.zeros(3)
        steps = max(2, int(distance / speed / self.config.time_step))
        profile = trapezoid_profile(steps, 0.1)
        rate = np.gradient(profile) / self.config.time_step * distance
        settle = int(0.4 / self.config.time_step)
        for index in range(steps + settle):
            s = profile[min(index, steps - 1)]
            ds = rate[index] if index < steps else 0.0
            pose = self.arm.tool_pose()
            reference = start[:3, 3] + direction * distance * s
            linear = direction * ds + 5.0 * (reference - pose[:3, 3])
            angular = 4.0 * rotation_vector(goal[:3, :3] @ pose[:3, :3].T)
            self.arm.command_velocities(self.arm.twist_to_joint_velocity(np.concatenate((linear, angular))))
            self.step()
            if index % 12 == 0:
                trajectory.append(tuple(self.arm.true_link_pose(self.arm.tool_link)[:3, 3]))
            if monitor is not None:
                reason = monitor(index)
                if reason is not None:
                    self.arm.command_velocities(np.zeros(7))
                    return reason
            if index >= steps and np.linalg.norm(pose[:3, 3] - goal[:3, 3]) < 2e-4:
                break
        self.arm.command_velocities(np.zeros(7))
        return None

    def _hold(self, seconds: float) -> None:
        self.arm.command_velocities(np.zeros(7))
        self.step(int(seconds / self.config.time_step))

    # ----- perception -------------------------------------------------------
    def _detect(self, roi=None) -> list[FaceObservation]:
        true_camera = self.arm.true_link_pose(self.arm.camera_link)
        model_camera = self.arm.camera_pose()
        observations = self.vision.detect(self.detector_mode, true_camera, model_camera, roi)
        for obs in observations:
            if obs.block_id is None:
                key = self.map.associate(obs.world_center, obs.world_normal)
                if key is not None:
                    obs.block_id = key[0] * self.config.blocks_per_layer + key[1]
        return observations

    def _overview_pose(self, distance: Optional[float] = None, height: float = 0.0) -> np.ndarray:
        """Tool pose that puts the camera on the tower diagonal, seeing both
        pushable faces at once (as in the paper's dataset views)."""
        cfg = self.config
        corner = np.array([
            cfg.tower_origin[0] + cfg.block_length / 2,
            cfg.tower_origin[1] - cfg.block_length / 2,
            cfg.tower_base_z + cfg.block_height * (cfg.min_level + cfg.max_level + 1) / 2,
        ])
        diagonal = np.array([math.sqrt(0.5), -math.sqrt(0.5), 0.0])
        distance = cfg.overview_distance if distance is None else distance
        eye = corner + diagonal * distance + np.array([0.0, 0.0, height])
        z_axis = corner - eye
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.cross(z_axis, WORLD_UP)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        camera = make_pose(np.column_stack((x_axis, y_axis, z_axis)), eye)
        return camera @ self.tool_in_camera

    def overview(self, trajectory: Optional[list] = None) -> dict[int, FaceObservation]:
        """Move to the overview pose, register the tower map, list candidates."""
        trajectory = [] if trajectory is None else trajectory
        if self._overview_q is None:
            for distance, height in ((0.30, 0.0), (0.30, 0.10), (0.30, 0.25), (0.38, 0.0), (0.45, 0.0)):
                self._overview_q = self.arm.ik(self._overview_pose(distance, height))
                if self._overview_q is not None:
                    break
            else:
                raise RuntimeError("No overview pose is reachable")
        self._move_joints(self._overview_q, trajectory)
        self._hold(0.2)
        observations = self._detect()
        if self.detector_mode == "opencv":
            self.map.register(observations)
            for obs in observations:
                key = self.map.associate(obs.world_center, obs.world_normal)
                obs.block_id = None if key is None else key[0] * self.config.blocks_per_layer + key[1]
        return {obs.block_id: obs for obs in observations if obs.block_id is not None}

    # ----- stage 4: alignment -------------------------------------------------
    def _desired_face_in_camera(self) -> np.ndarray:
        face_in_tool = invert_pose(tool_in_face(self.config.probe_clearance))
        return self.tool_in_camera @ face_in_tool

    def _track(self, block: Block, predicted_world: np.ndarray, locked: bool = True) -> Optional[FaceObservation]:
        """Find the target face in the close-up image.

        The first match (`locked=False`) cannot compare world positions with
        the overview estimate: both depend on the arm's calibration error,
        which differs between the two poses. Like the overview, it registers
        the tower map from the close-up image and picks the face associated
        with the target's level and slot. Once locked, the prediction is the
        last close-up detection and a tight gate keeps the track continuous.
        """
        model_camera = self.arm.camera_pose()
        if not locked and self.detector_mode == "opencv":
            observations = self._detect()
            local_map = TowerMap(self.config)
            local_map.origin = self.map.origin.copy()
            local_map.register(observations)
            for obs in observations:
                if local_map.associate(obs.world_center, obs.world_normal) == (block.layer, block.slot):
                    return obs
            return None
        predicted_cam = invert_pose(model_camera) @ np.append(predicted_world, 1.0)
        pixel = self.vision.project(predicted_cam[:3])
        roi = None if pixel is None else (pixel[0], pixel[1], 90.0)
        observations = self._detect(roi)
        best, best_distance = None, 0.012
        for obs in observations:
            if self.detector_mode == "ground_truth" and obs.block_id != block.block_id:
                continue
            distance = float(np.linalg.norm(obs.world_center - predicted_world))
            if distance < best_distance:
                best, best_distance = obs, distance
        return best

    def _servo(self, block: Block, estimate: np.ndarray, trajectory: list) -> tuple[Optional[str], float, int]:
        """2 1/2-D visual servoing at ~9 Hz until the fingertip is aligned."""
        cfg = self.config
        desired = self._desired_face_in_camera()
        self.arm.posture = self.arm.q_meas()
        start_time = self.sim_time
        misses = 0
        iterations = 0
        locked = False
        twist_world = np.zeros(6)
        while self.sim_time - start_time < cfg.max_servo_seconds:
            iterations += 1
            obs = self._track(block, estimate, locked)
            if obs is None:
                misses += 1
                if misses > cfg.max_tracking_misses:
                    self._hold(0.1)
                    return "tracking_lost", self.sim_time - start_time, iterations
                twist_world = np.zeros(6)
            else:
                misses = 0
                locked = True
                estimate = obs.world_center
                current = make_pose(obs.rotation_cam, obs.center_cam)
                camera_twist, _ = visual_servo_twist(current, desired, cfg.servo_gain)
                # Converged when the estimated fingertip pose error is small.
                tool_error = self.camera_in_tool @ current @ invert_pose(desired) @ self.tool_in_camera
                lateral = float(np.linalg.norm(tool_error[:2, 3]))
                axial = abs(float(tool_error[2, 3]))
                angle = float(np.linalg.norm(rotation_vector(tool_error[:3, :3])))
                if lateral < cfg.servo_tolerance_m and axial < 2 * cfg.servo_tolerance_m \
                        and angle < cfg.servo_tolerance_rad:
                    self._hold(0.15)
                    return None, self.sim_time - start_time, iterations
                linear = camera_twist[:3]
                angular = camera_twist[3:]
                linear *= min(1.0, cfg.servo_max_linear_m_s / max(np.linalg.norm(linear), 1e-12))
                angular *= min(1.0, cfg.servo_max_angular_rad_s / max(np.linalg.norm(angular), 1e-12))
                camera = self.arm.camera_pose()
                omega = camera[:3, :3] @ angular
                v_camera = camera[:3, :3] @ linear
                tool_position = self.arm.tool_pose()[:3, 3]
                twist_world = np.concatenate((
                    v_camera + np.cross(omega, tool_position - camera[:3, 3]), omega
                ))
            # Motion-rate control at the physics rate between visual updates.
            for index in range(self.servo_steps):
                self.arm.command_velocities(self.arm.twist_to_joint_velocity(twist_world))
                self.step()
            trajectory.append(tuple(self.arm.true_link_pose(self.arm.tool_link)[:3, 3]))
        self._hold(0.1)
        return "servo_timeout", self.sim_time - start_time, iterations

    def _alignment_error(self, block: Block) -> tuple[float, float, float]:
        """Distance from the fingertip axis to the true face centre (mm)."""
        center, normal = self.tower.face(block)
        tool = self.arm.true_link_pose(self.arm.tool_link)
        axis = tool[:3, 2]
        denominator = float(np.dot(axis, -normal))
        if denominator < 1e-3:
            return float("inf"), float("inf"), float("inf")
        distance = float(np.dot(tool[:3, 3] - center, normal)) / denominator
        contact = tool[:3, 3] + axis * distance
        offset = contact - center
        frame = face_frame(normal)
        horizontal = float(np.dot(offset, frame[:, 0])) * 1000.0
        vertical = float(np.dot(offset, frame[:, 1])) * 1000.0
        return math.hypot(horizontal, vertical), horizontal, vertical

    # ----- push primitive -----------------------------------------------------
    def _push(self, block: Block, threshold: float, trajectory: list) -> tuple[str, str, float, list, float]:
        cfg = self.config
        tool = self.arm.tool_pose()
        travel = cfg.probe_clearance + cfg.block_length + cfg.push_overtravel
        goal = make_pose(tool[:3, :3], tool[:3, 3] + tool[:3, 2] * travel)
        start_center, _ = self.tower.pose(block)
        push_axis = -self.tower.face(block)[1]
        self.arm.tare()
        profile: list[tuple[float, float]] = []
        state = {"peak": 0.0, "displacement": 0.0}
        t0 = self.sim_time

        def monitor(index: int) -> Optional[str]:
            position, _ = self.tower.pose(block)
            state["displacement"] = float(np.dot(position - start_center, push_axis))
            if state["displacement"] >= cfg.block_length + 0.001:
                return "extracted"
            if index % self.sensor_steps == 0:
                force = self.arm.sensor_force()
                profile.append((self.sim_time - t0, force))
                # The threshold classifies the block while it starts to move.
                # Once it has slid `classification_travel` it is mobile; near
                # the end it tips against the level above and friction rises,
                # so only a higher safety limit applies.
                if state["displacement"] < cfg.classification_travel:
                    state["peak"] = max(state["peak"], force)
                    if force >= threshold:
                        return "immobile"
                elif force >= cfg.push_safety_force:
                    return "jammed"
            return None

        reason = self._move_linear(goal, cfg.push_speed_m_s, trajectory, monitor)
        if reason == "extracted":
            self.tower.remove(block)
            return "extracted", "", state["peak"], profile, state["displacement"]
        if reason == "immobile":
            # The paper scores an immobile call as correct only if the arm was
            # really testing the target. If the finger is resting on another
            # block (e.g. the level above after a misaligned approach), the
            # force came from the wrong block and the call is an error.
            touched = {
                c[2] for c in p.getContactPoints(bodyA=self.arm.robot_id)
                if c[2] != self.tower.stand_id and c[9] > 1e-3
            }
            if touched - {block.body_id}:
                return "error", "wrong_block_contact", state["peak"], profile, state["displacement"]
            return "immobile", "", state["peak"], profile, state["displacement"]
        if reason == "jammed":
            return "error", "push_jammed", state["peak"], profile, state["displacement"]
        return "error", "push_incomplete", state["peak"], profile, state["displacement"]

    # ----- one attempt ------------------------------------------------------
    def run_attempt(self) -> Optional[AttemptResult]:
        cfg = self.config
        start_time = self.sim_time
        trajectory: list[tuple[float, float, float]] = []
        candidates = self.overview(trajectory)
        block = self.policy.next_block(set(candidates))
        if block is None:
            return None
        phase = self.policy.phase
        threshold = self.policy.force_threshold
        before = self.tower.snapshot()
        estimate = candidates[block.block_id]
        center = estimate.world_center
        _, normal = self.map.nominal_face(block.layer, block.slot)

        outcome, reason = "error", ""
        peak, profile, displacement = 0.0, [], None
        servo_time, iterations = None, 0
        alignment = (None, None, None)
        staging = self._tool_target(center, normal, cfg.staging_distance)
        pre_push = self._tool_target(center, normal, cfg.approach_distance)
        q_staging = self.arm.ik(staging)
        if q_staging is None or self.arm.ik(pre_push) is None:
            reason = "ik_failure"
        else:
            self._move_joints(q_staging, trajectory)
            self._move_linear(pre_push, cfg.approach_speed_m_s, trajectory)
            if self.controller == "closed_loop":
                reason, servo_time, iterations = self._servo(block, center, trajectory)
                reason = reason or ""
            else:
                goal = self._tool_target(center, normal, cfg.probe_clearance)
                self._move_linear(goal, cfg.approach_speed_m_s / 2, trajectory)
                self._hold(0.15)
            if not reason:
                # If the approach already knocked the tower over, the target
                # has moved and an alignment error would be meaningless.
                if not self.tower.has_collapsed(exclude=block.block_id):
                    alignment = self._alignment_error(block)
                outcome, reason, peak, profile, displacement = self._push(block, threshold, trajectory)
            back = self.arm.tool_pose()
            retreat = make_pose(back[:3, :3], back[:3, 3] - back[:3, 2] * (
                cfg.approach_distance + max(0.0, displacement or 0.0)))
            self._move_linear(retreat, cfg.approach_speed_m_s, trajectory)
        self._hold(0.5)
        # A target that lost the finger and tipped out of the tower has been
        # extracted: in the paper such blocks fall out behind the tower.
        if (block.status != "extracted" and reason in ("push_incomplete", "push_jammed")
                and self.tower.has_fallen(block)):
            self.tower.remove(block)
            outcome, reason = "extracted", ""
        # A half-pushed target is a failed attempt, not a collapse; later
        # checks measure it from where it was left.
        collapsed = self.tower.has_collapsed(exclude=block.block_id)
        if block.status != "extracted":
            self.tower.initial_poses[block.block_id] = self.tower.pose(block)
        after = self.tower.snapshot()
        perturbation = max(
            (float(np.linalg.norm(after[k] - before[k]))
             for k in after if k in before and k != block.block_id),
            default=0.0,
        ) * 1000.0
        self.policy.observe(block, outcome, peak)
        result = AttemptResult(
            trial=self.trial_index,
            seed=self.seed,
            detector=self.detector_mode,
            controller=self.controller,
            attempt=len(self.attempts),
            block_id=block.block_id,
            layer=block.layer,
            slot=block.slot,
            phase=phase,
            threshold_n=threshold,
            outcome=outcome,
            success=outcome in ("extracted", "immobile") and not collapsed,
            failure_reason="tower_collapse" if collapsed else reason,
            peak_force_n=peak,
            alignment_error_mm=alignment[0],
            alignment_error_h_mm=alignment[1],
            alignment_error_v_mm=alignment[2],
            servo_time_s=servo_time,
            servo_iterations=iterations,
            axial_displacement_mm=None if displacement is None else displacement * 1000.0,
            tower_perturbation_mm=perturbation,
            tower_collapsed=collapsed,
            sim_time_s=self.sim_time - start_time,
            trajectory=trajectory,
            force_profile=profile,
        )
        self.attempts.append(result)
        return result

    def run_game(self, trial_index: int = 0, max_attempts: Optional[int] = None) -> list[AttemptResult]:
        self.trial_index = trial_index
        limit = max_attempts or self.config.max_attempts_per_game
        start = len(self.attempts)
        for _ in range(limit):
            attempt = self.run_attempt()
            if attempt is None or attempt.tower_collapsed:
                break
        return self.attempts[start:]

    def close(self) -> None:
        if p.isConnected():
            p.disconnect()
        shutil.rmtree(self.asset_dir, ignore_errors=True)

    def __enter__(self) -> "JengaSimulation":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# --------------------------------------------------------------------------
# Metrics, outputs and CLI
# --------------------------------------------------------------------------

def aggregate(results: Sequence[AttemptResult]) -> dict[str, Any]:
    total = len(results)
    count = {o: sum(r.outcome == o and not r.tower_collapsed for r in results) for o in OUTCOMES}
    collapses = sum(r.tower_collapsed for r in results)
    successes = sum(r.success for r in results)
    errors = total - successes
    trials: dict[int, list[AttemptResult]] = {}
    for result in results:
        trials.setdefault(result.trial, []).append(result)
    per_trial = {"successes": [], "extractions": [], "immobile": [], "errors": [], "longest_streak": []}
    for attempts in trials.values():
        per_trial["successes"].append(sum(r.success for r in attempts))
        per_trial["extractions"].append(sum(r.success and r.outcome == "extracted" for r in attempts))
        per_trial["immobile"].append(sum(r.success and r.outcome == "immobile" for r in attempts))
        per_trial["errors"].append(sum(not r.success for r in attempts))
        longest = streak = 0
        for r in attempts:
            streak = streak + 1 if r.success else 0
            longest = max(longest, streak)
        per_trial["longest_streak"].append(longest)
    aligned = [r for r in results if r.alignment_error_mm is not None and math.isfinite(r.alignment_error_mm)]
    servo = [r for r in results if r.servo_time_s is not None
             and r.failure_reason not in ("servo_timeout", "tracking_lost")]
    servo_by_level: dict[int, list[float]] = {}
    for r in servo:
        servo_by_level.setdefault(r.layer, []).append(r.servo_time_s)
    reasons: dict[str, int] = {}
    for r in results:
        if not r.success:
            key = r.failure_reason or r.outcome
            reasons[key] = reasons.get(key, 0) + 1
    return {
        "trials": len(trials),
        "attempts": total,
        "extracted": count["extracted"],
        "immobile_classified": count["immobile"],
        "errors": errors,
        "tower_collapses": collapses,
        "successful_attempts": successes,
        "extraction_rate": count["extracted"] / total if total else 0.0,
        "immobile_rate": count["immobile"] / total if total else 0.0,
        "error_rate": errors / total if total else 0.0,
        "success_rate": successes / total if total else 0.0,
        "failure_reasons": reasons,
        "successful_attempts_per_trial": mean_std_ci(per_trial["successes"]),
        "max_successful_attempts": max(per_trial["successes"], default=0),
        "extractions_per_trial": mean_std_ci(per_trial["extractions"]),
        "max_extractions": max(per_trial["extractions"], default=0),
        "immobile_per_trial": mean_std_ci(per_trial["immobile"]),
        "errors_per_trial": mean_std_ci(per_trial["errors"]),
        "longest_success_streak_per_trial": mean_std_ci(per_trial["longest_streak"]),
        "alignment_error_mm": mean_std_ci([r.alignment_error_mm for r in aligned]),
        "alignment_error_h_mm": mean_std_ci([r.alignment_error_h_mm for r in aligned]),
        "alignment_error_v_mm": mean_std_ci([r.alignment_error_v_mm for r in aligned]),
        "within_7mm_rate": (
            sum(r.alignment_error_mm <= 7.0 for r in aligned) / len(aligned) if aligned else None
        ),
        "servo_time_s": mean_std_ci([r.servo_time_s for r in servo]),
        "servo_time_s_by_level": {
            str(level): mean_std_ci(times) for level, times in sorted(servo_by_level.items())
        },
        "peak_force_n_extracted": mean_std_ci([r.peak_force_n for r in results if r.outcome == "extracted"]),
        "peak_force_n_immobile": mean_std_ci([r.peak_force_n for r in results if r.outcome == "immobile"]),
        "tower_perturbation_mm": mean_std_ci([r.tower_perturbation_mm for r in results]),
    }


def _save_results(results: Sequence[AttemptResult], output_dir: str, config: SimulationConfig) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    columns = [f.name for f in fields(AttemptResult) if f.name not in ("trajectory", "force_profile")]
    with open(os.path.join(output_dir, "attempts.csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            writer.writerow({key: row[key] for key in columns})
    combinations: dict[str, list[AttemptResult]] = {}
    for result in results:
        combinations.setdefault(f"{result.detector}_{result.controller}", []).append(result)
    summary = {
        "assumptions": {
            "block_mass_kg": config.block_mass_kg,
            "block_friction_range": config.block_friction_range,
            "block_height_tolerance_m": config.block_height_tolerance_m,
            "joint_bias_std_rad": config.joint_bias_std_rad,
            "physical_parameters_are_calibrated": False,
            "note": (
                "Mass and friction come from the mechanics paper (Ziglar 2006); "
                "height tolerances and joint-encoder offsets are modelling choices. "
                "PyBullet friction is an approximation of Coulomb friction. Results "
                "are simulation outcomes, not a physical replication."
            ),
        },
        "configuration": asdict(config),
        "comparisons": {key: aggregate(group) for key, group in sorted(combinations.items())},
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    _plot_results(results, output_dir, combinations, config)
    return summary


def _plot_results(results: Sequence[AttemptResult], output_dir: str,
                  combinations: dict[str, list[AttemptResult]], config: SimulationConfig) -> None:
    if not results:
        return
    names = sorted(combinations)

    # Outcome breakdown (paper Fig. 10).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["extracted", "immobile", "error", "collapse"]
    colors = ["#2a9d8f", "#e9c46a", "#e76f51", "#6d2e46"]
    bottoms = np.zeros(len(names))
    for label, color in zip(labels, colors):
        values = []
        for name in names:
            group = combinations[name]
            if label == "collapse":
                n = sum(r.tower_collapsed for r in group)
            elif label == "error":
                n = sum(not r.success and not r.tower_collapsed for r in group)
            else:
                n = sum(r.success and r.outcome == label for r in group)
            values.append(100.0 * n / len(group))
        ax.bar(names, values, bottom=bottoms, color=color, label=label)
        bottoms += values
    ax.set_ylabel("Attempts (%)")
    ax.set_title("Attempt outcomes")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "outcomes.png"), dpi=140)
    plt.close(fig)

    # Force profiles, mobile vs immobile (paper Fig. 6).
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in results:
        if not r.force_profile:
            continue
        data = np.asarray(r.force_profile)
        color = "#d1495b" if r.outcome == "extracted" else "#00798c" if r.outcome == "immobile" else "#999999"
        ax.plot(data[:, 0], data[:, 1], color=color, alpha=0.35, linewidth=1)
    ax.axhline(config.first_force_threshold, color="black", linestyle="--", linewidth=1)
    ax.axhline(config.second_force_threshold, color="black", linestyle=":", linewidth=1)
    ax.plot([], [], color="#d1495b", label="extracted (mobile)")
    ax.plot([], [], color="#00798c", label="immobile")
    ax.plot([], [], color="black", linestyle="--", label=f"{config.first_force_threshold} N")
    ax.plot([], [], color="black", linestyle=":", label=f"{config.second_force_threshold} N")
    ax.set_xlabel("Push time (s)")
    ax.set_ylabel("Sensor force (N)")
    ax.set_title("Fingertip force during pushes (20 Hz)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "force_profiles.png"), dpi=140)
    plt.close(fig)

    # Fingertip trajectories (top and side views).
    fig, (top, side) = plt.subplots(1, 2, figsize=(10, 4.5))
    for r in results[:40]:
        if r.trajectory:
            path = np.asarray(r.trajectory)
            top.plot(path[:, 0], path[:, 1], alpha=0.5, linewidth=0.8)
            side.plot(np.hypot(path[:, 0], path[:, 1]), path[:, 2], alpha=0.5, linewidth=0.8)
    for axis, (xl, yl) in ((top, ("x (m)", "y (m)")), (side, ("distance from tower axis (m)", "z (m)"))):
        axis.set_xlabel(xl)
        axis.set_ylabel(yl)
    top.set_aspect("equal")
    top.set_title("Fingertip paths (top view)")
    side.set_title("Fingertip paths (side view)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectories.png"), dpi=140)
    plt.close(fig)

    # Alignment error scatter (paper Fig. 9) and servo convergence (Fig. 8).
    fig, (scatter, conv) = plt.subplots(1, 2, figsize=(11, 4.5))
    markers = {"closed_loop": "o", "open_loop": "x"}
    for name in names:
        group = [r for r in combinations[name] if r.alignment_error_h_mm is not None
                 and math.isfinite(r.alignment_error_h_mm)]
        if group:
            scatter.scatter([r.alignment_error_h_mm for r in group], [r.alignment_error_v_mm for r in group],
                            s=14, alpha=0.6, label=name, marker=markers[group[0].controller])
    scatter.add_patch(plt.Circle((0, 0), config.position_tolerance * 1000, fill=False, linestyle="--"))
    scatter.set_aspect("equal")
    scatter.set_xlabel("horizontal error (mm)")
    scatter.set_ylabel("vertical error (mm)")
    scatter.set_title("Contact point vs face centre (7 mm tolerance)")
    scatter.legend(fontsize=7, loc="upper right")
    for offset, name in enumerate(names):
        group = [r for r in combinations[name] if r.servo_time_s is not None]
        levels = sorted({r.layer for r in group})
        if not levels:
            continue
        means = [np.mean([r.servo_time_s for r in group if r.layer == L]) for L in levels]
        stds = [np.std([r.servo_time_s for r in group if r.layer == L]) for L in levels]
        conv.errorbar(np.array(levels) + 0.15 * offset, means, yerr=stds, fmt="o", capsize=3, label=name)
    conv.set_xlabel("Tower level (0 = bottom)")
    conv.set_ylabel("Servo convergence time (s)")
    conv.set_title("Visual servo convergence by level")
    if conv.has_data():
        conv.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "alignment_and_servo.png"), dpi=140)
    plt.close(fig)


def run_trial(job: dict[str, Any]) -> list[AttemptResult]:
    """Run one game (picklable entry point for worker processes)."""
    with JengaSimulation(
        config=SimulationConfig(**job["config"]),
        detector=job["detector"],
        controller=job["controller"],
        seed=job["seed"],
        gui=job.get("gui", False),
    ) as simulation:
        results = simulation.run_game(job["trial"], job["max_attempts"])
    if job.get("verbose"):
        extracted = sum(r.success and r.outcome == "extracted" for r in results)
        print(f"  {job['detector']}/{job['controller']} trial {job['trial']}: "
              f"{len(results)} attempts, {extracted} extracted", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=DETECTORS, default="ground_truth")
    parser.add_argument("--controller", choices=CONTROLLERS, default="closed_loop")
    parser.add_argument("--compare", action="store_true",
                        help="Run all detector x controller combinations on the same towers.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)),
                        help="Parallel worker processes (ignored with --gui).")
    parser.add_argument("--block-mass-kg", type=float, default=SimulationConfig.block_mass_kg)
    parser.add_argument("--friction-min", type=float, default=SimulationConfig.block_friction_range[0])
    parser.add_argument("--friction-max", type=float, default=SimulationConfig.block_friction_range[1])
    parser.add_argument("--height-tolerance-mm", type=float,
                        default=SimulationConfig.block_height_tolerance_m * 1000,
                        help="Std of per-block height error (makes some blocks load-bearing).")
    parser.add_argument("--joint-bias-deg", type=float,
                        default=math.degrees(SimulationConfig.joint_bias_std_rad),
                        help="Std of joint encoder offsets (robot calibration error).")
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
            block_height_tolerance_m=args.height_tolerance_mm / 1000,
            joint_bias_std_rad=math.radians(args.joint_bias_deg),
        )
    except ValueError as error:
        parser.error(str(error))
    combinations = (
        [(d, c) for d in DETECTORS for c in CONTROLLERS]
        if args.compare else [(args.detector, args.controller)]
    )
    jobs = [
        {"config": asdict(config), "detector": d, "controller": c, "trial": t,
         "seed": args.seed + t, "max_attempts": args.max_attempts, "gui": args.gui, "verbose": True}
        for d, c in combinations for t in range(args.trials)
    ]
    started = time.time()
    print(f"Running {len(jobs)} game(s)...", flush=True)
    batches: dict[int, list[AttemptResult]] = {}
    failed: list[str] = []

    def collect(index: int, outcome: Any) -> None:
        job = jobs[index]
        name = f"{job['detector']}/{job['controller']} trial {job['trial']}"
        if isinstance(outcome, BaseException):
            failed.append(f"{name}: {outcome!r}")
            print(f"  {name} FAILED: {outcome!r}", flush=True)
            return
        batches[index] = outcome
        # Save after every game so a long run can be inspected or survive a crash.
        partial = [r for i in sorted(batches) for r in batches[i]]
        _save_results(partial, args.output, config)
        print(f"  saved {len(batches)}/{len(jobs)} games "
              f"({time.time() - started:.0f} s elapsed)", flush=True)

    if args.gui or args.workers <= 1 or len(jobs) == 1:
        for index, job in enumerate(jobs):
            try:
                collect(index, run_trial(job))
            except Exception as error:  # keep going with the other games
                collect(index, error)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_trial, job): i for i, job in enumerate(jobs)}
            for future in as_completed(futures):
                error = future.exception()
                collect(futures[future], error if error is not None else future.result())
    results = [r for i in sorted(batches) for r in batches[i]]
    summary = _save_results(results, args.output, config)
    if failed:
        print("\nGames that failed:\n  " + "\n  ".join(failed))
    for name, stats in summary["comparisons"].items():
        print(f"\n{name}: {stats['attempts']} attempts, "
              f"{100 * stats['extraction_rate']:.0f}% extracted, "
              f"{100 * stats['immobile_rate']:.0f}% immobile, "
              f"{100 * stats['error_rate']:.0f}% errors; "
              f"successes/trial {stats['successful_attempts_per_trial']['mean']:.1f}, "
              f"max extractions {stats['max_extractions']}; "
              f"alignment {stats['alignment_error_mm']['mean'] or float('nan'):.2f} mm")
    print(f"\nFinished in {time.time() - started:.0f} s. Results written to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
