"""Render simulator images and record example force traces for the write-up.

Runs a few closed-loop attempts with the OpenCV detector and saves:
  figures/scene_push_*.png   external views of a block being pushed out
  figures/camera_*.png     eye-in-hand images with OpenCV face detections
  figures/force_traces.json  sensor force profiles of the attempts
Run from the repository root:  python docs/capture_scene.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pybullet as p

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jenga_replication as jr  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
ATTEMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
VIEWS = {
    "front": ((0.95, 0.45, 0.62), (0.18, -0.22, 0.27)),
    "side": ((-0.55, -0.85, 0.55), (0.15, -0.15, 0.27)),
}


def render_external(name, eye, target, width=1600, height=1200, fov=40):
    view = p.computeViewMatrix(eye, target, [0, 0, 1])
    projection = p.computeProjectionMatrixFOV(fov, width / height, 0.02, 5.0)
    image = p.getCameraImage(width, height, view, projection, renderer=p.ER_TINY_RENDERER,
                             lightDirection=[0.6, -0.4, 1.0], lightAmbientCoeff=0.55,
                             lightDiffuseCoeff=0.5, lightSpecularCoeff=0.05, shadow=0,
                             flags=p.ER_NO_SEGMENTATION_MASK)
    rgb = np.reshape(image[2], (height, width, 4))[:, :, :3].astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, name), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def camera_with_detections(sim, name, target=None):
    """Eye-in-hand RGB image with the OpenCV face estimates drawn on it."""
    true_camera = sim.arm.true_link_pose(sim.arm.camera_link)
    state = sim.vision.rng.bit_generator.state
    rgb, _, _ = sim.vision.render(true_camera, segmentation=False)
    observations = sim._detect()
    sim.vision.rng.bit_generator.state = state
    image = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for obs in observations:
        if not np.all(np.isfinite(obs.pixel)):
            continue
        u, v = (int(round(2 * c)) for c in obs.pixel)
        is_target = target is not None and obs.block_id == target
        colour = (40, 40, 230) if is_target else (214, 120, 42)  # BGR
        cv2.circle(image, (u, v), 9 if is_target else 6, colour, 2, cv2.LINE_AA)
        axis = obs.rotation_cam
        for column, length in ((0, 0.0125), (1, 0.0075)):
            end = sim.vision.project(obs.center_cam + axis[:, column] * length)
            if end is not None:
                cv2.line(image, (u, v), (int(2 * end[0]), int(2 * end[1])), colour, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, name), image)
    return len(observations)


def main():
    os.makedirs(OUT, exist_ok=True)
    traces = []
    shots = {"push": False, "camera_close": False}
    with jr.JengaSimulation(config=jr.SimulationConfig(), detector="opencv",
                            controller="closed_loop", seed=SEED) as sim:
        original_push = sim._push
        original_overview = sim.overview

        def overview(trajectory=None):
            result = original_overview(trajectory)
            if not os.path.exists(os.path.join(OUT, "camera_overview.png")):
                print("overview detections:", camera_with_detections(sim, "camera_overview.png"))
            return result

        def push(block, threshold, trajectory):
            if not shots["camera_close"]:
                camera_with_detections(sim, "camera_aligned.png", block.block_id)
                shots["camera_close"] = True
            original_step = sim.step
            start, _ = sim.tower.pose(block)

            def step(count=1):
                original_step(count)
                if not shots["push"] and block.status != "extracted":
                    moved = np.linalg.norm(sim.tower.pose(block)[0] - start)
                    if moved > 0.035:
                        for view_name, (eye, target) in VIEWS.items():
                            render_external(f"scene_push_{view_name}.png", eye, target)
                        shots["push"] = True

            sim.step = step
            try:
                return original_push(block, threshold, trajectory)
            finally:
                sim.step = original_step

        sim._push = push
        sim.overview = overview
        for index in range(ATTEMPTS):
            attempt = sim.run_attempt()
            if attempt is None:
                break
            print(index, attempt.layer, attempt.slot, attempt.outcome, attempt.failure_reason,
                  round(attempt.peak_force_n, 3), flush=True)
            traces.append({
                "layer": attempt.layer, "slot": attempt.slot, "outcome": attempt.outcome,
                "threshold": attempt.threshold_n,
                "displacement_mm": attempt.axial_displacement_mm,
                "profile": attempt.force_profile,
            })
            if attempt.tower_collapsed:
                break
    with open(os.path.join(OUT, "force_traces.json"), "w", encoding="utf-8") as stream:
        json.dump(traces, stream)


if __name__ == "__main__":
    main()
