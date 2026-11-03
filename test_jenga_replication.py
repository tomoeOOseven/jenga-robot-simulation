import math
import unittest

import numpy as np

from jenga_replication import (
    AttemptResult,
    Block,
    BlockSelectionPolicy,
    SimulationConfig,
    aggregate,
    face_frame,
    interaction_matrix,
    invert_pose,
    make_pose,
    mean_std_ci,
    quantize_force,
    quat_from_matrix,
    matrix_from_quat,
    rotation_vector,
    trapezoid_profile,
    visual_servo_twist,
)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _result(trial, outcome, collapsed=False, alignment=1.0):
    success = outcome in ("extracted", "immobile") and not collapsed
    return AttemptResult(
        trial=trial, seed=0, detector="ground_truth", controller="closed_loop",
        attempt=0, block_id=0, layer=4, slot=1, phase=0, threshold_n=0.32,
        outcome=outcome, success=success, failure_reason="" if success else "x",
        peak_force_n=0.1, alignment_error_mm=alignment, alignment_error_h_mm=alignment,
        alignment_error_v_mm=0.0, servo_time_s=2.0, servo_iterations=10,
        axial_displacement_mm=0.0, tower_perturbation_mm=0.0,
        tower_collapsed=collapsed, sim_time_s=10.0,
    )


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = SimulationConfig()
        self.blocks = [
            Block(layer * 3 + slot, layer, slot, "x" if layer % 2 == 0 else "y", -1)
            for layer in range(18) for slot in range(3)
        ]

    def policy(self, seed=0):
        return BlockSelectionPolicy(self.blocks, self.config, seed)

    def test_only_paper_levels_are_candidates(self):
        levels = {b.layer for s in (0, 1) for b in self.policy().candidates(s)}
        self.assertEqual(levels, set(range(self.config.min_level, self.config.max_level + 1)))
        self.assertEqual(len(levels), 14)  # paper: 14 levels -> 42 attempts minimum

    def test_centre_block_is_tried_first(self):
        self.assertEqual(self.policy().next_block().slot, 1)

    def test_one_extraction_per_level(self):
        policy = self.policy(2)
        first = policy.next_block()
        policy.observe(first, "extracted")
        for _ in range(30):
            block = policy.next_block()
            if block is None:
                break
            self.assertNotEqual(block.layer, first.layer)
            policy.observe(block, "immobile")

    def test_immobile_block_keeps_the_same_level(self):
        policy = self.policy()
        first = policy.next_block()
        policy.observe(first, "immobile")
        second = policy.next_block()
        self.assertEqual(second.layer, first.layer)
        self.assertNotEqual(second.block_id, first.block_id)
        self.assertEqual(first.status, "tested")

    def test_threshold_follows_game_phase_not_level(self):
        for seed in range(6):
            policy = self.policy(seed)
            start_subspace = policy.current_subspace
            self.assertEqual(policy.force_threshold, 0.32)
            while policy.phase == 0:
                block = policy.next_block()
                if policy.phase == 0:
                    policy.observe(block, "immobile")
            self.assertNotEqual(policy.current_subspace, start_subspace)
            self.assertEqual(policy.force_threshold, 0.18)

    def test_both_start_subspaces_occur(self):
        starts = {self.policy(seed).current_subspace for seed in range(20)}
        self.assertEqual(starts, {0, 1})

    def test_errors_retry_then_skip(self):
        policy = self.policy()
        block = policy.next_block()
        policy.observe(block, "error")
        self.assertEqual(block.status, "present")
        self.assertIs(policy.next_block(), block)
        policy.observe(block, "error")
        self.assertEqual(block.status, "skipped")
        self.assertIsNot(policy.next_block(), block)

    def test_only_visible_blocks_are_selected(self):
        visible = {b.block_id for b in self.blocks if b.layer in (3, 12) and b.slot == 0}
        block = self.policy().next_block(visible)
        self.assertIn(block.block_id, visible)

    def test_game_ends_when_no_candidates(self):
        policy = self.policy()
        self.assertIsNone(policy.next_block(visible=set()))


class MathTests(unittest.TestCase):
    def test_trapezoid_profile_is_monotonic_and_normalized(self):
        profile = trapezoid_profile(200)
        self.assertEqual(profile[0], 0)
        self.assertEqual(profile[-1], 1)
        self.assertTrue(np.all(np.diff(profile) >= 0))
        velocity = np.diff(profile)
        self.assertAlmostEqual(velocity[100], velocity.max(), places=6)  # cruise phase

    def test_rotation_helpers_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            axis, angle = rng.normal(size=3), rng.uniform(0, math.pi * 0.99)
            rotation = _rotation(axis, angle)
            vector = rotation_vector(rotation)
            self.assertAlmostEqual(np.linalg.norm(vector), angle, places=6)
            np.testing.assert_allclose(matrix_from_quat(quat_from_matrix(rotation)), rotation, atol=1e-9)

    def test_face_frame_is_right_handed_with_vertical_y(self):
        frame = face_frame(np.array([0.0, -1.0, 0.0]))
        np.testing.assert_allclose(frame[:, 1], [0, 0, 1])
        self.assertAlmostEqual(np.linalg.det(frame), 1.0)

    def test_interaction_matrix_matches_finite_differences(self):
        """Feature rates (x, y, log Z) for a camera twist match L @ v."""
        point = np.array([0.03, -0.02, 0.2])
        twist = np.array([0.01, -0.02, 0.03, 0.2, -0.1, 0.3])
        dt = 1e-6

        def features(p):
            return np.array([p[0] / p[2], p[1] / p[2], math.log(p[2])])

        # A static point seen from a moving camera: p_dot = -v - w x p.
        moved = point + dt * (-twist[:3] - np.cross(twist[3:], point))
        numeric = (features(moved) - features(point)) / dt
        analytic = interaction_matrix(point[0] / point[2], point[1] / point[2], point[2])[:3] @ twist
        np.testing.assert_allclose(numeric, analytic, rtol=1e-4, atol=1e-6)

    def test_visual_servo_law_converges(self):
        """Integrating v = -lambda L+ e drives the camera onto the goal pose."""
        desired = make_pose(_rotation([1, 0, 0], 0.4), [0.0, 0.03, 0.19])
        camera = make_pose(_rotation([0.3, 1, 0.2], 0.25), [0.02, -0.03, -0.08])
        face_world = desired.copy()  # camera starts away from the identity goal
        dt = 0.02
        for _ in range(600):
            current = invert_pose(camera) @ face_world
            twist, error = visual_servo_twist(current, desired, 1.5)
            camera[:3, 3] += camera[:3, :3] @ twist[:3] * dt
            camera[:3, :3] = camera[:3, :3] @ _rotation(twist[3:] + 1e-12, np.linalg.norm(twist[3:]) * dt)
        self.assertLess(np.linalg.norm(error), 1e-4)
        np.testing.assert_allclose(camera, np.eye(4), atol=1e-4)

    def test_force_quantisation_matches_12_bit_5_newton_sensor(self):
        config = SimulationConfig()
        resolution = 5.0 / 4096
        self.assertAlmostEqual(quantize_force(0.32, config) % resolution, 0.0, places=9)
        self.assertEqual(quantize_force(-1.0, config), 0.0)
        self.assertEqual(quantize_force(10.0, config), 5.0)


class MetricTests(unittest.TestCase):
    def test_aggregate_counts_attempt_outcomes(self):
        results = [
            _result(0, "extracted"), _result(0, "immobile"), _result(0, "error", alignment=None),
            _result(1, "extracted"), _result(1, "extracted", collapsed=True),
        ]
        summary = aggregate(results)
        self.assertEqual(summary["attempts"], 5)
        self.assertEqual(summary["extracted"], 2)
        self.assertEqual(summary["immobile_classified"], 1)
        self.assertEqual(summary["errors"], 2)
        self.assertEqual(summary["tower_collapses"], 1)
        self.assertEqual(summary["max_extractions"], 1)
        self.assertAlmostEqual(summary["success_rate"], 0.6)
        self.assertEqual(summary["successful_attempts_per_trial"]["mean"], 1.5)

    def test_mean_std_ci_uses_t_distribution(self):
        stats = mean_std_ci([1.0, 2.0, 3.0])
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["std"], 1.0)
        self.assertAlmostEqual(stats["ci95"], 4.303 / math.sqrt(3), places=3)

    def test_config_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            SimulationConfig(block_mass_kg=0)
        with self.assertRaises(ValueError):
            SimulationConfig(block_friction_range=(0.5, 0.4))
        with self.assertRaises(ValueError):
            SimulationConfig(max_level=17)


class SimulationTests(unittest.TestCase):
    """Slower tests that run PyBullet (about a minute in total)."""

    def test_kinematics_jacobian_and_ik(self):
        from jenga_replication import JengaSimulation

        with JengaSimulation(seed=1) as sim:
            arm = sim.arm
            rng = np.random.default_rng(0)
            for _ in range(3):
                q = rng.uniform(arm.lower + 0.3, arm.upper - 0.3)
                pose, jacobian = arm.pose_and_jacobian(q)
                numeric = []
                for joint in range(7):
                    dq = np.zeros(7)
                    dq[joint] = 1e-4
                    moved = arm.tool_pose(q + dq)
                    numeric.append(np.concatenate((
                        (moved[:3, 3] - pose[:3, 3]) / 1e-4,
                        rotation_vector(moved[:3, :3] @ pose[:3, :3].T) / 1e-4,
                    )))
                np.testing.assert_allclose(np.array(numeric).T, jacobian, atol=5e-3)
                solution = arm.ik(pose)
                self.assertIsNotNone(solution)
                np.testing.assert_allclose(arm.tool_pose(solution)[:3, 3], pose[:3, 3], atol=3e-4)

    def test_force_sensor_reads_applied_force(self):
        import pybullet as p
        from jenga_replication import JengaSimulation

        with JengaSimulation(seed=1) as sim:
            arm = sim.arm
            # Horizontal finger, as during a push (away from the tower).
            center, normal = sim.map.nominal_face(10, 1)
            sim._move_joints(arm.ik(sim._tool_target(center, normal, 0.2)), [])
            sim._hold(0.3)
            arm.tare()
            for _ in range(240):
                tool = arm.true_link_pose(arm.tool_link)
                p.applyExternalForce(arm.robot_id, arm.tip_link, (-0.25 * tool[:3, 2]).tolist(),
                                     tool[:3, 3].tolist(), p.WORLD_FRAME)
                sim.step()
            self.assertAlmostEqual(arm.sensor_force(), 0.25, delta=0.02)

    def test_both_detectors_see_and_associate_the_tower(self):
        from jenga_replication import JengaSimulation

        config = SimulationConfig(depth_scale_error=0.0)
        estimates = {}
        for detector in ("ground_truth", "opencv"):
            with JengaSimulation(config=config, detector=detector, seed=5) as sim:
                candidates = sim.overview()
                self.assertGreaterEqual(len(candidates), 30, detector)
                truth = {}
                for block_id, obs in candidates.items():
                    center, _ = sim.tower.face(sim.tower.blocks[block_id])
                    truth[block_id] = center
                estimates[detector] = {k: v.world_center for k, v in candidates.items()}
        shared = set(estimates["ground_truth"]) & set(estimates["opencv"])
        self.assertGreaterEqual(len(shared), 25)
        # Same robot calibration error, so both estimates agree closely.
        differences = [np.linalg.norm(estimates["ground_truth"][k] - estimates["opencv"][k]) for k in shared]
        self.assertLess(float(np.median(differences)), 0.002)

    def test_closed_loop_attempt_aligns_and_extracts_a_free_block(self):
        from jenga_replication import JengaSimulation

        for detector in ("ground_truth", "opencv"):
            with self.subTest(detector=detector):
                with JengaSimulation(detector=detector, controller="closed_loop", seed=5) as sim:
                    # Force the policy onto a light upper level (free block).
                    sim.policy.current_subspace = 1
                    sim.policy.current_level = 14
                    result = sim.run_attempt()
                    self.assertEqual(result.layer, 14)
                    self.assertLess(result.alignment_error_mm, 1.0)
                    self.assertEqual(result.outcome, "extracted")
                    self.assertTrue(result.success)
                    self.assertLess(result.peak_force_n, 0.32)
                    self.assertEqual(sim.tower.blocks[result.block_id].status, "extracted")
                    self.assertGreater(result.axial_displacement_mm, 75.0)
                    self.assertTrue(result.force_profile)


if __name__ == "__main__":
    unittest.main()
