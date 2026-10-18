import unittest

import numpy as np

from jenga_replication import (
    AttemptResult,
    Block,
    BlockSelectionPolicy,
    SimulationConfig,
    aggregate,
    force_feedback_correction,
    trapezoid_profile,
)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.blocks = [
            Block(i, layer, slot, "x", i)
            for i, (layer, slot) in enumerate(
                (layer, slot) for layer in range(18) for slot in range(3)
            )
        ]

    def test_policy_selects_one_block_per_layer_after_extraction(self):
        policy = BlockSelectionPolicy(self.blocks, layers=18, seed=2)
        selected = policy.next_block()
        self.assertIsNotNone(selected)
        policy.observe(selected, "extracted")
        following = policy.next_block()
        self.assertIsNotNone(following)
        self.assertNotEqual(following.layer, selected.layer)

    def test_policy_tries_other_blocks_after_immobile_result(self):
        policy = BlockSelectionPolicy(self.blocks, layers=18, seed=0)
        first = policy.next_block()
        policy.observe(first, "immobile")
        second = policy.next_block()
        self.assertEqual(first.layer, second.layer)
        self.assertNotEqual(first.block_id, second.block_id)

    def test_policy_defers_a_target_after_a_perception_error(self):
        policy = BlockSelectionPolicy(self.blocks, layers=18, seed=0)
        first = policy.next_block()
        policy.observe(first, "error")
        second = policy.next_block()
        self.assertNotEqual(first.block_id, second.block_id)

    def test_phase_thresholds_match_presentation(self):
        policy = BlockSelectionPolicy(self.blocks, layers=18, seed=0)
        self.assertIn(policy.force_threshold, (0.32, 0.18))
        policy.phase = 0
        self.assertEqual(policy.force_threshold, 0.32)
        policy.phase = 1
        self.assertEqual(policy.force_threshold, 0.18)

    def test_policy_excludes_unstable_second_to_top_layer(self):
        policy = BlockSelectionPolicy(self.blocks, layers=18, seed=0)
        eligible = policy._eligible(1)
        self.assertNotIn(16, {block.layer for block in eligible})


class MetricAndTrajectoryTests(unittest.TestCase):
    def test_trapezoid_profile_is_monotonic_and_normalized(self):
        profile = trapezoid_profile(50)
        self.assertEqual(profile[0], 0)
        self.assertEqual(profile[-1], 1)
        self.assertTrue(np.all(np.diff(profile) >= 0))
        self.assertTrue(np.all(profile <= 1))

    def test_aggregate_counts_attempt_outcomes(self):
        results = [
            AttemptResult(0, 0, 4, "ground_truth", "closed_loop", 0.32,
                          "extracted", 0.1, 2.0, 1.0, True, False),
            AttemptResult(0, 1, 5, "ground_truth", "closed_loop", 0.32,
                          "immobile", 0.4, 3.0, 1.2, False, False),
            AttemptResult(0, 2, 6, "ground_truth", "closed_loop", 0.18,
                          "error", 0.0, None, 0.0, False, False),
        ]
        summary = aggregate(results)
        self.assertEqual(summary["attempts"], 3)
        self.assertEqual(summary["extracted"], 1)
        self.assertEqual(summary["immobile_classified"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertAlmostEqual(summary["alignment_error_mm_mean"], 2.5)
        self.assertEqual(summary["maximum_extractions"], 1)

    def test_default_physics_uses_mechanics_paper_mass_and_friction(self):
        config = SimulationConfig()
        self.assertAlmostEqual(config.block_mass_kg, 0.0196)
        self.assertEqual(config.block_friction_range, (0.4, 0.4))

    def test_force_feedback_yields_above_setpoint_and_pushes_below_it(self):
        config = SimulationConfig()
        below_target = force_feedback_correction(
            config.force_setpoint_n - 0.1,
            config.force_setpoint_n,
            config.force_control_gain_m_s_n,
            config.force_control_max_correction_m_s,
        )
        above_target = force_feedback_correction(
            config.force_setpoint_n + 0.1,
            config.force_setpoint_n,
            config.force_control_gain_m_s_n,
            config.force_control_max_correction_m_s,
        )
        self.assertGreater(below_target, 0)
        self.assertLess(above_target, 0)
        self.assertAlmostEqual(below_target, -above_target)


class SimulationSmokeTests(unittest.TestCase):
    def test_policy_selects_only_candidates_seen_by_each_camera_path(self):
        from jenga_replication import JengaSimulation

        for detector in ("ground_truth", "opencv"):
            with self.subTest(detector=detector):
                with JengaSimulation(
                    detector=detector,
                    controller="closed_loop",
                    seed=5,
                ) as simulation:
                    simulation._move_to_overview()
                    visible = simulation.vision.visible_block_ids(
                        simulation.tower.blocks,
                        detector,
                    )
                    selected = simulation.policy.next_block(visible)

                self.assertTrue(visible)
                self.assertIsNotNone(selected)
                self.assertIn(selected.block_id, visible)

    def test_tower_and_both_camera_detection_paths(self):
        from jenga_replication import JengaSimulation

        for detector in ("ground_truth", "opencv"):
            with JengaSimulation(detector=detector, seed=5) as simulation:
                self.assertEqual(len(simulation.tower.blocks), 54)
                self.assertFalse(simulation.tower.has_collapsed())
                block = next(
                    item for item in simulation.tower.blocks
                    if item.layer == 8 and item.slot == 1
                )
                face, push = simulation.tower.face_center_and_push(block)
                orientation = simulation._tool_orientation(push)
                pre_push = face - push * (
                    simulation.config.approach_distance
                    + simulation.config.probe_clearance
                )
                simulation.arm.move_cartesian(
                    pre_push, orientation, duration=0.5
                )
                detection = simulation.vision.target_detection(block, detector)
                self.assertIsNotNone(detection, detector)
                self.assertIsNotNone(detection.pnp_pose, detector)
                self.assertTrue(np.all(np.isfinite(detection.world_point)))

    def test_closed_loop_push_finishes_extraction_after_target_is_occluded(self):
        from jenga_replication import JengaSimulation

        for detector in ("ground_truth", "opencv"):
            with self.subTest(detector=detector):
                with JengaSimulation(
                    detector=detector,
                    controller="closed_loop",
                    seed=5,
                ) as simulation:
                    result = simulation.run_attempt()
                    self.assertEqual(result.outcome, "extracted")
                    self.assertFalse(result.tower_collapsed)
                    self.assertGreater(result.peak_force_n, 0.05)
                    self.assertLessEqual(
                        result.peak_force_n,
                        SimulationConfig().force_setpoint_n + 0.02,
                    )
                    self.assertTrue(any(
                        force > 0 for _, force in result.force_profile
                    ))
                    self.assertEqual(
                        next(
                            block.status
                            for block in simulation.tower.blocks
                            if block.block_id == result.block_id
                        ),
                        "extracted",
                    )
                    axial_clearance_mm = SimulationConfig().block_length * 1000
                    self.assertTrue(
                        result.axial_displacement_mm >= axial_clearance_mm
                        or (
                            result.axial_displacement_mm >= 0.9 * axial_clearance_mm
                            and result.vertical_drop_mm
                            >= SimulationConfig().block_height * 1000
                        )
                    )
                    self.assertGreater(result.vision_tracking_misses, 0)


if __name__ == "__main__":
    unittest.main()
