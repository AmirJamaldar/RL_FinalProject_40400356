from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from environments.generator import (
    PENALTY,
    generate_source_map,
    generate_transfer_map,
    load_map,
    maze_fingerprint,
    mission_path,
    validate_map,
)
from environments.maze import REQUIRED_EVENTS, DynamicMazeEnv, RewardSpec
from experiments.run_experiments import required_event_witnesses


class MapGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = generate_source_map("40400356")

    def test_student_specific_seed_and_size(self) -> None:
        self.assertEqual(self.source.metadata["base_seed"], 5)
        self.assertEqual(self.source.size, 16)

    def test_generation_is_byte_reproducible(self) -> None:
        again = generate_source_map("40400356")
        self.assertEqual(self.source.grid, again.grid)
        self.assertEqual(self.source.metadata, again.metadata)
        self.assertEqual(maze_fingerprint(self.source), maze_fingerprint(again))

    def test_hard_map_requirements_and_bfs(self) -> None:
        evidence = validate_map(self.source)
        self.assertGreaterEqual(evidence["wall_ratio"], 0.15)
        self.assertGreaterEqual(evidence["penalty_count"], 5)
        path = mission_path(self.source)
        self.assertIsNotNone(path)
        positions = [(row, col) for row, col, _ in path]
        self.assertLess(positions.index(self.source.key), positions.index(self.source.door))
        self.assertLess(positions.index(self.source.door), positions.index(self.source.goal))

    def test_map_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.source.save(Path(directory) / "map.json")
            loaded = load_map(path)
        self.assertEqual(self.source.grid, loaded.grid)
        self.assertEqual(self.source.metadata, loaded.metadata)

    def test_transfer_map_constraints(self) -> None:
        similar = generate_transfer_map(self.source, "similar", seed=106)
        different = generate_transfer_map(self.source, "different", seed=207)
        self.assertGreaterEqual(similar.metadata["actual_all_obstacle_relocation_fraction"], 0.15)
        self.assertLessEqual(similar.metadata["actual_all_obstacle_relocation_fraction"], 0.20)
        self.assertGreaterEqual(different.metadata["actual_all_obstacle_relocation_fraction"], 0.35)
        self.assertEqual(similar.start, self.source.start)
        self.assertEqual(similar.key, self.source.key)
        self.assertEqual(similar.goal, self.source.goal)
        self.assertNotEqual(different.key, self.source.key)
        self.assertGreaterEqual(len(different.positions(PENALTY)) - len(self.source.positions(PENALTY)), 3)
        self.assertIsNotNone(mission_path(similar))
        self.assertIsNotNone(mission_path(different))


class EnvironmentTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.maze = generate_source_map("40400356")
        self.env = DynamicMazeEnv(self.maze, reward_mode="sparse", gamma=0.95, seed=5)

    def test_shapes_and_episode_limit(self) -> None:
        self.assertEqual(self.env.q_shape, (2, 16, 16, 4))
        self.assertEqual(self.env.value_shape, (2, 16, 16))
        self.assertEqual(self.env.action_space_n, 4)
        self.assertEqual(self.env.max_steps, 3 * self.maze.walkable_count)

    def test_exact_action_noise_probabilities(self) -> None:
        outcomes = self.env.transition_distribution(self.env.start_state, 3)
        probability_by_action = {outcome.executed_action: outcome.probability for outcome in outcomes}
        self.assertAlmostEqual(sum(probability_by_action.values()), 1.0)
        self.assertAlmostEqual(probability_by_action[3], 0.8)
        self.assertAlmostEqual(probability_by_action[0], 0.1)
        self.assertAlmostEqual(probability_by_action[1], 0.1)

    def test_wall_collision_stays_and_penalizes(self) -> None:
        outcome = self.env.transition_for_executed_action(self.env.start_state, 0)
        self.assertEqual(outcome.next_state, self.env.start_state)
        self.assertEqual(outcome.events, ("wall_collision",))
        self.assertAlmostEqual(outcome.reward, -5.0)

    def test_closed_door_key_door_and_goal_semantics(self) -> None:
        door_row, door_col = self.maze.door
        closed_state = (door_row, door_col - 1, 0)
        closed = self.env.transition_for_executed_action(closed_state, 3)
        self.assertEqual(closed.next_state, closed_state)
        self.assertEqual(closed.events, ("closed_door_attempt",))
        self.assertAlmostEqual(closed.reward, -8.0)

        key_row, key_col = self.maze.key
        key_pickup = self.env.transition_for_executed_action((key_row, key_col - 1, 0), 3)
        self.assertEqual(key_pickup.next_state, (key_row, key_col, 1))
        self.assertEqual(key_pickup.events, ("key_collected",))
        self.assertAlmostEqual(key_pickup.reward, 29.0)

        door_pass = self.env.transition_for_executed_action((door_row, door_col - 1, 1), 3)
        self.assertEqual(door_pass.next_state, (door_row, door_col, 1))
        self.assertEqual(door_pass.events, ("door_passed",))
        self.assertAlmostEqual(door_pass.reward, -1.0)

        goal_row, goal_col = self.maze.goal
        goal = self.env.transition_for_executed_action((goal_row - 1, goal_col, 1), 1)
        self.assertTrue(goal.terminated)
        self.assertEqual(goal.events, ("goal_reached",))
        self.assertAlmostEqual(goal.reward, 99.0)

    def test_teleporter_changes_transition(self) -> None:
        portal_a, portal_b = sorted(self.maze.portals)
        # A is (2,3), entered from its left.
        outcome = self.env.transition_for_executed_action((portal_a[0], portal_a[1] - 1, 0), 3)
        self.assertEqual(outcome.next_state[:2], self.maze.portals[portal_a])
        self.assertIn("teleport", outcome.events)

    def test_termination_and_truncation_are_distinct(self) -> None:
        env = DynamicMazeEnv(self.maze, reward_mode="sparse", max_steps=1, seed=5)
        env.reset(seed=5)
        _, _, terminated, truncated, info = env.step(3)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIn("max_steps_truncation", info["events"])
        with self.assertRaises(RuntimeError):
            env.step(3)

    def test_terminal_model_state_is_absorbing(self) -> None:
        outcome = self.env.transition_for_executed_action(self.env.terminal_state, 0)
        self.assertEqual(outcome.next_state, self.env.terminal_state)
        self.assertTrue(outcome.terminated)
        self.assertEqual(outcome.reward, 0.0)
        self.assertEqual(outcome.events, ("terminal_absorbing",))

    def test_invalid_state_and_zero_step_limit_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DynamicMazeEnv(self.maze, max_steps=0)
        with self.assertRaises(ValueError):
            self.env.transition_for_executed_action((0, 0, 0), 1)
        with self.assertRaises(ValueError):
            self.env.transition_for_executed_action((1, 1, 2), 1)

    def test_nonfinite_reward_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RewardSpec(goal_reached=float("nan"))
        with self.assertRaises(ValueError):
            RewardSpec(shaping_scale=-0.1)

    def test_seeded_step_sequence_is_reproducible(self) -> None:
        actions = [3, 3, 1, 1, 2, 0, 3, 1]
        trajectories = []
        for _ in range(2):
            env = DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95)
            state, _ = env.reset(seed=12345)
            trajectory = [state]
            for action in actions:
                state, reward, terminated, truncated, info = env.step(action)
                trajectory.append((state, reward, info["executed_action"]))
                if terminated or truncated:
                    break
            trajectories.append(trajectory)
        self.assertEqual(trajectories[0], trajectories[1])

    def test_same_state_action_has_same_model_independent_of_history(self) -> None:
        state = self.env.start_state
        before = self.env.transition_distribution(state, 3)
        self.env.reset(seed=99)
        self.env.step(3)
        after = self.env.transition_distribution(state, 3)
        self.assertEqual(before, after)

    def test_all_required_events_have_real_witnesses(self) -> None:
        shaped = DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95)
        with tempfile.TemporaryDirectory() as directory:
            rows = required_event_witnesses(shaped, Path(directory) / "events.jsonl")
        self.assertEqual({row["event"] for row in rows}, REQUIRED_EVENTS)

    def test_invalid_action_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.env.transition_distribution(self.env.start_state, 4)


if __name__ == "__main__":
    unittest.main()
