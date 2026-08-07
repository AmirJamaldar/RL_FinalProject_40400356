from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np

from agents.common import episode_seed, evaluate_q_policy, load_q_checkpoint, policy_from_q, q_at, save_q_checkpoint
from agents.q_learning import EpsilonSchedule, QLearningAgent, QLearningConfig, train_q_learning
from agents.sarsa_lambda import SarsaLambdaConfig, train_sarsa_lambda
from agents.value_iteration import bellman_residual, value_iteration
from environments.generator import generate_source_map
from environments.maze import DynamicMazeEnv


class ValueIterationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.maze = generate_source_map("40400356")

    def test_convergence_and_bellman_residual(self) -> None:
        env = DynamicMazeEnv(self.maze, reward_mode="sparse", gamma=0.95)
        result = value_iteration(env, threshold=1e-8)
        self.assertTrue(result.converged)
        self.assertLess(result.delta_history[-1], 1e-8)
        self.assertLess(bellman_residual(env, result), 1e-6)
        self.assertTrue(np.all(np.isfinite(result.values)))

    def test_potential_shaping_preserves_optimal_policy(self) -> None:
        sparse_env = DynamicMazeEnv(self.maze, reward_mode="sparse", gamma=0.95)
        shaped_env = DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95)
        sparse = value_iteration(sparse_env, threshold=1e-8)
        shaped = value_iteration(shaped_env, threshold=1e-8)
        mask = (sparse.policy >= 0) & (shaped.policy >= 0)
        np.testing.assert_array_equal(sparse.policy[mask], shaped.policy[mask])


class QLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.maze = generate_source_map("40400356")

    def test_exact_q_update_equation_and_terminal_bootstrap(self) -> None:
        shape = (2, 16, 16, 4)
        agent = QLearningAgent(shape, alpha=0.2, gamma=0.9, rng=np.random.default_rng(1))
        state = (1, 1, 0)
        next_state = (1, 2, 0)
        agent.q[0, 1, 1, 3] = 2.0
        agent.q[0, 1, 2] = [1.0, 4.0, 3.0, 2.0]
        record = agent.update(state, 3, 5.0, next_state, terminated=False)
        expected_target = 5.0 + 0.9 * 4.0
        expected_new = 2.0 + 0.2 * (expected_target - 2.0)
        self.assertAlmostEqual(record["target"], expected_target)
        self.assertAlmostEqual(agent.q[0, 1, 1, 3], expected_new)
        terminal_record = agent.update(state, 3, 5.0, next_state, terminated=True)
        self.assertAlmostEqual(terminal_record["target"], 5.0)

    def test_epsilon_schedules_endpoints_and_monotonicity(self) -> None:
        for mode in ("linear", "exponential"):
            schedule = EpsilonSchedule(mode, 1.0, 0.05, 100)
            values = [schedule.value(index) for index in range(101)]
            self.assertAlmostEqual(values[0], 1.0)
            self.assertAlmostEqual(values[-1], 0.05)
            self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))

    def test_fixed_seed_training_reproducibility(self) -> None:
        config = QLearningConfig(episodes=12, epsilon=EpsilonSchedule("linear", 0.5, 0.05, 10))
        first = train_q_learning(DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95), config, seed=5)
        second = train_q_learning(DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95), config, seed=5)
        np.testing.assert_array_equal(first.q_table, second.q_table)
        self.assertEqual(first.episode_metrics, second.episode_metrics)

    def test_evaluation_is_greedy_and_read_only(self) -> None:
        config = QLearningConfig(episodes=20, epsilon=EpsilonSchedule("linear", 0.5, 0.05, 15))
        result = train_q_learning(DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95), config, seed=5)
        before = result.q_table.copy()
        evaluation = evaluate_q_policy(DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95), result.q_table, episodes=3, seed=123)
        np.testing.assert_array_equal(before, result.q_table)
        self.assertEqual(evaluation.metrics["episodes"], 3)
        self.assertEqual(evaluation.metrics["base_seed"], 123)
        self.assertEqual(evaluation.episodes[0]["environment_seed"], episode_seed(123, 0, stream=7))

    def test_checkpoint_round_trip(self) -> None:
        rng = np.random.default_rng(3)
        q_table = rng.normal(size=(2, 16, 16, 4))
        visits = rng.integers(0, 100, size=(2, 16, 16))
        with tempfile.TemporaryDirectory() as directory:
            path = save_q_checkpoint(Path(directory) / "q.npz", q_table, metadata={"seed": 3}, visits=visits)
            loaded_q, metadata, loaded_visits = load_q_checkpoint(path)
        np.testing.assert_array_equal(q_table, loaded_q)
        np.testing.assert_array_equal(visits, loaded_visits)
        self.assertEqual(metadata, {"seed": 3})

    def test_checkpoint_and_policy_validation_reject_silent_corruption(self) -> None:
        q_table = np.zeros((2, 16, 16, 4), dtype=np.float64)
        env = DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95)
        corrupted = q_table.copy()
        corrupted[1, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            policy_from_q(corrupted, env)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_q_checkpoint(
                    Path(directory) / "bad_visits.npz",
                    q_table,
                    metadata={"seed": 3},
                    visits=np.zeros((16, 16), dtype=np.int64),
                )

    def test_direct_agent_hyperparameters_are_validated(self) -> None:
        shape = (2, 16, 16, 4)
        with self.assertRaises(ValueError):
            QLearningAgent(shape, alpha=0.0, gamma=0.95, rng=np.random.default_rng(1))
        with self.assertRaises(ValueError):
            QLearningAgent(shape, alpha=0.2, gamma=1.0, rng=np.random.default_rng(1))


class SarsaLambdaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.maze = generate_source_map("40400356")

    def test_lambda_zero_is_one_step_sarsa_update(self) -> None:
        env = DynamicMazeEnv(self.maze, reward_mode="sparse", gamma=0.95, max_steps=1)
        config = SarsaLambdaConfig(
            episodes=1,
            alpha=0.2,
            gamma=0.95,
            lambda_value=0.0,
            epsilon=EpsilonSchedule("linear", 0.0, 0.0, 1),
            trace_type="replacing",
            trace_steps=2,
        )
        result = train_sarsa_lambda(env, config, seed=5)
        record = result.update_trace[0]
        state = tuple(record["state"])
        action = int(record["action"])
        expected = record["old_q"] + config.alpha * record["td_error_delta"]
        self.assertAlmostEqual(q_at(result.q_table, state)[action], expected)
        self.assertAlmostEqual(record["trace_l1_after_decay"], 0.0)

    def test_positive_lambda_carries_multiple_active_traces(self) -> None:
        env = DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95, max_steps=8)
        config = SarsaLambdaConfig(
            episodes=1,
            alpha=0.12,
            gamma=0.95,
            lambda_value=0.7,
            epsilon=EpsilonSchedule("linear", 0.5, 0.5, 1),
            trace_type="replacing",
            trace_steps=8,
        )
        result = train_sarsa_lambda(env, config, seed=5)
        self.assertGreater(max(row["active_traces_before_decay"] for row in result.update_trace), 1)
        self.assertTrue(np.all(np.isfinite(result.q_table)))
        record = result.update_trace[0]
        before = json.loads(record["eligibility_before_decay_json"])
        after = json.loads(record["eligibility_after_decay_json"])
        self.assertEqual(len(before), 1)
        self.assertAlmostEqual(before[0]["eligibility"], 1.0)
        self.assertAlmostEqual(after[0]["eligibility"], config.gamma * config.lambda_value)

    def test_memory_includes_eligibility_workspace(self) -> None:
        env = DynamicMazeEnv(self.maze, reward_mode="shaped", gamma=0.95, max_steps=1)
        result = train_sarsa_lambda(
            env,
            SarsaLambdaConfig(
                episodes=1,
                gamma=0.95,
                lambda_value=0.7,
                epsilon=EpsilonSchedule("linear", 0.0, 0.0, 1),
            ),
            seed=5,
        )
        expected = result.q_table.nbytes + result.visits.nbytes + result.q_table.nbytes
        self.assertEqual(result.memory_bytes, expected)
        self.assertEqual(result.auxiliary_memory_bytes, result.q_table.nbytes)


if __name__ == "__main__":
    unittest.main()
