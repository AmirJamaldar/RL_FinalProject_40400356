from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile
import unittest

import numpy as np

from agents.common import save_q_checkpoint
from agents.q_learning import EpsilonSchedule
from environments.generator import load_map, maze_fingerprint
from environments.maze import DynamicMazeEnv
from gui.app import MazeApp, ensure_maps, load_gui_checkpoint


class _Variable:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _Root:
    def update_idletasks(self) -> None:
        return None


class GuiEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.map_paths = ensure_maps()
        cls.source = load_map(cls.map_paths["source"])
        cls.source_env = DynamicMazeEnv(cls.source, reward_mode="shaped", gamma=0.95)

    def test_packaged_q_and_sarsa_evaluation_checkpoints_load(self) -> None:
        for algorithm in ("Q-Learning", "SARSA(λ)"):
            loaded = load_gui_checkpoint(algorithm, "source", self.source_env)
            self.assertEqual(loaded.q_table.shape, self.source_env.q_shape)
            self.assertTrue(np.all(np.isfinite(loaded.q_table)))
            self.assertGreater(np.count_nonzero(loaded.q_table), 0)

    def test_checkpoint_semantics_are_validated_not_just_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q_shaped_linear_seed_5.npz"
            save_q_checkpoint(
                path,
                np.ones(self.source_env.q_shape),
                metadata={
                    "algorithm": "sarsa_lambda",
                    "map_kind": "source",
                    "map_fingerprint": maze_fingerprint(self.source),
                    "reward_mode": "shaped",
                    "config": {"gamma": 0.95, "lambda_value": 0.7},
                },
            )
            with self.assertRaises(ValueError):
                load_gui_checkpoint("Q-Learning", "source", self.source_env, model_dir=directory)
            save_q_checkpoint(
                path,
                np.ones(self.source_env.q_shape),
                metadata={
                    "algorithm": "q_learning",
                    "map_kind": "source",
                    "map_fingerprint": "0" * 64,
                    "reward_mode": "shaped",
                    "config": {"gamma": 0.95},
                },
            )
            with self.assertRaisesRegex(ValueError, "different map contents"):
                load_gui_checkpoint("Q-Learning", "source", self.source_env, model_dir=directory)

    def test_unpackaged_sarsa_target_evaluation_fails_loudly(self) -> None:
        similar = load_map(self.map_paths["similar"])
        env = DynamicMazeEnv(similar, reward_mode="shaped", gamma=0.95)
        with self.assertRaisesRegex(FileNotFoundError, "train it in the GUI"):
            load_gui_checkpoint("SARSA(λ)", "similar", env)

    def test_mode_change_preserves_live_learned_q_table(self) -> None:
        app = MazeApp.__new__(MazeApp)
        app.running = False
        app.paused = False
        app.callback_generation = 0
        app.mode_var = _Variable("evaluation")
        app.algorithm_var = _Variable("SARSA(λ)")
        app.environment_var = _Variable("source")
        app.env = DynamicMazeEnv(self.source, reward_mode="shaped", gamma=0.95)
        app.q_table = np.random.default_rng(9).normal(size=app.env.q_shape)
        expected = app.q_table.copy()
        app.eligibility = np.zeros_like(app.q_table)
        app.training_updates = 11
        app.checkpoint_metadata = None
        app.model_source = "live GUI training"
        app.evaluation_ready = True
        app.episode = 4
        app.current_action = None
        app.event_var = _Variable("")
        app.render = lambda: None

        app._on_mode_changed()

        np.testing.assert_array_equal(app.q_table, expected)
        self.assertTrue(app.evaluation_ready)
        self.assertIn("ε is exactly zero", app.event_var.get())

    def test_rerun_from_scratch_zeros_q_resets_rng_and_enters_training(self) -> None:
        app = MazeApp.__new__(MazeApp)
        app.root = _Root()
        app.map_paths = self.map_paths
        app.algorithm_var = _Variable("Q-Learning")
        app.environment_var = _Variable("source")
        app.mode_var = _Variable("evaluation")
        app.event_var = _Variable("")
        app.running = False
        app.paused = False
        app.callback_generation = 0
        app.gamma = 0.95
        app.alpha = 0.15
        app.lambda_value = 0.7
        app.epsilon_schedule = EpsilonSchedule("linear", 1.0, 0.05, 400)
        app.recent_success = deque(maxlen=20)
        app.render = lambda: None

        app.rerun()

        self.assertEqual(app.mode_var.get(), "training")
        self.assertTrue(np.all(app.q_table == 0.0))
        self.assertEqual(app.training_updates, 0)
        self.assertEqual(app.model_source, "fresh zero Q")

    def test_resume_before_start_delegates_to_start(self) -> None:
        app = MazeApp.__new__(MazeApp)
        app.running = False
        app.paused = False
        calls = []
        app.start = lambda: calls.append("start")
        app.resume()
        self.assertEqual(calls, ["start"])


if __name__ == "__main__":
    unittest.main()
