from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd

from agents.value_iteration import value_iteration
from environments.generator import WALL, generate_source_map, generate_transfer_map, local_signature
from environments.maze import DynamicMazeEnv
from gui.renderer import COLORS
from experiments.run_experiments import (
    code_hash,
    csv_row_count,
    load_config,
    remove_generated_files,
    validate_config,
    write_dataframe_verified,
    write_json,
)
from transfer.transfer_learning import find_negative_transfer_witness, initialize_transfer, policy_agreement


ROOT = Path(__file__).resolve().parents[1]


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = generate_source_map("40400356")
        cls.similar = generate_transfer_map(cls.source, "similar", seed=106)
        cls.different = generate_transfer_map(cls.source, "different", seed=207)
        cls.source_env = DynamicMazeEnv(cls.source, reward_mode="shaped", gamma=0.95)
        cls.source_vi = value_iteration(cls.source_env, threshold=1e-8)

    def test_four_transfer_modes_and_three_betas(self) -> None:
        q = self.source_vi.q_values
        scratch = initialize_transfer(q, self.source, self.similar, mode="scratch")
        full = initialize_transfer(q, self.source, self.similar, mode="full")
        self.assertTrue(np.all(scratch.q_table == 0))
        np.testing.assert_array_equal(full.q_table, q)
        for beta in (0.25, 0.5, 0.75):
            scaled = initialize_transfer(q, self.source, self.similar, mode="scaled", beta=beta)
            np.testing.assert_allclose(scaled.q_table, beta * q)
        selective = initialize_transfer(q, self.source, self.similar, mode="selective")
        self.assertGreater(selective.transferred_state_count, 0)
        self.assertLess(selective.transferred_state_count, int(np.prod(selective.transferred_state_mask.shape)))
        for row in range(self.source.height):
            for col in range(self.source.width):
                if selective.transferred_state_mask[0, row, col]:
                    self.assertEqual(local_signature(self.source, (row, col)), local_signature(self.similar, (row, col)))

    def test_policy_agreement_identity(self) -> None:
        result = policy_agreement(self.source_vi.q_values, self.source_vi.policy, self.source_env)
        self.assertAlmostEqual(result["agreement_percent"], 100.0)

    def test_negative_transfer_witness_uses_changed_structure_and_exact_regret(self) -> None:
        target_env = DynamicMazeEnv(self.different, reward_mode="shaped", gamma=0.95)
        target_vi = value_iteration(target_env, threshold=1e-8)
        witness = find_negative_transfer_witness(
            self.source_vi.q_values,
            self.source,
            target_env,
            target_vi.policy,
            target_vi.q_values,
        )
        self.assertIsNotNone(witness)
        self.assertGreater(witness["exact_one_state_regret"], 0)
        self.assertNotEqual(witness["source_neighborhood"], witness["target_neighborhood"])
        self.assertNotEqual(witness["transferred_action"], witness["target_optimal_action"])

    def test_renderer_defines_every_tile(self) -> None:
        self.assertTrue(set("#.SKDGXAB").issubset(COLORS))


class CliSmokeTests(unittest.TestCase):
    def test_shipped_configs_cover_required_experiment_matrix(self) -> None:
        for filename in ("quick.json", "full.json"):
            config, _ = load_config(str(ROOT / "experiments/configs" / filename))
            self.assertEqual(config["seeds"][0], 5)
            self.assertGreaterEqual(len(config["seeds"]), 3)
        default_config, default_path = load_config(None)
        self.assertEqual(default_config["profile"], "full_experiment")
        self.assertEqual(default_path.name, "full.json")
        invalid = deepcopy(default_config)
        invalid["seeds"] = [5, 5, 17]
        with self.assertRaises(ValueError):
            validate_config(invalid)
        invalid = deepcopy(default_config)
        invalid["epsilon"]["start"] = 1.2
        with self.assertRaises(ValueError):
            validate_config(invalid)

    def test_generated_output_cleanup_and_source_hash_ignore_unrelated_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "old.csv").write_text("generated", encoding="utf-8")
            (root / "old.json").write_text("{}", encoding="utf-8")
            (root / "notes.txt").write_text("keep", encoding="utf-8")
            removed = remove_generated_files(root, {".csv", ".json"})
            self.assertEqual(removed, ["old.csv", "old.json"])
            self.assertTrue((root / "notes.txt").is_file())

            (root / "agents").mkdir()
            (root / "agents" / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "experiments" / "configs").mkdir(parents=True)
            (root / "experiments" / "configs" / "full.json").write_text(
                '{"profile": "test"}\n', encoding="utf-8"
            )
            (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
            baseline_hash = code_hash(root)

            (root / ".venv" / "Lib" / "site-packages").mkdir(parents=True)
            (root / ".venv" / "Lib" / "site-packages" / "foreign.py").write_text(
                "VALUE = 999\n", encoding="utf-8"
            )
            (root / "results" / "raw_data").mkdir(parents=True)
            (root / "results" / "raw_data" / "generated.json").write_text(
                '{"generated": true}\n', encoding="utf-8"
            )
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "foreign.py").write_text("ignored\n", encoding="utf-8")
            self.assertEqual(code_hash(root), baseline_hash)

            (root / "agents" / "agent.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(code_hash(root), baseline_hash)

    def test_verified_artifact_writers_round_trip_without_staging_debris(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(
                {
                    "episode": np.arange(10_003, dtype=np.int64),
                    "return": np.linspace(-5.0, 5.0, 10_003),
                }
            )
            csv_path = root / "episodes.csv"
            json_path = root / "metadata.json"
            write_dataframe_verified(csv_path, frame)
            write_json(json_path, {"rows": len(frame), "finite": True})
            self.assertEqual(csv_row_count(csv_path), len(frame))
            pd.testing.assert_frame_equal(pd.read_csv(csv_path), frame)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["rows"], len(frame))
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["episodes.csv", "metadata.json"])

    def test_main_headless_smoke(self) -> None:
        completed = subprocess.run(
            [sys.executable, "main.py", "--headless", "--algorithm", "q", "--episodes", "2"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["algorithm"], "q_learning")
        self.assertEqual(payload["training_episodes"], 2)
        self.assertGreater(payload["samples"], 0)

    def test_experiment_entrypoint_help(self) -> None:
        completed = subprocess.run(
            [sys.executable, "experiments/run_experiments.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=15,
            check=True,
        )
        self.assertIn("--analysis-only", completed.stdout)


if __name__ == "__main__":
    unittest.main()
