from pathlib import Path

import numpy as np

from agents.common import empty_q
from environments.maze import DynamicMazeEnv
from transfer.transfer_learning import initialize_transfer_q

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "environments" / "maps" / "source_map.json"
SIMILAR = ROOT / "environments" / "maps" / "target_similar_map.json"
DIFFERENT = ROOT / "environments" / "maps" / "target_different_map.json"


def test_scaled_transfer_applies_beta() -> None:
    source_env = DynamicMazeEnv(SOURCE, seed=5)
    target_env = DynamicMazeEnv(SIMILAR, seed=5)
    source_q = empty_q(source_env)
    source_q.fill(2.0)
    transferred, summary = initialize_transfer_q(source_env, target_env, source_q, mode="scaled", beta=0.25)
    common = set(source_env.valid_positions) & set(target_env.valid_positions)
    r, c = next(iter(common))
    assert np.allclose(transferred[r, c], 0.5)
    assert summary.beta == 0.25


def test_selective_transfer_copies_less_after_large_change() -> None:
    source_env = DynamicMazeEnv(SOURCE, seed=5)
    target_env = DynamicMazeEnv(DIFFERENT, seed=5)
    source_q = empty_q(source_env)
    source_q.fill(1.0)
    transferred, summary = initialize_transfer_q(source_env, target_env, source_q, mode="selective")
    assert 0.0 < summary.copied_fraction < 1.0
    assert np.count_nonzero(transferred) > 0
