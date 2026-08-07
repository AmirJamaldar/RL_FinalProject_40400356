"""Shared tabular-agent data structures, evaluation, and checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from environments.maze import DynamicMazeEnv, State


def state_index(state: State) -> tuple[int, int, int]:
    row, col, has_key = state
    return has_key, row, col


def q_at(q_table: np.ndarray, state: State) -> np.ndarray:
    key_status, row, col = state_index(state)
    return q_table[key_status, row, col]


def deterministic_greedy_action(values: np.ndarray) -> int:
    """Stable greedy selection: smallest index among exact ties."""

    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected four finite action values, received {values}")
    return int(np.argmax(values))


def exploratory_greedy_action(values: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(np.isclose(values, np.max(values), rtol=1e-12, atol=1e-12))
    return int(rng.choice(best))


def episode_seed(base_seed: int, episode: int, stream: int = 0) -> int:
    """Deterministically separate environment RNG streams without global RNG."""

    return int((base_seed * 1_000_003 + episode * 9_176 + stream * 104_729) % (2**32 - 1))


@dataclass
class TrainingResult:
    algorithm: str
    q_table: np.ndarray
    visits: np.ndarray
    episode_metrics: list[dict]
    update_trace: list[dict]
    runtime_seconds: float
    total_samples: int
    config: dict
    # Peak algorithm-specific tabular workspace not retained in the returned
    # Q/visit arrays.  SARSA(lambda), for example, owns one eligibility array
    # with the same shape as Q during every episode.
    auxiliary_memory_bytes: int = 0

    @property
    def memory_bytes(self) -> int:
        return int(self.q_table.nbytes + self.visits.nbytes + self.auxiliary_memory_bytes)


@dataclass
class EvaluationResult:
    metrics: dict
    episodes: list[dict]
    representative_path: list[State]
    visits: np.ndarray


def policy_from_q(q_table: np.ndarray, env: DynamicMazeEnv) -> np.ndarray:
    if q_table.shape != env.q_shape or not np.all(np.isfinite(q_table)):
        raise ValueError(f"Q table must be finite with shape {env.q_shape}, received {q_table.shape}")
    policy = np.full(env.value_shape, -1, dtype=np.int8)
    for state in env.reachable_states(include_terminal=False):
        has_key, row, col = state_index(state)
        policy[has_key, row, col] = deterministic_greedy_action(q_at(q_table, state))
    return policy


def evaluate_q_policy(
    env: DynamicMazeEnv,
    q_table: np.ndarray,
    *,
    episodes: int = 100,
    seed: int = 0,
) -> EvaluationResult:
    """Evaluate greedily with epsilon exactly zero and no parameter updates."""

    if q_table.shape != env.q_shape:
        raise ValueError(f"Q shape {q_table.shape} does not match environment {env.q_shape}")
    if not np.all(np.isfinite(q_table)):
        raise ValueError("Q table contains non-finite values")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    before = q_table.copy()
    rows: list[dict] = []
    visits = np.zeros(env.value_shape, dtype=np.int64)
    successful_paths: list[list[State]] = []
    fallback_path: list[State] = []
    for episode in range(episodes):
        environment_seed = episode_seed(seed, episode, stream=7)
        state, _ = env.reset(seed=environment_seed)
        path = [state]
        total_reward = 0.0
        counts = {
            "wall_collisions": 0,
            "penalty_entries": 0,
            "closed_door_attempts": 0,
            "teleports": 0,
        }
        terminated = truncated = False
        while not (terminated or truncated):
            has_key, row, col = state_index(state)
            visits[has_key, row, col] += 1
            action = deterministic_greedy_action(q_at(q_table, state))
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            path.append(state)
            events = set(info["events"])
            counts["wall_collisions"] += int("wall_collision" in events)
            counts["penalty_entries"] += int("penalty_entry" in events)
            counts["closed_door_attempts"] += int("closed_door_attempt" in events)
            counts["teleports"] += int("teleport" in events)
        success = bool(terminated and (state[0], state[1]) == env.maze.goal)
        rows.append(
            {
                "episode": episode,
                "environment_seed": environment_seed,
                "return": total_reward,
                "steps": env.step_count,
                "success": int(success),
                "terminated": int(terminated),
                "truncated": int(truncated),
                **counts,
            }
        )
        if success:
            successful_paths.append(path)
        if not fallback_path:
            fallback_path = path
    if not np.array_equal(before, q_table):
        raise AssertionError("Evaluation mutated the Q table")
    representative = min(successful_paths, key=len) if successful_paths else fallback_path
    metrics = {
        "episodes": episodes,
        "base_seed": int(seed),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "std_return": float(np.std([row["return"] for row in rows], ddof=0)),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_wall_collisions": float(np.mean([row["wall_collisions"] for row in rows])),
        "mean_penalty_entries": float(np.mean([row["penalty_entries"] for row in rows])),
    }
    return EvaluationResult(metrics, rows, representative, visits)


def save_q_checkpoint(
    path: str | Path,
    q_table: np.ndarray,
    *,
    metadata: dict,
    visits: np.ndarray | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if q_table.ndim != 4 or q_table.shape[-1] != 4 or not np.all(np.isfinite(q_table)):
        raise ValueError("Checkpoint Q table must have shape (2,H,W,4) and finite values")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a dictionary")
    payload = {
        "q_table": np.asarray(q_table, dtype=np.float64),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    }
    if visits is not None:
        visits_array = np.asarray(visits)
        if visits_array.shape != q_table.shape[:-1]:
            raise ValueError("Checkpoint visits must match Q without its action dimension")
        if not np.issubdtype(visits_array.dtype, np.integer) or np.any(visits_array < 0):
            raise ValueError("Checkpoint visits must contain non-negative integers")
        payload["visits"] = visits_array.astype(np.int64, copy=False)
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(destination)
    return destination


def load_q_checkpoint(path: str | Path) -> tuple[np.ndarray, dict, np.ndarray | None]:
    with np.load(Path(path), allow_pickle=False) as archive:
        q_table = archive["q_table"].copy()
        metadata = json.loads(str(archive["metadata_json"].item()))
        visits = archive["visits"].copy() if "visits" in archive.files else None
    if q_table.ndim != 4 or q_table.shape[-1] != 4 or not np.all(np.isfinite(q_table)):
        raise ValueError("Invalid Q checkpoint contents")
    if not isinstance(metadata, dict):
        raise ValueError("Invalid checkpoint metadata")
    if visits is not None:
        if visits.shape != q_table.shape[:-1]:
            raise ValueError("Checkpoint visits have an incompatible shape")
        if not np.issubdtype(visits.dtype, np.integer) or np.any(visits < 0):
            raise ValueError("Checkpoint visits must contain non-negative integers")
    return q_table, metadata, visits
