"""Shared utilities for tabular reinforcement-learning agents."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from environments.maze import Action, DynamicMazeEnv, State


@dataclass(frozen=True)
class EpsilonSchedule:
    kind: str = "exponential"
    start: float = 1.0
    end: float = 0.05
    decay_fraction: float = 0.85

    def value(self, episode: int, total_episodes: int) -> float:
        if total_episodes <= 1:
            return self.end
        progress = min(max(episode / max(1, total_episodes - 1), 0.0), 1.0)
        if self.kind == "linear":
            active = min(progress / max(self.decay_fraction, 1e-12), 1.0)
            return float(self.start + active * (self.end - self.start))
        if self.kind == "exponential":
            # Calibrated so epsilon approximately reaches end at decay_fraction.
            active = min(progress / max(self.decay_fraction, 1e-12), 1.0)
            if self.start <= 0 or self.end <= 0:
                raise ValueError("exponential epsilon schedule requires positive endpoints")
            return float(self.start * ((self.end / self.start) ** active))
        if self.kind == "constant":
            return float(self.start)
        raise ValueError(f"unknown epsilon schedule: {self.kind}")


def empty_q(env: DynamicMazeEnv) -> np.ndarray:
    return np.zeros((env.size, env.size, 2, env.gate_period, env.action_space_n), dtype=np.float64)


def greedy_action(q: np.ndarray, state: State, *, rng: np.random.Generator | None = None, random_ties: bool = True) -> int:
    values = q[state]
    best = np.flatnonzero(np.isclose(values, np.max(values), rtol=1e-10, atol=1e-12))
    if random_ties and rng is not None:
        return int(rng.choice(best))
    return int(best[0])


def epsilon_greedy(q: np.ndarray, state: State, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, 4))
    return greedy_action(q, state, rng=rng, random_ties=True)


def save_q(q: np.ndarray, path: str | Path, metadata: dict[str, object] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, q=q, metadata=json.dumps(metadata or {}, ensure_ascii=False))


def load_q(path: str | Path) -> tuple[np.ndarray, dict[str, object]]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    return np.array(data["q"], dtype=np.float64), metadata


def evaluate_q_policy(
    env: DynamicMazeEnv,
    q: np.ndarray,
    *,
    episodes: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for episode in range(episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)), phase=episode % env.gate_period)
        total_reward = 0.0
        walls = penalties = gate_blocks = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action = greedy_action(q, state, rng=None, random_ties=False)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            walls += int(info["event"] == "wall_collision")
            penalties += int(info["event"] == "penalty_cell_entered")
            gate_blocks += int(info["event"] == "periodic_gate_blocked")
        rows.append(
            {
                "return": total_reward,
                "steps": env.steps,
                "success": float(terminated),
                "wall_hits": walls,
                "penalty_entries": penalties,
                "gate_blocks": gate_blocks,
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "episodes": float(episodes),
        "mean_return": float(frame["return"].mean()),
        "std_return": float(frame["return"].std(ddof=1)),
        "mean_steps": float(frame["steps"].mean()),
        "success_rate": float(frame["success"].mean()),
        "mean_wall_hits": float(frame["wall_hits"].mean()),
        "mean_penalty_entries": float(frame["penalty_entries"].mean()),
        "mean_gate_blocks": float(frame["gate_blocks"].mean()),
    }


def write_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
