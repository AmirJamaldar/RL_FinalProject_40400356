"""From-scratch on-policy SARSA(lambda) with eligibility traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time

import numpy as np

from environments.maze import DynamicMazeEnv
from .common import (
    TrainingResult,
    episode_seed,
    exploratory_greedy_action,
    q_at,
    state_index,
)
from .q_learning import EpsilonSchedule


@dataclass(frozen=True)
class SarsaLambdaConfig:
    episodes: int = 500
    alpha: float = 0.12
    gamma: float = 0.95
    lambda_value: float = 0.7
    epsilon: EpsilonSchedule = EpsilonSchedule()
    trace_type: str = "replacing"
    trace_steps: int = 30

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0 <= self.gamma < 1 or not 0 <= self.lambda_value <= 1:
            raise ValueError("gamma and lambda_value must lie in [0, 1]")
        if self.trace_type not in {"replacing", "accumulating"}:
            raise ValueError("trace_type must be 'replacing' or 'accumulating'")


def _select_action(q_table: np.ndarray, state: tuple[int, int, int], epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(4))
    return exploratory_greedy_action(q_at(q_table, state), rng)


def _eligibility_snapshot(eligibility: np.ndarray) -> str:
    """Serialize every numerically active trace for manual multi-step auditing."""

    entries = []
    for key_status, row, col, action in np.argwhere(np.abs(eligibility) > 1e-15):
        entries.append(
            {
                "state": [int(row), int(col), int(key_status)],
                "action": int(action),
                "eligibility": float(eligibility[key_status, row, col, action]),
            }
        )
    return json.dumps(entries, separators=(",", ":"), sort_keys=True)


def train_sarsa_lambda(
    env: DynamicMazeEnv,
    config: SarsaLambdaConfig,
    *,
    seed: int,
    initial_q: np.ndarray | None = None,
) -> TrainingResult:
    if env.reward_mode == "shaped" and not np.isclose(env.gamma, config.gamma):
        raise ValueError("Potential shaping gamma must match SARSA gamma")
    q_table = np.zeros(env.q_shape, dtype=np.float64) if initial_q is None else np.array(initial_q, dtype=np.float64, copy=True)
    if q_table.shape != env.q_shape or not np.all(np.isfinite(q_table)):
        raise ValueError("initial_q has the wrong shape or non-finite values")
    visits = np.zeros(env.value_shape, dtype=np.int64)
    rng = np.random.default_rng(seed)
    metrics: list[dict] = []
    trace_records: list[dict] = []
    total_samples = 0
    started = time.perf_counter()
    for episode in range(config.episodes):
        epsilon = config.epsilon.value(episode)
        eligibility = np.zeros_like(q_table)
        # Use the same environment-noise stream as Q-Learning for matched-seed
        # comparisons.  The agents retain independent action-selection state.
        environment_seed = episode_seed(seed, episode, stream=1)
        state, _ = env.reset(seed=environment_seed)
        action = _select_action(q_table, state, epsilon, rng)
        total_reward = 0.0
        counts = {
            "wall_collisions": 0,
            "penalty_entries": 0,
            "closed_door_attempts": 0,
            "door_passages": 0,
            "teleports": 0,
        }
        terminated = truncated = False
        while not (terminated or truncated):
            key_status, row, col = state_index(state)
            visits[key_status, row, col] += 1
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = None if terminated else _select_action(q_table, next_state, epsilon, rng)
            old_q = float(q_table[key_status, row, col, action])
            bootstrap = 0.0 if terminated else float(q_at(q_table, next_state)[next_action])
            td_error = float(reward + config.gamma * bootstrap - old_q)

            if config.trace_type == "replacing":
                eligibility[key_status, row, col, action] = 1.0
            else:
                eligibility[key_status, row, col, action] += 1.0
            record_trace = len(trace_records) < config.trace_steps
            if record_trace:
                active_before = int(np.count_nonzero(eligibility))
                trace_l1_before = float(np.sum(np.abs(eligibility)))
                snapshot_before = _eligibility_snapshot(eligibility)
            q_table += config.alpha * td_error * eligibility
            eligibility *= config.gamma * config.lambda_value
            if record_trace:
                snapshot_after = _eligibility_snapshot(eligibility)
                trace_records.append(
                    {
                        "episode": episode,
                        "step": env.step_count,
                        "state": list(state),
                        "action": action,
                        "reward": float(reward),
                        "next_state": list(next_state),
                        "next_action": next_action,
                        "old_q": old_q,
                        "bootstrap_q": bootstrap,
                        "td_error_delta": td_error,
                        "active_traces_before_decay": active_before,
                        "trace_l1_before_decay": trace_l1_before,
                        "trace_l1_after_decay": float(np.sum(np.abs(eligibility))),
                        "eligibility_before_decay_json": snapshot_before,
                        "eligibility_after_decay_json": snapshot_after,
                        "current_trace_after_decay": float(eligibility[key_status, row, col, action]),
                        "lambda": config.lambda_value,
                    }
                )
            total_samples += 1
            events = set(info["events"])
            counts["wall_collisions"] += int("wall_collision" in events)
            counts["penalty_entries"] += int("penalty_entry" in events)
            counts["closed_door_attempts"] += int("closed_door_attempt" in events)
            counts["door_passages"] += int("door_passed" in events)
            counts["teleports"] += int("teleport" in events)
            total_reward += reward
            state = next_state
            if next_action is not None:
                action = next_action
        success = bool(terminated and (state[0], state[1]) == env.maze.goal)
        metrics.append(
            {
                "episode": episode,
                "environment_seed": environment_seed,
                "return": total_reward,
                "steps": env.step_count,
                "success": int(success),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "epsilon": epsilon,
                **counts,
            }
        )
    runtime = time.perf_counter() - started
    if not np.all(np.isfinite(q_table)):
        raise FloatingPointError("SARSA(lambda) produced non-finite Q values")
    return TrainingResult(
        algorithm="sarsa_lambda",
        q_table=q_table,
        visits=visits,
        episode_metrics=metrics,
        update_trace=trace_records,
        runtime_seconds=runtime,
        total_samples=total_samples,
        config={
            **asdict(config),
            "epsilon": asdict(config.epsilon),
            "seed": seed,
        },
        auxiliary_memory_bytes=q_table.nbytes,
    )
