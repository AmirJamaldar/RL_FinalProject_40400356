"""Independent tabular Value Iteration implementation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np

from environments.maze import DynamicMazeEnv, State
from .common import state_index


@dataclass
class ValueIterationResult:
    values: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray
    iterations: int
    delta_history: list[float]
    runtime_seconds: float
    converged: bool
    bellman_backups: int
    gamma: float
    threshold: float

    @property
    def memory_bytes(self) -> int:
        return int(self.values.nbytes + self.q_values.nbytes + self.policy.nbytes)


def action_values(env: DynamicMazeEnv, values: np.ndarray, state: State, gamma: float) -> np.ndarray:
    result = np.empty(4, dtype=np.float64)
    for action in range(4):
        expected = 0.0
        for outcome in env.transition_distribution(state, action):
            continuation = 0.0
            if not outcome.terminated:
                has_key, row, col = state_index(outcome.next_state)
                continuation = values[has_key, row, col]
            expected += outcome.probability * (outcome.reward + gamma * continuation)
        result[action] = expected
    return result


def value_iteration(
    env: DynamicMazeEnv,
    *,
    gamma: float | None = None,
    threshold: float = 1e-8,
    max_iterations: int = 10_000,
) -> ValueIterationResult:
    """Synchronous Bellman optimality updates until max-norm convergence."""

    discount = env.gamma if gamma is None else float(gamma)
    if not 0.0 <= discount < 1.0:
        raise ValueError("gamma must lie in [0, 1)")
    if env.reward_mode == "shaped" and not np.isclose(discount, env.gamma):
        raise ValueError("Potential shaping gamma must match Value Iteration gamma")
    if threshold <= 0 or max_iterations <= 0:
        raise ValueError("threshold and max_iterations must be positive")
    states = env.reachable_states(include_terminal=False)
    values = np.zeros(env.value_shape, dtype=np.float64)
    deltas: list[float] = []
    backups = 0
    started = time.perf_counter()
    converged = False
    for iteration in range(1, max_iterations + 1):
        previous = values.copy()
        delta = 0.0
        for state in states:
            has_key, row, col = state_index(state)
            updated = float(np.max(action_values(env, previous, state, discount)))
            values[has_key, row, col] = updated
            delta = max(delta, abs(updated - previous[has_key, row, col]))
            backups += 4
        deltas.append(delta)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Non-finite values encountered in Value Iteration")
        if delta < threshold:
            converged = True
            break
    runtime = time.perf_counter() - started

    policy = np.full(env.value_shape, -1, dtype=np.int8)
    # Unreachable/wall entries remain zero and are excluded by the policy mask.
    # Finite padding keeps checkpoints safe to load and compare byte-for-byte.
    q_values = np.zeros(env.q_shape, dtype=np.float64)
    for state in states:
        has_key, row, col = state_index(state)
        q = action_values(env, values, state, discount)
        q_values[has_key, row, col] = q
        policy[has_key, row, col] = int(np.argmax(q))
    terminal_key, terminal_row, terminal_col = state_index(env.terminal_state)
    values[terminal_key, terminal_row, terminal_col] = 0.0
    policy[terminal_key, terminal_row, terminal_col] = -1
    return ValueIterationResult(
        values=values,
        q_values=q_values,
        policy=policy,
        iterations=iteration,
        delta_history=deltas,
        runtime_seconds=runtime,
        converged=converged,
        bellman_backups=backups,
        gamma=discount,
        threshold=threshold,
    )


def bellman_residual(env: DynamicMazeEnv, result: ValueIterationResult) -> float:
    residual = 0.0
    for state in env.reachable_states(include_terminal=False):
        has_key, row, col = state_index(state)
        target = np.max(action_values(env, result.values, state, result.gamma))
        residual = max(residual, abs(float(target) - result.values[has_key, row, col]))
    return float(residual)


def save_value_iteration(path: str | Path, result: ValueIterationResult, metadata: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_metadata = {
        **metadata,
        "iterations": result.iterations,
        "runtime_seconds": result.runtime_seconds,
        "converged": result.converged,
        "bellman_backups": result.bellman_backups,
        "gamma": result.gamma,
        "threshold": result.threshold,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
    np.savez_compressed(
        temporary,
        values=result.values,
        q_values=result.q_values,
        policy=result.policy,
        delta_history=np.asarray(result.delta_history),
        metadata_json=np.array(json.dumps(payload_metadata, sort_keys=True)),
    )
    temporary.replace(destination)
    return destination
