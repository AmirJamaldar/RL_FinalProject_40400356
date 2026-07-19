"""Independent tabular Value Iteration implementation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from environments.maze import Action, DynamicMazeEnv, State


@dataclass
class ValueIterationResult:
    values: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray
    iterations: int
    converged: bool
    final_delta: float
    runtime_seconds: float
    gamma: float
    tolerance: float


def _empty_arrays(env: DynamicMazeEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (env.size, env.size, 2, env.gate_period)
    values = np.zeros(shape, dtype=np.float64)
    q_values = np.zeros(shape + (env.action_space_n,), dtype=np.float64)
    policy = np.full(shape, -1, dtype=np.int8)
    return values, q_values, policy


def state_action_value(env: DynamicMazeEnv, values: np.ndarray, state: State, action: int, gamma: float) -> float:
    total = 0.0
    for transition in env.transition_outcomes(state, action):
        continuation = 0.0 if transition.terminated else values[transition.next_state]
        total += transition.probability * (transition.reward + gamma * continuation)
    return float(total)


def value_iteration(
    env: DynamicMazeEnv,
    *,
    gamma: float = 0.95,
    tolerance: float = 1e-8,
    max_iterations: int = 10_000,
) -> ValueIterationResult:
    if not np.isclose(env.gamma, gamma):
        env = env.clone(gamma=gamma)
    values, q_values, policy = _empty_arrays(env)
    states = env.states()
    started = time.perf_counter()
    converged = False
    delta = float("inf")
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        old = values.copy()
        delta = 0.0
        for state in states:
            if env.is_terminal(state):
                values[state] = 0.0
                continue
            candidates = [state_action_value(env, old, state, action, gamma) for action in Action]
            new_value = max(candidates)
            values[state] = new_value
            delta = max(delta, abs(new_value - old[state]))
        if delta < tolerance:
            converged = True
            break

    for state in states:
        if env.is_terminal(state):
            q_values[state] = 0.0
            policy[state] = -1
            continue
        for action in Action:
            q_values[state + (int(action),)] = state_action_value(env, values, state, int(action), gamma)
        policy[state] = int(np.argmax(q_values[state]))

    runtime = time.perf_counter() - started
    return ValueIterationResult(
        values=values,
        q_values=q_values,
        policy=policy,
        iterations=iteration,
        converged=converged,
        final_delta=float(delta),
        runtime_seconds=float(runtime),
        gamma=float(gamma),
        tolerance=float(tolerance),
    )


def save_result(result: ValueIterationResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        values=result.values,
        q_values=result.q_values,
        policy=result.policy,
        metadata=json.dumps(
            {
                "iterations": result.iterations,
                "converged": result.converged,
                "final_delta": result.final_delta,
                "runtime_seconds": result.runtime_seconds,
                "gamma": result.gamma,
                "tolerance": result.tolerance,
            }
        ),
    )
