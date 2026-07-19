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
    """Run synchronous Value Iteration using a precomputed tabular model.

    The Bellman backup itself is vectorized, but the transition model is still
    generated independently from ``DynamicMazeEnv.transition_outcomes``.
    No RL or planning library is used.
    """
    if not np.isclose(env.gamma, gamma):
        env = env.clone(gamma=gamma)
    values, q_values, policy = _empty_arrays(env)
    states = env.states()
    state_to_index = {state: i for i, state in enumerate(states)}
    n_states = len(states)
    max_branches = 3
    probabilities = np.zeros((n_states, env.action_space_n, max_branches), dtype=np.float64)
    rewards = np.zeros_like(probabilities)
    next_indices = np.zeros((n_states, env.action_space_n, max_branches), dtype=np.int32)
    nonterminal = np.zeros_like(probabilities)
    terminal_mask = np.array([env.is_terminal(state) for state in states], dtype=bool)

    for i, state in enumerate(states):
        for action in Action:
            outcomes = env.transition_outcomes(state, action)
            for branch, transition in enumerate(outcomes):
                probabilities[i, int(action), branch] = transition.probability
                rewards[i, int(action), branch] = transition.reward
                next_indices[i, int(action), branch] = state_to_index[transition.next_state]
                nonterminal[i, int(action), branch] = 0.0 if transition.terminated else 1.0

    v = np.zeros(n_states, dtype=np.float64)
    started = time.perf_counter()
    converged = False
    delta = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        continuation = v[next_indices]
        q_table = np.sum(probabilities * (rewards + gamma * nonterminal * continuation), axis=2)
        new_v = np.max(q_table, axis=1)
        new_v[terminal_mask] = 0.0
        delta = float(np.max(np.abs(new_v - v)))
        v = new_v
        if delta < tolerance:
            converged = True
            break

    final_continuation = v[next_indices]
    final_q = np.sum(probabilities * (rewards + gamma * nonterminal * final_continuation), axis=2)
    final_q[terminal_mask, :] = 0.0
    final_policy = np.argmax(final_q, axis=1).astype(np.int8)
    final_policy[terminal_mask] = -1

    for i, state in enumerate(states):
        values[state] = v[i]
        q_values[state] = final_q[i]
        policy[state] = final_policy[i]

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
