"""Q-table transfer strategies and negative-transfer diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from agents.common import empty_q
from environments.maze import DynamicMazeEnv, State

TransferMode = Literal["scratch", "full", "scaled", "selective"]


@dataclass(frozen=True)
class TransferSummary:
    mode: str
    beta: float | None
    copied_state_count: int
    transferable_state_count: int
    copied_fraction: float


def initialize_transfer_q(
    source_env: DynamicMazeEnv,
    target_env: DynamicMazeEnv,
    source_q: np.ndarray,
    *,
    mode: TransferMode,
    beta: float = 0.5,
    neighborhood_radius: int = 1,
) -> tuple[np.ndarray, TransferSummary]:
    if source_q.shape != empty_q(source_env).shape:
        raise ValueError("source_q shape does not match source environment")
    if source_env.size != target_env.size or source_env.gate_period != target_env.gate_period:
        raise ValueError("this tabular transfer implementation requires compatible grid and phase dimensions")
    if mode == "scaled" and not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")

    target_q = empty_q(target_env)
    target_valid = set(target_env.valid_positions)
    source_valid = set(source_env.valid_positions)
    common_positions = source_valid & target_valid
    transferable_state_count = len(common_positions) * 2 * target_env.gate_period
    copied_states = 0

    if mode == "scratch":
        return target_q, TransferSummary(mode, None, 0, transferable_state_count, 0.0)

    if mode in {"full", "scaled"}:
        scale = 1.0 if mode == "full" else float(beta)
        for r, c in common_positions:
            target_q[r, c, :, :, :] = scale * source_q[r, c, :, :, :]
            copied_states += 2 * target_env.gate_period
        return target_q, TransferSummary(
            mode,
            None if mode == "full" else float(beta),
            copied_states,
            transferable_state_count,
            copied_states / max(1, transferable_state_count),
        )

    if mode == "selective":
        for r, c in common_positions:
            if source_env.local_signature(r, c, neighborhood_radius) != target_env.local_signature(
                r, c, neighborhood_radius
            ):
                continue
            target_q[r, c, :, :, :] = source_q[r, c, :, :, :]
            copied_states += 2 * target_env.gate_period
        return target_q, TransferSummary(
            mode,
            None,
            copied_states,
            transferable_state_count,
            copied_states / max(1, transferable_state_count),
        )

    raise ValueError(f"unknown transfer mode: {mode}")


def expected_action_values_from_model(
    env: DynamicMazeEnv,
    values: np.ndarray,
    state: State,
    gamma: float,
) -> np.ndarray:
    result = np.zeros(4, dtype=np.float64)
    for action in range(4):
        for transition in env.transition_outcomes(state, action):
            continuation = 0.0 if transition.terminated else values[transition.next_state]
            result[action] += transition.probability * (transition.reward + gamma * continuation)
    return result


def find_negative_transfer_example(
    target_env: DynamicMazeEnv,
    transferred_q: np.ndarray,
    target_optimal_q: np.ndarray,
    *,
    states: set[State] | None = None,
) -> dict[str, object]:
    """Find a reachable state with maximal one-step decision regret.

    Regret is measured using the target-environment optimal Q values, not the
    transferred table itself.  This makes the diagnosis environment-grounded.
    """
    candidates = states or target_env.reachable_states()
    best: dict[str, object] | None = None
    for state in candidates:
        if target_env.is_terminal(state):
            continue
        transferred_action = int(np.argmax(transferred_q[state]))
        optimal_action = int(np.argmax(target_optimal_q[state]))
        if transferred_action == optimal_action:
            continue
        optimal_values = np.asarray(target_optimal_q[state], dtype=float)
        regret = float(optimal_values[optimal_action] - optimal_values[transferred_action])
        if regret <= 1e-10:
            continue
        record = {
            "state": state,
            "transferred_action": transferred_action,
            "target_optimal_action": optimal_action,
            "regret": regret,
            "transferred_q_values": np.asarray(transferred_q[state], dtype=float).tolist(),
            "target_optimal_q_values": optimal_values.tolist(),
            "local_signature": target_env.local_signature(state[0], state[1]),
            "tile": str(target_env.grid[state[0], state[1]]),
        }
        if best is None or regret > float(best["regret"]):
            best = record
    if best is None:
        raise RuntimeError("no negative-transfer state found")
    return best
