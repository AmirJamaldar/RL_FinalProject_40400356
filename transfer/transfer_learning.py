"""Controlled Q-table transfer between source and target mazes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.common import deterministic_greedy_action, q_at, state_index
from environments.generator import WALL, MazeMap, local_signature
from environments.maze import ACTION_NAMES, DynamicMazeEnv


@dataclass
class TransferInitialization:
    q_table: np.ndarray
    transferred_state_mask: np.ndarray
    mode: str
    beta: float | None
    transferred_state_count: int


def initialize_transfer(
    source_q: np.ndarray,
    source_map: MazeMap,
    target_map: MazeMap,
    *,
    mode: str,
    beta: float | None = None,
) -> TransferInitialization:
    """Build scratch, full, scaled, or local-structure-selective initialization."""

    if source_q.ndim != 4 or source_q.shape != (2, source_map.height, source_map.width, 4):
        raise ValueError("source_q shape is incompatible with the source map")
    if source_map.height != target_map.height or source_map.width != target_map.width:
        raise ValueError("Tabular transfer requires maps with identical dimensions")
    if mode not in {"scratch", "full", "scaled", "selective"}:
        raise ValueError("mode must be scratch, full, scaled, or selective")
    if mode == "scaled":
        if beta not in {0.25, 0.50, 0.75}:
            raise ValueError("scaled transfer beta must be one of {0.25, 0.50, 0.75}")
    elif beta is not None:
        raise ValueError("beta is only valid for scaled transfer")

    initialized = np.zeros((2, target_map.height, target_map.width, 4), dtype=np.float64)
    mask = np.zeros((2, target_map.height, target_map.width), dtype=bool)
    if mode == "full":
        initialized[...] = source_q
        mask[...] = True
    elif mode == "scaled":
        initialized[...] = float(beta) * source_q
        mask[...] = True
    elif mode == "selective":
        for row in range(target_map.height):
            for col in range(target_map.width):
                position = (row, col)
                if source_map.tile(position) == WALL or target_map.tile(position) == WALL:
                    continue
                if local_signature(source_map, position) == local_signature(target_map, position):
                    initialized[:, row, col, :] = source_q[:, row, col, :]
                    mask[:, row, col] = True
    return TransferInitialization(
        q_table=initialized,
        transferred_state_mask=mask,
        mode=mode,
        beta=beta,
        transferred_state_count=int(np.count_nonzero(mask)),
    )


def policy_agreement(q_table: np.ndarray, reference_policy: np.ndarray, env: DynamicMazeEnv) -> dict:
    agreements = 0
    total = 0
    mismatch_states: list[tuple[int, int, int]] = []
    for state in env.reachable_states(include_terminal=False):
        has_key, row, col = state_index(state)
        reference_action = int(reference_policy[has_key, row, col])
        if reference_action < 0:
            continue
        learned_action = deterministic_greedy_action(q_at(q_table, state))
        total += 1
        if learned_action == reference_action:
            agreements += 1
        else:
            mismatch_states.append(state)
    return {
        "agreement_count": agreements,
        "state_count": total,
        "agreement_percent": 100.0 * agreements / total if total else float("nan"),
        "mismatch_states": mismatch_states,
    }


def _neighborhood_rows(maze: MazeMap, row: int, col: int, radius: int = 1) -> list[str]:
    rows: list[str] = []
    for dr in range(-radius, radius + 1):
        chars: list[str] = []
        for dc in range(-radius, radius + 1):
            rr, cc = row + dr, col + dc
            chars.append(maze.grid[rr][cc] if 0 <= rr < maze.height and 0 <= cc < maze.width else WALL)
        rows.append("".join(chars))
    return rows


def find_negative_transfer_witness(
    source_q: np.ndarray,
    source_map: MazeMap,
    target_env: DynamicMazeEnv,
    target_vi_policy: np.ndarray,
    target_vi_q: np.ndarray,
    *,
    trained_target_q: np.ndarray | None = None,
) -> dict | None:
    """Find a changed local state where transferred greed is suboptimal.

    Candidates are ranked by exact target-model action regret, not selected by
    final training performance.
    """

    candidates: list[tuple[bool, float, tuple[int, int, int], int, int]] = []
    for state in target_env.reachable_states(include_terminal=False):
        has_key, row, col = state_index(state)
        if source_map.tile((row, col)) == WALL:
            continue
        if local_signature(source_map, (row, col)) == local_signature(target_env.maze, (row, col)):
            continue
        values = q_at(source_q, state)
        if np.ptp(values) <= 1e-10:
            continue
        transferred_action = deterministic_greedy_action(values)
        optimal_action = int(target_vi_policy[has_key, row, col])
        if optimal_action < 0 or transferred_action == optimal_action:
            continue
        model_q = target_vi_q[has_key, row, col]
        if not np.all(np.isfinite(model_q)):
            continue
        regret = float(model_q[optimal_action] - model_q[transferred_action])
        if regret > 1e-10:
            corrected = False
            if trained_target_q is not None:
                corrected = deterministic_greedy_action(q_at(trained_target_q, state)) == optimal_action
            candidates.append((corrected, regret, state, transferred_action, optimal_action))
    if not candidates:
        return None
    # When post-training values are available, prefer a witness that actually
    # demonstrates the required correction; within that class maximize exact
    # target-model regret. Fall back to the strongest uncorrected witness only
    # if target training corrected none of them.
    corrected, regret, state, transferred_action, optimal_action = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    has_key, row, col = state_index(state)
    after_values = None if trained_target_q is None else q_at(trained_target_q, state).tolist()
    after_action = None if trained_target_q is None else deterministic_greedy_action(q_at(trained_target_q, state))
    return {
        "state": list(state),
        "position": [row, col],
        "has_key": bool(has_key),
        "source_neighborhood": _neighborhood_rows(source_map, row, col),
        "target_neighborhood": _neighborhood_rows(target_env.maze, row, col),
        "source_q_values": q_at(source_q, state).tolist(),
        "transferred_action": transferred_action,
        "transferred_action_name": ACTION_NAMES[transferred_action],
        "target_optimal_action": optimal_action,
        "target_optimal_action_name": ACTION_NAMES[optimal_action],
        "target_model_q_values": target_vi_q[has_key, row, col].tolist(),
        "exact_one_state_regret": regret,
        "trained_target_q_values": after_values,
        "trained_target_action": after_action,
        "trained_target_action_name": None if after_action is None else ACTION_NAMES[after_action],
        "behavior_corrected": None if after_action is None else bool(after_action == optimal_action),
    }
