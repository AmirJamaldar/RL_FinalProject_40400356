"""Statistical summaries and visual analytics generated only from raw data."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from environments.maze import ACTION_DELTAS, Action, DynamicMazeEnv


ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→", -1: "·"}


def rolling_metrics(frame: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    out = frame.copy()
    for column in ("return", "steps", "success", "wall_hits", "penalty_entries"):
        if column in out:
            out[f"rolling_{column}"] = out[column].rolling(window, min_periods=max(5, window // 5)).mean()
    return out


def aggregate_runs(frame: pd.DataFrame, group_columns: list[str], value: str, window: int = 100) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for keys, group in frame.groupby(group_columns + ["seed"], dropna=False):
        group = group.sort_values("episode").copy()
        group["smoothed"] = group[value].rolling(window, min_periods=max(5, window // 5)).mean()
        rows.append(group)
    smoothed = pd.concat(rows, ignore_index=True)
    aggregate = smoothed.groupby(group_columns + ["episode"], dropna=False)["smoothed"].agg(["mean", "std", "count"]).reset_index()
    aggregate["ci95"] = 1.96 * aggregate["std"].fillna(0.0) / np.sqrt(aggregate["count"].clip(lower=1))
    return aggregate


def plot_learning_curves(
    frame: pd.DataFrame,
    *,
    group_column: str,
    value: str,
    output_path: str | Path,
    ylabel: str,
    window: int = 100,
    title: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_runs(frame, [group_column], value, window=window)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for name, group in aggregate.groupby(group_column, dropna=False):
        line = ax.plot(group["episode"], group["mean"], label=str(name))[0]
        ax.fill_between(
            group["episode"].to_numpy(),
            (group["mean"] - group["ci95"]).to_numpy(),
            (group["mean"] + group["ci95"]).to_numpy(),
            alpha=0.18,
            color=line.get_color(),
        )
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def position_projection(array: np.ndarray, *, has_key: int = 0, phase: int = 0, reduce_actions: bool = False) -> np.ndarray:
    if reduce_actions:
        return np.max(array[:, :, has_key, phase, :], axis=-1)
    return array[:, :, has_key, phase]


def _masked_position_values(env: DynamicMazeEnv, values: np.ndarray) -> np.ma.MaskedArray:
    mask = env.grid == "#"
    return np.ma.array(values, mask=mask)


def plot_value_heatmap(
    env: DynamicMazeEnv,
    values: np.ndarray,
    output_path: str | Path,
    *,
    has_key: int,
    phase: int,
    title: str,
) -> None:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(_masked_position_values(env, values[:, :, has_key, phase]))
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(image, ax=ax, label="Value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_policy_map(
    env: DynamicMazeEnv,
    policy: np.ndarray,
    output_path: str | Path,
    *,
    has_key: int,
    phase: int,
    title: str,
) -> None:
    output_path = Path(output_path)
    background = np.where(env.grid == "#", np.nan, 0.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(background)
    for r, c in env.valid_positions:
        tile = str(env.grid[r, c])
        if tile in {"T"}:
            label = "T"
        elif tile in {"S", "K", "D", "G", "P"}:
            label = tile
        else:
            label = ARROWS.get(int(policy[r, c, has_key, phase]), "·")
        ax.text(c, r, label, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def visitation_counts(history_states: Iterable[tuple[int, int, int, int]], env: DynamicMazeEnv) -> np.ndarray:
    counts = np.zeros((env.size, env.size), dtype=int)
    for state in history_states:
        counts[state[0], state[1]] += 1
    return counts


def plot_visitation_heatmap(env: DynamicMazeEnv, counts: np.ndarray, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(_masked_position_values(env, counts.astype(float)))
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(image, ax=ax, label="Visits")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def policy_from_q(q: np.ndarray) -> np.ndarray:
    return np.argmax(q, axis=-1).astype(np.int8)


def policy_agreement(
    env: DynamicMazeEnv,
    candidate_policy: np.ndarray,
    reference_policy: np.ndarray,
) -> dict[str, object]:
    states = sorted(env.reachable_states())
    comparisons: list[dict[str, object]] = []
    for state in states:
        if env.is_terminal(state):
            continue
        a = int(candidate_policy[state])
        b = int(reference_policy[state])
        comparisons.append({"state": state, "candidate_action": a, "reference_action": b, "agree": int(a == b)})
    frame = pd.DataFrame(comparisons)
    return {
        "agreement_rate": float(frame["agree"].mean()),
        "state_count": int(len(frame)),
        "comparisons": frame,
    }


def plot_policy_disagreement(
    env: DynamicMazeEnv,
    comparison_frame: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    disagreement = np.full((env.size, env.size), np.nan, dtype=float)
    grouped: dict[tuple[int, int], list[int]] = {}
    for row in comparison_frame.itertuples(index=False):
        state = row.state
        grouped.setdefault((state[0], state[1]), []).append(1 - int(row.agree))
    for pos, values in grouped.items():
        disagreement[pos] = float(np.mean(values))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(disagreement, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(image, ax=ax, label="Disagreement fraction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def training_summary(frame: pd.DataFrame, tail: int = 200) -> dict[str, float]:
    last = frame.sort_values("episode").tail(tail)
    return {
        "episodes": float(frame["episode"].max() + 1),
        "final_success_rate": float(last["success"].mean()),
        "final_mean_return": float(last["return"].mean()),
        "final_mean_steps": float(last["steps"].mean()),
        "final_mean_wall_hits": float(last.get("wall_hits", pd.Series([0.0])).mean()),
        "final_mean_penalty_entries": float(last.get("penalty_entries", pd.Series([0.0])).mean()),
    }


def episodes_to_threshold(frame: pd.DataFrame, threshold: float = 0.8, window: int = 100) -> float:
    rolling = frame.sort_values("episode")["success"].rolling(window, min_periods=window).mean()
    hit = np.flatnonzero(rolling.to_numpy() >= threshold)
    return float(hit[0]) if len(hit) else math.inf


def auc_early_success(frame: pd.DataFrame, fraction: float = 0.3, window: int = 50) -> float:
    sorted_frame = frame.sort_values("episode")
    cutoff = max(2, int(len(sorted_frame) * fraction))
    curve = sorted_frame["success"].rolling(window, min_periods=5).mean().iloc[:cutoff].fillna(0.0)
    return float(np.trapz(curve.to_numpy(), dx=1.0) / max(1, len(curve) - 1))


def write_json(data: dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def plot_static_map(env: DynamicMazeEnv, output_path: str | Path, title: str) -> None:
    """Render the categorical maze map with distinct symbols."""
    output_path = Path(output_path)
    tile_values = {"#": 0, ".": 1, "S": 2, "K": 3, "D": 4, "T": 5, "P": 6, "G": 7}
    matrix = np.vectorize(tile_values.get)(env.grid)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(matrix)
    for r in range(env.size):
        for c in range(env.size):
            tile = str(env.grid[r, c])
            if tile in {"S", "K", "D", "T", "P", "G"}:
                ax.text(c, r, tile, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_trajectory(
    env: DynamicMazeEnv,
    positions: list[tuple[int, int]],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    background = np.where(env.grid == "#", np.nan, 0.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(background)
    if positions:
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        ax.plot(cols, rows, marker="o", markersize=2.5, linewidth=1.2)
        ax.scatter([cols[0]], [rows[0]], marker="s", s=55, label="Start")
        ax.scatter([cols[-1]], [rows[-1]], marker="*", s=75, label="End")
    for symbol, pos in (("K", env.key_pos), ("D", env.door_pos), ("G", env.gate_pos), ("T", env.goal_pos)):
        ax.text(pos[1], pos[0], symbol, ha="center", va="center", fontsize=9, fontweight="bold")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_q_change_heatmap(
    env: DynamicMazeEnv,
    q_before: np.ndarray,
    q_after: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    change = np.max(np.abs(q_after - q_before), axis=(2, 3, 4))
    masked = _masked_position_values(env, change)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(masked)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    fig.colorbar(image, ax=ax, label="max |Q_after - Q_before|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
