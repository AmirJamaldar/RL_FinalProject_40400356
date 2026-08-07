"""Generate every figure from persisted raw CSV/JSON and model checkpoints."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-rl-final-40400356")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from PIL import Image

from agents.common import load_q_checkpoint
from environments.generator import (
    DOOR,
    FLOOR,
    GOAL,
    KEY,
    PENALTY,
    PORTAL_A,
    PORTAL_B,
    START,
    WALL,
    MazeMap,
    load_map,
)


ACTION_VECTORS = {
    0: (0.0, -0.30),
    1: (0.0, 0.30),
    2: (-0.30, 0.0),
    3: (0.30, 0.0),
}
TILE_ORDER = [FLOOR, WALL, PENALTY, START, KEY, DOOR, GOAL, PORTAL_A, PORTAL_B]
TILE_COLORS = ["#f7f7f7", "#252525", "#e76f51", "#457b9d", "#f4d35e", "#9c6644", "#2a9d8f", "#9b5de5", "#6a4c93"]
TILE_LABELS = {START: "S", KEY: "K", DOOR: "D", GOAL: "G", PORTAL_A: "A", PORTAL_B: "B", PENALTY: "!"}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files if name != "metadata_json"}


def _base_array(maze: MazeMap) -> np.ndarray:
    index = {tile: number for number, tile in enumerate(TILE_ORDER)}
    return np.asarray([[index[value] for value in row] for row in maze.grid])


def draw_maze(ax, maze: MazeMap, *, annotate: bool = True, alpha: float = 1.0) -> None:
    ax.imshow(_base_array(maze), cmap=ListedColormap(TILE_COLORS), vmin=0, vmax=len(TILE_ORDER) - 1, alpha=alpha)
    ax.set_xticks(np.arange(-0.5, maze.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.height, 1), minor=True)
    ax.grid(which="minor", color="#b8b8b8", linewidth=0.35)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    if annotate:
        for row, line in enumerate(maze.grid):
            for col, tile in enumerate(line):
                if tile in TILE_LABELS:
                    color = "white" if tile in (DOOR, PORTAL_B) else "black"
                    ax.text(col, row, TILE_LABELS[tile], ha="center", va="center", fontsize=7, fontweight="bold", color=color)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    try:
        for _attempt in range(2):
            try:
                buffer = BytesIO()
                fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
                payload = buffer.getvalue()
                with Image.open(BytesIO(payload)) as image:
                    image.verify()
                path.write_bytes(payload)
                with Image.open(path) as image:
                    image.verify()
                return
            except Exception as error:
                last_error = error
        raise OSError(f"Failed to write a valid PNG after two attempts: {path}") from last_error
    finally:
        plt.close(fig)


def _rolling_by_seed(frame: pd.DataFrame, metric: str, window: int = 25) -> pd.DataFrame:
    pieces = []
    for (_, seed), group in frame.groupby(["variant", "seed"]):
        ordered = group.sort_values("episode").copy()
        ordered[metric] = ordered[metric].rolling(window, min_periods=1).mean()
        pieces.append(ordered)
    return pd.concat(pieces, ignore_index=True) if pieces else frame.copy()


def plot_variant_curves(
    frame: pd.DataFrame,
    *,
    variants: list[str],
    labels: dict[str, str],
    title: str,
    path: Path,
    colors: dict[str, str] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for metric, axis, ylabel in (("return", axes[0], "Smoothed episode return"), ("success", axes[1], "Smoothed success rate")):
        for variant in variants:
            subset = frame[frame["variant"] == variant]
            pivots = []
            for _, run in subset.groupby("seed"):
                ordered = run.sort_values("episode")
                pivots.append(ordered.set_index("episode")[metric].rolling(25, min_periods=1).mean())
            if not pivots:
                continue
            values = pd.concat(pivots, axis=1)
            mean = values.mean(axis=1)
            std = values.std(axis=1, ddof=0)
            color = None if colors is None else colors.get(variant)
            axis.plot(mean.index, mean.values, label=labels.get(variant, variant), color=color, linewidth=1.8)
            axis.fill_between(mean.index, (mean - std).to_numpy(), (mean + std).to_numpy(), color=color, alpha=0.16)
        axis.set_xlabel("Episode")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        if metric == "success":
            axis.set_ylim(-0.03, 1.03)
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, path)


def plot_map_set(maps: dict[str, MazeMap], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
    for axis, (name, maze) in zip(axes, maps.items()):
        draw_maze(axis, maze)
        axis.set_title(name.capitalize())
    fig.suptitle("BFS-validated source and transfer maps")
    fig.tight_layout()
    save_figure(fig, path)


def plot_value_heatmap(maze: MazeMap, values: np.ndarray, policy: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    finite_values = []
    for key_status in (0, 1):
        mask = (policy[key_status] < 0) | np.asarray([[tile == WALL for tile in row] for row in maze.grid])
        finite_values.extend(values[key_status][~mask].tolist())
    vmin, vmax = min(finite_values), max(finite_values)
    image = None
    for key_status, axis in enumerate(axes):
        mask = (policy[key_status] < 0) | np.asarray([[tile == WALL for tile in row] for row in maze.grid])
        image = axis.imshow(np.ma.masked_where(mask, values[key_status]), cmap="viridis", vmin=vmin, vmax=vmax)
        draw_maze(axis, maze, alpha=0.18)
        axis.set_title("Before key" if key_status == 0 else "After key")
    fig.colorbar(image, ax=axes, shrink=0.78, label="Optimal V(s)")
    fig.suptitle("Value Iteration heatmap - shaped reward")
    save_figure(fig, path)


def plot_policy(maze: MazeMap, policy: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for key_status, axis in enumerate(axes):
        draw_maze(axis, maze)
        for row in range(maze.height):
            for col in range(maze.width):
                action = int(policy[key_status, row, col])
                if action < 0:
                    continue
                dx, dy = ACTION_VECTORS[action]
                axis.arrow(col, row, dx, dy, width=0.018, head_width=0.16, head_length=0.12, color="#0b132b", length_includes_head=True)
        axis.set_title("Policy before key" if key_status == 0 else "Policy after key")
    fig.suptitle("Optimal policy with terminal and semantic tiles")
    fig.tight_layout()
    save_figure(fig, path)


def plot_visitation(maze: MazeMap, visits: np.ndarray, path: Path) -> None:
    aggregate = visits.sum(axis=0)
    mask = np.asarray([[tile == WALL for tile in row] for row in maze.grid])
    fig, axis = plt.subplots(figsize=(5.3, 4.8))
    image = axis.imshow(np.ma.masked_where(mask, np.log1p(aggregate)), cmap="magma")
    draw_maze(axis, maze, alpha=0.16)
    fig.colorbar(image, ax=axis, shrink=0.82, label="log(1 + visits)")
    axis.set_title("Q-Learning state visitation - designated seed 5")
    fig.tight_layout()
    save_figure(fig, path)


def plot_trajectory(maze: MazeMap, states: list[list[int]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.3, 4.8))
    draw_maze(axis, maze)
    coordinates = np.asarray([[state[1], state[0]] for state in states])
    axis.plot(coordinates[:, 0], coordinates[:, 1], color="#00b4d8", linewidth=2.0, alpha=0.9)
    axis.scatter(coordinates[0, 0], coordinates[0, 1], color="#0077b6", s=55, label="start", zorder=5)
    axis.scatter(coordinates[-1, 0], coordinates[-1, 1], color="#d00000", s=55, label="end", zorder=5)
    axis.legend(loc="upper right", fontsize=8)
    axis.set_title(f"Representative greedy evaluation path ({len(states) - 1} steps)")
    fig.tight_layout()
    save_figure(fig, path)


def plot_policy_mismatch(maze: MazeMap, vi_policy: np.ndarray, models: dict[str, np.ndarray], path: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5.1 * len(models), 4.4))
    axes = np.atleast_1d(axes)
    cmap = ListedColormap(["#d8f3dc", "#ef233c"])
    wall_mask = np.asarray([[tile == WALL for tile in row] for row in maze.grid])
    for axis, (label, q_table) in zip(axes, models.items()):
        mismatch = np.zeros((maze.height, maze.width), dtype=float)
        valid = np.zeros_like(mismatch, dtype=bool)
        for key_status in (0, 1):
            layer_valid = vi_policy[key_status] >= 0
            valid |= layer_valid
            mismatch += layer_valid & (np.argmax(q_table[key_status], axis=-1) != vi_policy[key_status])
        display = (mismatch > 0).astype(float)
        axis.imshow(np.ma.masked_where(~valid | wall_mask, display), cmap=cmap, vmin=0, vmax=1)
        draw_maze(axis, maze, alpha=0.18)
        axis.set_title(label + "\nred = differs in >=1 key layer")
    fig.suptitle("Model-free greedy policy disagreement with Value Iteration")
    fig.tight_layout()
    save_figure(fig, path)


def plot_algorithm_comparison(summary: pd.DataFrame, path: Path) -> None:
    selected = pd.concat(
        [
            summary[summary["algorithm"] == "value_iteration"],
            summary[(summary["algorithm"] == "q_learning") & (summary["variant"] == "linear") & (summary["reward_mode"] == "shaped")],
            summary[(summary["algorithm"] == "sarsa_lambda") & (summary["variant"] == "lambda_0.7")],
        ],
        ignore_index=True,
    )
    selected["method"] = selected["algorithm"].map({"value_iteration": "Value Iteration", "q_learning": "Q-Learning", "sarsa_lambda": "SARSA(λ=0.7)"})
    methods = ["Value Iteration", "Q-Learning", "SARSA(λ=0.7)"]
    metrics = [
        ("runtime_seconds", "Runtime (s)"),
        ("total_samples", "Environment samples"),
        ("memory_bytes", "Memory (bytes)"),
        ("eval_success_rate", "Evaluation success"),
        ("eval_mean_steps", "Path steps (lower is better)"),
        ("policy_agreement_percent", "Policy agreement (%)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for axis, (metric, title) in zip(axes.flat, metrics):
        means = [selected[selected["method"] == method][metric].mean() for method in methods]
        stds = [selected[selected["method"] == method][metric].std(ddof=0) if len(selected[selected["method"] == method]) > 1 else 0.0 for method in methods]
        axis.bar(range(3), means, yerr=stds, color=["#6a4c93", "#1982c4", "#8ac926"], capsize=3)
        axis.set_xticks(range(3), ["VI", "Q", "SARSA"], fontsize=8)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.22)
        if metric == "eval_success_rate":
            axis.set_ylim(0, 1.05)
    fig.suptitle("Three-algorithm comparison (mean ± across-seed SD)")
    fig.tight_layout()
    save_figure(fig, path)


def plot_vi_gamma(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    axes[0].plot(frame["gamma"], frame["iterations"], marker="o", color="#6a4c93")
    axes[0].set_ylabel("Iterations to convergence")
    axes[1].plot(frame["gamma"], frame["runtime_seconds"], marker="o", color="#1982c4")
    axes[1].set_ylabel("Runtime (s)")
    for axis in axes:
        axis.set_xlabel("Discount factor γ")
        axis.grid(alpha=0.25)
    fig.suptitle("Value Iteration sensitivity to γ (same sparse reward)")
    fig.tight_layout()
    save_figure(fig, path)


def plot_transfer_curves(frame: pd.DataFrame, path: Path) -> None:
    targets = ["similar", "different"]
    colors = plt.cm.tab10(np.linspace(0, 1, frame["scenario"].nunique()))
    color_map = dict(zip(sorted(frame["scenario"].unique()), colors))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for axis, target in zip(axes, targets):
        target_frame = frame[frame["target"] == target]
        for scenario, group in target_frame.groupby("scenario"):
            series = []
            for _, run in group.groupby("seed"):
                ordered = run.sort_values("episode")
                series.append(ordered.set_index("episode")["success"].rolling(25, min_periods=1).mean())
            values = pd.concat(series, axis=1)
            mean, std = values.mean(axis=1), values.std(axis=1, ddof=0)
            axis.plot(mean.index, mean, label=scenario, color=color_map[scenario], linewidth=1.5)
            axis.fill_between(mean.index, (mean - std).to_numpy(), (mean + std).to_numpy(), color=color_map[scenario], alpha=0.11)
        axis.set_title(target.capitalize() + " target")
        axis.set_xlabel("Target-training episode")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Smoothed success rate")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Transfer learning speed: all six initialization scenarios")
    fig.tight_layout()
    save_figure(fig, path)


def plot_transfer_performance(summary: pd.DataFrame, path: Path) -> None:
    ordered_scenarios = ["scratch", "full", "scaled_beta_0.25", "scaled_beta_0.50", "scaled_beta_0.75", "selective"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for row_index, target in enumerate(("similar", "different")):
        target_frame = summary[summary["target"] == target]
        aggregate = target_frame.groupby("scenario").agg(initial=("initial_success_rate", "mean"), final=("final_success_rate", "mean"), return_final=("final_mean_return", "mean"), agreement=("policy_agreement_percent", "mean"))
        labels = [scenario for scenario in ordered_scenarios if scenario in aggregate.index]
        x = np.arange(len(labels))
        axis = axes[row_index, 0]
        axis.bar(x - 0.18, [aggregate.loc[label, "initial"] for label in labels], 0.36, label="initial")
        axis.bar(x + 0.18, [aggregate.loc[label, "final"] for label in labels], 0.36, label="final")
        axis.set_ylim(0, 1.05)
        axis.set_ylabel(target.capitalize() + " success rate")
        axis.grid(axis="y", alpha=0.2)
        if row_index == 0:
            axis.legend(fontsize=8)
        axis = axes[row_index, 1]
        axis.bar(x, [aggregate.loc[label, "agreement"] for label in labels], color="#2a9d8f")
        axis.set_ylabel(target.capitalize() + " policy agreement (%)")
        axis.grid(axis="y", alpha=0.2)
        for target_axis in axes[row_index]:
            target_axis.set_xticks(x, [label.replace("scaled_beta_", "β=") for label in labels], rotation=25, ha="right", fontsize=7)
    fig.suptitle("Initial vs final transfer performance and final policy quality")
    fig.tight_layout()
    save_figure(fig, path)


def plot_transfer_q_change(source_q: np.ndarray, targets: dict[str, np.ndarray], maps: dict[str, MazeMap], path: Path) -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(5.3 * len(targets), 4.4))
    axes = np.atleast_1d(axes)
    image = None
    maximum = max(float(np.max(np.abs(q - source_q))) for q in targets.values())
    for axis, (name, q_table) in zip(axes, targets.items()):
        delta = np.max(np.abs(q_table - source_q), axis=(0, 3))
        wall_mask = np.asarray([[tile == WALL for tile in row] for row in maps[name].grid])
        image = axis.imshow(np.ma.masked_where(wall_mask, delta), cmap="inferno", vmin=0, vmax=maximum)
        draw_maze(axis, maps[name], alpha=0.15)
        axis.set_title(name.capitalize() + " target")
    fig.colorbar(image, ax=axes, shrink=0.8, label="max |Q_after - Q_transferred|")
    fig.suptitle("Q-table change after full transfer and target retraining")
    save_figure(fig, path)


def generate_all_figures(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    raw = root / "results/raw_data"
    models = root / "results/models"
    figures = root / "results/figures"
    maps = {name: load_map(root / f"environments/maps/{name}_map.json") for name in ("source", "similar", "different")}
    config = json.loads((raw / "executed_config.json").read_text(encoding="utf-8"))
    base_seed = int(config["seeds"][0])
    training = pd.read_csv(raw / "training_episodes.csv")
    summary = pd.read_csv(raw / "run_summary.csv")
    vi_gamma = pd.read_csv(raw / "vi_gamma_sensitivity.csv")
    transfer_episodes = pd.read_csv(raw / "transfer_episodes.csv")
    transfer_summary = pd.read_csv(raw / "transfer_summary.csv")
    paths = json.loads((raw / "representative_paths.json").read_text(encoding="utf-8"))
    vi = _load_npz(models / "vi_shaped_reference.npz")
    q_linear, _, q_visits = load_q_checkpoint(models / f"q_shaped_linear_seed_{base_seed}.npz")
    sarsa, _, _ = load_q_checkpoint(models / f"sarsa_shaped_lambda_0p7_seed_{base_seed}.npz")

    manifest: dict[str, list[str]] = {}

    def record(filename: str, inputs: list[str], callback: Callable[[Path], None]) -> None:
        destination = figures / filename
        callback(destination)
        manifest[filename] = inputs

    record("maps.png", ["environments/maps/*_map.json"], lambda path: plot_map_set(maps, path))
    record("value_heatmap.png", ["results/models/vi_shaped_reference.npz"], lambda path: plot_value_heatmap(maps["source"], vi["values"], vi["policy"], path))
    record("value_iteration_policy.png", ["results/models/vi_shaped_reference.npz"], lambda path: plot_policy(maps["source"], vi["policy"], path))
    record("q_visitation.png", [f"results/models/q_shaped_linear_seed_{base_seed}.npz"], lambda path: plot_visitation(maps["source"], q_visits, path))
    record("q_final_trajectory.png", ["results/raw_data/representative_paths.json"], lambda path: plot_trajectory(maps["source"], paths["q_learning"], path))
    record("policy_mismatch.png", ["results/models/vi_shaped_reference.npz", f"results/models/q_shaped_linear_seed_{base_seed}.npz", f"results/models/sarsa_shaped_lambda_0p7_seed_{base_seed}.npz"], lambda path: plot_policy_mismatch(maps["source"], vi["policy"], {"Q-Learning": q_linear, "SARSA(λ=0.7)": sarsa}, path))
    record("algorithm_comparison.png", ["results/raw_data/run_summary.csv"], lambda path: plot_algorithm_comparison(summary, path))
    record("vi_gamma_sensitivity.png", ["results/raw_data/vi_gamma_sensitivity.csv"], lambda path: plot_vi_gamma(vi_gamma, path))

    q_shaped = training[(training["algorithm"] == "q_learning") & (training["reward_mode"] == "shaped")]
    record("q_epsilon_schedules.png", ["results/raw_data/training_episodes.csv"], lambda path: plot_variant_curves(q_shaped, variants=["linear", "exponential"], labels={"linear": "linear ε", "exponential": "exponential ε"}, title="Q-Learning epsilon-schedule sensitivity", path=path))
    sarsa_frame = training[(training["algorithm"] == "sarsa_lambda") & (training["reward_mode"] == "shaped")]
    lambda_variants = [f"lambda_{float(value)}" for value in config["sarsa_lambdas"]]
    record("sarsa_lambda_sensitivity.png", ["results/raw_data/training_episodes.csv"], lambda path: plot_variant_curves(sarsa_frame, variants=lambda_variants, labels={value: value.replace("lambda_", "λ=") for value in lambda_variants}, title="SARSA(λ) sensitivity", path=path))
    reward_frame = training[(training["algorithm"] == "q_learning") & (training["variant"] == "linear")].copy()
    reward_frame["variant"] = reward_frame["reward_mode"]
    record("reward_design_comparison.png", ["results/raw_data/training_episodes.csv", "results/raw_data/reward_design_verification.json"], lambda path: plot_variant_curves(reward_frame, variants=["sparse", "shaped"], labels={"sparse": "sparse", "shaped": "potential-shaped"}, title="Sparse versus potential-shaped reward", path=path))
    record("transfer_learning_curves.png", ["results/raw_data/transfer_episodes.csv"], lambda path: plot_transfer_curves(transfer_episodes, path))
    record("transfer_performance.png", ["results/raw_data/transfer_summary.csv"], lambda path: plot_transfer_performance(transfer_summary, path))
    transferred_targets = {}
    for name in ("similar", "different"):
        transferred_targets[name], _, _ = load_q_checkpoint(models / f"transfer_{name}_full_seed_{base_seed}.npz")
    record("transfer_q_change.png", [f"results/models/q_shaped_linear_seed_{base_seed}.npz", f"results/models/transfer_similar_full_seed_{base_seed}.npz", f"results/models/transfer_different_full_seed_{base_seed}.npz"], lambda path: plot_transfer_q_change(q_linear, transferred_targets, maps, path))

    manifest_path = raw / "figure_manifest.json"
    for filename in manifest:
        with Image.open(figures / filename) as image:
            image.verify()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    generate_all_figures(Path(__file__).resolve().parents[1])
