"""Independent Tkinter GUI with training/evaluation playback controls."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np

from agents.common import (
    deterministic_greedy_action,
    load_q_checkpoint,
    policy_from_q,
    q_at,
    state_index,
)
from agents.q_learning import EpsilonSchedule
from agents.value_iteration import value_iteration
from environments.generator import generate_source_map, generate_transfer_map, load_map, maze_fingerprint
from environments.maze import DynamicMazeEnv, RewardSpec
from .renderer import MazeRenderer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "results/models"


@dataclass(frozen=True)
class LoadedGuiCheckpoint:
    q_table: np.ndarray
    metadata: dict
    visits: np.ndarray | None
    path: Path


GUI_CHECKPOINTS = {
    ("Q-Learning", "source"): "q_shaped_linear_seed_5.npz",
    # Scratch target runs are genuine target-trained Q-Learning policies and
    # avoid silently choosing a transfer scenario that the GUI does not expose.
    ("Q-Learning", "similar"): "transfer_similar_scratch_seed_5.npz",
    ("Q-Learning", "different"): "transfer_different_scratch_seed_5.npz",
    ("SARSA(λ)", "source"): "sarsa_shaped_lambda_0p7_seed_5.npz",
}


def load_gui_checkpoint(
    algorithm: str,
    environment: str,
    env: DynamicMazeEnv,
    *,
    model_dir: str | Path = MODEL_DIR,
) -> LoadedGuiCheckpoint:
    """Load and semantically validate the exact model used for GUI evaluation."""

    filename = GUI_CHECKPOINTS.get((algorithm, environment))
    if filename is None:
        raise FileNotFoundError(
            f"No packaged {algorithm} checkpoint exists for the {environment} map; "
            "train it in the GUI before evaluation."
        )
    path = Path(model_dir) / filename
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation checkpoint is missing: {path}")
    q_table, metadata, visits = load_q_checkpoint(path)
    if q_table.shape != env.q_shape:
        raise ValueError(f"Checkpoint shape {q_table.shape} does not match GUI environment {env.q_shape}")

    expected_algorithms = {
        "Q-Learning": {"q_learning"} if environment == "source" else {"q_learning_transfer"},
        "SARSA(λ)": {"sarsa_lambda"},
    }
    if metadata.get("algorithm") not in expected_algorithms.get(algorithm, set()):
        raise ValueError("Checkpoint algorithm metadata is incompatible with the GUI selection")
    if environment == "source":
        if metadata.get("map_kind") != "source":
            raise ValueError("Checkpoint map metadata is not the source map")
    elif metadata.get("target") != environment:
        raise ValueError("Checkpoint target metadata is incompatible with the GUI selection")
    if metadata.get("map_fingerprint") != maze_fingerprint(env.maze):
        raise ValueError("Checkpoint was trained on different map contents")
    reward_mode = metadata.get("reward_mode")
    if reward_mode is not None and reward_mode != env.reward_mode:
        raise ValueError("Checkpoint reward mode is incompatible with the GUI environment")
    checkpoint_gamma = metadata.get("config", {}).get("gamma")
    if checkpoint_gamma is not None and not np.isclose(float(checkpoint_gamma), env.gamma):
        raise ValueError("Checkpoint discount factor is incompatible with the GUI environment")
    if algorithm == "SARSA(λ)":
        checkpoint_lambda = metadata.get("config", {}).get("lambda_value")
        if checkpoint_lambda is None or not np.isclose(float(checkpoint_lambda), 0.7):
            raise ValueError("GUI SARSA evaluation requires the lambda=0.7 checkpoint")
    return LoadedGuiCheckpoint(q_table, metadata, visits, path)


def ensure_maps() -> dict[str, Path]:
    map_dir = ROOT / "environments/maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: map_dir / f"{name}_map.json" for name in ("source", "similar", "different")}
    if not all(path.exists() for path in paths.values()):
        source = generate_source_map("40400356")
        similar = generate_transfer_map(source, "similar", seed=106)
        different = generate_transfer_map(source, "different", seed=207)
        for name, maze in (("source", source), ("similar", similar), ("different", different)):
            maze.save(paths[name])
    return paths


class MazeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RL Final Project 40400356 - Dynamic Maze")
        self.map_paths = ensure_maps()
        self.algorithm_var = tk.StringVar(value="Q-Learning")
        self.environment_var = tk.StringVar(value="source")
        self.mode_var = tk.StringVar(value="training")
        self.speed_var = tk.DoubleVar(value=6.0)
        self.show_policy_var = tk.BooleanVar(value=True)
        self.info_var = tk.StringVar()
        self.event_var = tk.StringVar(value="Ready")
        self.running = False
        self.paused = False
        self.callback_generation = 0
        self.episode = 0
        self.recent_success: deque[int] = deque(maxlen=20)
        self.rng = np.random.default_rng(5)
        self.epsilon_schedule = EpsilonSchedule("linear", 1.0, 0.05, 400)
        self.gamma = 0.95
        self.alpha = 0.15
        self.lambda_value = 0.7
        self.current_action: int | None = None
        self.eligibility: np.ndarray | None = None
        self.vi_result = None
        self.checkpoint_metadata: dict | None = None
        self.model_source = "fresh zero Q"
        self.evaluation_ready = False
        self.training_updates = 0

        controls = ttk.Frame(root, padding=8)
        controls.grid(row=0, column=0, sticky="ns")
        display = ttk.Frame(root, padding=8)
        display.grid(row=0, column=1, sticky="nsew")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        ttk.Label(controls, text="Algorithm").grid(row=0, column=0, sticky="w")
        algorithm = ttk.Combobox(controls, textvariable=self.algorithm_var, values=("Value Iteration", "Q-Learning", "SARSA(λ)"), state="readonly", width=18)
        algorithm.grid(row=1, column=0, sticky="ew", pady=(0, 7))
        ttk.Label(controls, text="Environment").grid(row=2, column=0, sticky="w")
        environment = ttk.Combobox(controls, textvariable=self.environment_var, values=("source", "similar", "different"), state="readonly", width=18)
        environment.grid(row=3, column=0, sticky="ew", pady=(0, 7))
        ttk.Label(controls, text="Mode").grid(row=4, column=0, sticky="w")
        mode = ttk.Combobox(controls, textvariable=self.mode_var, values=("training", "evaluation"), state="readonly", width=18)
        mode.grid(row=5, column=0, sticky="ew", pady=(0, 7))

        ttk.Button(controls, text="Start", command=self.start).grid(row=6, column=0, sticky="ew", pady=2)
        ttk.Button(controls, text="Pause", command=self.pause).grid(row=7, column=0, sticky="ew", pady=2)
        ttk.Button(controls, text="Resume", command=self.resume).grid(row=8, column=0, sticky="ew", pady=2)
        ttk.Button(controls, text="Reset episode", command=self.reset_episode).grid(row=9, column=0, sticky="ew", pady=2)
        ttk.Button(controls, text="Rerun from scratch", command=self.rerun).grid(row=10, column=0, sticky="ew", pady=2)

        ttk.Label(controls, text="Animation speed").grid(row=11, column=0, sticky="w", pady=(10, 0))
        ttk.Scale(controls, from_=1, to=20, variable=self.speed_var, orient="horizontal").grid(row=12, column=0, sticky="ew")
        ttk.Checkbutton(controls, text="Show policy", variable=self.show_policy_var, command=self.render).grid(row=13, column=0, sticky="w", pady=6)
        ttk.Separator(controls).grid(row=14, column=0, sticky="ew", pady=6)
        ttk.Label(controls, textvariable=self.info_var, justify="left", wraplength=190).grid(row=15, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.event_var, justify="left", wraplength=190, foreground="#9d0208").grid(row=16, column=0, sticky="w", pady=(8, 0))

        self.canvas = tk.Canvas(display, background="white", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.renderer = MazeRenderer(self.canvas, cell_size=34)
        for widget in (algorithm, environment):
            widget.bind("<<ComboboxSelected>>", self._on_problem_changed)
        mode.bind("<<ComboboxSelected>>", self._on_mode_changed)
        self._initialize_learning()

    def _initialize_learning(self, *, load_evaluation_model: bool = True) -> None:
        self.maze = load_map(self.map_paths[self.environment_var.get()])
        self.env = DynamicMazeEnv(
            self.maze,
            reward_mode="shaped",
            reward_spec=RewardSpec(),
            gamma=self.gamma,
            max_steps=3 * self.maze.walkable_count,
            seed=5,
        )
        self.q_table = np.zeros(self.env.q_shape, dtype=np.float64)
        self.eligibility = np.zeros_like(self.q_table)
        self.vi_result = None
        self.rng = np.random.default_rng(5)
        self.checkpoint_metadata = None
        self.model_source = "fresh zero Q"
        self.evaluation_ready = False
        self.training_updates = 0
        status = "Fresh Q table initialized for training"
        if self.algorithm_var.get() == "Value Iteration":
            self.event_var.set("Computing exact model-based policy…")
            self.root.update_idletasks()
            self.vi_result = value_iteration(self.env, threshold=1e-8)
            self.model_source = "exact Value Iteration policy"
            self.evaluation_ready = True
            status = "Exact Value Iteration policy computed"
        elif self.mode_var.get() == "evaluation" and load_evaluation_model:
            status = self._load_evaluation_checkpoint()
        self.episode = 0
        self.recent_success.clear()
        self._begin_episode()
        self.event_var.set(status)

    def _load_evaluation_checkpoint(self) -> str:
        try:
            loaded = load_gui_checkpoint(
                self.algorithm_var.get(),
                self.environment_var.get(),
                self.env,
            )
        except (OSError, KeyError, ValueError) as error:
            self.evaluation_ready = False
            self.model_source = "no compatible checkpoint"
            return f"Evaluation unavailable: {error}"
        self.q_table[...] = loaded.q_table
        self.checkpoint_metadata = loaded.metadata
        self.model_source = loaded.path.name
        self.evaluation_ready = True
        return f"Loaded evaluation checkpoint: {loaded.path.name}"

    def _begin_episode(self) -> None:
        self.state, _ = self.env.reset(seed=5_000_003 + self.episode * 9_176)
        self.eligibility.fill(0.0)
        self.current_action = None
        self.last_events: list[str] = []
        self.event_var.set("Episode started")
        self.render()

    def current_epsilon(self) -> float:
        return self.epsilon_schedule.value(self.episode) if self.mode_var.get() == "training" else 0.0

    def _choose_action(self, state: tuple[int, int, int]) -> int:
        if self.algorithm_var.get() == "Value Iteration":
            key_status, row, col = state_index(state)
            return int(self.vi_result.policy[key_status, row, col])
        epsilon = self.current_epsilon()
        values = q_at(self.q_table, state)
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.integers(4))
        if self.mode_var.get() == "evaluation":
            return deterministic_greedy_action(values)
        best = np.flatnonzero(np.isclose(values, np.max(values)))
        return int(self.rng.choice(best))

    def _q_update(self, state, action, reward, next_state, terminated) -> None:
        key_status, row, col = state_index(state)
        target = reward if terminated else reward + self.gamma * float(np.max(q_at(self.q_table, next_state)))
        self.q_table[key_status, row, col, action] += self.alpha * (target - self.q_table[key_status, row, col, action])
        self.training_updates += 1
        self.evaluation_ready = True
        self.model_source = "live GUI training"

    def _sarsa_update(self, state, action, reward, next_state, next_action, terminated) -> None:
        key_status, row, col = state_index(state)
        bootstrap = 0.0 if terminated else float(q_at(self.q_table, next_state)[next_action])
        delta = reward + self.gamma * bootstrap - self.q_table[key_status, row, col, action]
        self.eligibility[key_status, row, col, action] = 1.0
        self.q_table += self.alpha * delta * self.eligibility
        self.eligibility *= self.gamma * self.lambda_value
        self.training_updates += 1
        self.evaluation_ready = True
        self.model_source = "live GUI training"

    def _tick(self, generation: int | None = None) -> None:
        generation = self.callback_generation if generation is None else generation
        if generation != self.callback_generation or not self.running or self.paused:
            return
        algorithm = self.algorithm_var.get()
        if self.mode_var.get() == "evaluation" and algorithm != "Value Iteration" and not self.evaluation_ready:
            self.running = False
            self.event_var.set("Evaluation blocked: train this selection or provide a compatible checkpoint.")
            return
        state = self.state
        if algorithm == "SARSA(λ)" and self.current_action is not None:
            action = self.current_action
        else:
            action = self._choose_action(state)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_action = None if terminated else self._choose_action(next_state)
        if self.mode_var.get() == "training":
            if algorithm == "Q-Learning":
                self._q_update(state, action, reward, next_state, terminated)
            elif algorithm == "SARSA(λ)":
                self._sarsa_update(state, action, reward, next_state, next_action, terminated)
        self.current_action = next_action if algorithm == "SARSA(λ)" else None
        self.state = next_state
        self.last_events = list(info["events"])
        self.event_var.set(", ".join(self.last_events))
        self.render()
        delay = max(20, int(800 / self.speed_var.get()))
        if terminated or truncated:
            success = int(terminated and (next_state[0], next_state[1]) == self.maze.goal)
            self.recent_success.append(success)
            self.episode += 1
            self.root.after(delay * 3, lambda: self._continue_after_episode(generation))
        else:
            self.root.after(delay, lambda: self._tick(generation))

    def _continue_after_episode(self, generation: int) -> None:
        if generation != self.callback_generation or not self.running:
            return
        self._begin_episode()
        if not self.paused:
            self._tick(generation)

    def policy(self) -> np.ndarray | None:
        if self.algorithm_var.get() == "Value Iteration":
            return None if self.vi_result is None else self.vi_result.policy
        return policy_from_q(self.q_table, self.env)

    def render(self) -> None:
        recent = sum(self.recent_success) / len(self.recent_success) if self.recent_success else 0.0
        self.info_var.set(
            f"Episode: {self.episode}\n"
            f"Step: {self.env.step_count}/{self.env.max_steps}\n"
            f"Reward: {self.env.total_reward:.2f}\n"
            f"ε: {self.current_epsilon():.3f}\n"
            f"Key: {'yes' if self.state[2] else 'no'}\n"
            f"Recent success: {recent:.1%}\n"
            f"Model: {self.model_source}"
        )
        self.renderer.render(self.maze, self.state, policy=self.policy(), show_policy=self.show_policy_var.get(), last_events=self.last_events)

    def start(self) -> None:
        if self.algorithm_var.get() == "Value Iteration" and self.mode_var.get() == "training":
            self.event_var.set("Value Iteration uses model backups; showing its evaluated policy.")
        if (
            self.mode_var.get() == "evaluation"
            and self.algorithm_var.get() != "Value Iteration"
            and not self.evaluation_ready
        ):
            self.running = False
            self.event_var.set("Evaluation blocked: train this selection or provide a compatible checkpoint.")
            return
        if self.running and not self.paused:
            return
        self.running = True
        self.paused = False
        self._tick(self.callback_generation)

    def pause(self) -> None:
        self.paused = True
        self.event_var.set("Paused")

    def resume(self) -> None:
        if not self.running:
            self.start()
            return
        if self.paused:
            self.paused = False
            self.event_var.set("Resumed")
            self._tick(self.callback_generation)

    def reset_episode(self) -> None:
        self.callback_generation += 1
        self._begin_episode()
        if self.running and not self.paused:
            self._tick(self.callback_generation)

    def _on_problem_changed(self, _event=None) -> None:
        was_running = self.running
        self.callback_generation += 1
        self.running = False
        self.paused = False
        self._initialize_learning(load_evaluation_model=True)
        self.running = was_running and (
            self.mode_var.get() != "evaluation"
            or self.algorithm_var.get() == "Value Iteration"
            or self.evaluation_ready
        )
        if self.running:
            self._tick(self.callback_generation)

    def _on_mode_changed(self, _event=None) -> None:
        """Change behavior policy without discarding the learned table."""

        was_running = self.running
        self.callback_generation += 1
        self.running = False
        self.paused = False
        if self.mode_var.get() == "evaluation":
            if self.algorithm_var.get() == "Value Iteration":
                self.evaluation_ready = True
                status = "Evaluation uses the exact Value Iteration policy"
            elif self.training_updates > 0 or self.checkpoint_metadata is not None:
                self.evaluation_ready = True
                status = f"Evaluation preserves {self.model_source}; ε is exactly zero"
            else:
                status = self._load_evaluation_checkpoint()
        else:
            status = (
                f"Training will continue from {self.model_source}; use Rerun from scratch to zero Q"
                if self.checkpoint_metadata is not None
                else "Training mode; current learned Q table preserved"
            )
        self._begin_episode()
        self.event_var.set(status)
        self.running = was_running and (
            self.mode_var.get() != "evaluation"
            or self.algorithm_var.get() == "Value Iteration"
            or self.evaluation_ready
        )
        if self.running:
            self._tick(self.callback_generation)

    def rerun(self) -> None:
        was_running = self.running
        self.callback_generation += 1
        self.running = False
        self.paused = False
        # "From scratch" must be literal.  Evaluation of an all-zero table is
        # meaningless, so this control explicitly returns model-free methods to
        # training mode before resetting all learning state and RNG streams.
        if self.algorithm_var.get() != "Value Iteration":
            self.mode_var.set("training")
        self._initialize_learning(load_evaluation_model=False)
        self.running = was_running
        if was_running:
            self._tick(self.callback_generation)


def launch_gui() -> None:
    root = tk.Tk()
    MazeApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
