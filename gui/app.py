"""Interactive GUI required by the final project specification."""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

from agents.common import EpsilonSchedule, greedy_action, load_q
from agents.q_learning import QLearningConfig, train_q_learning
from agents.sarsa_lambda import SarsaLambdaConfig, train_sarsa_lambda
from agents.value_iteration import value_iteration
from environments.maze import DynamicMazeEnv
from experiments.analysis import policy_from_q
from gui.renderer import MazeRenderer

ROOT = Path(__file__).resolve().parents[1]
MAP_FILES = {
    "Source": ROOT / "environments" / "maps" / "source_map.json",
    "Target - Similar": ROOT / "environments" / "maps" / "target_similar_map.json",
    "Target - Different": ROOT / "environments" / "maps" / "target_different_map.json",
}


class TrainingStopped(Exception):
    pass


class RLProjectApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("RL Final Project 40400356 - Dynamic Maze")
        self.geometry("1120x720")
        self.minsize(1000, 680)
        self.env = DynamicMazeEnv(MAP_FILES["Source"], reward_mode="shaping", gamma=0.95, seed=5)
        self.state, _ = self.env.reset()
        self.policy: np.ndarray | None = None
        self.q_values: np.ndarray | None = None
        self.animation_job: str | None = None
        self.paused = False
        self.running = False
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.recent_success: list[int] = []
        self._build_ui()
        self.renderer = MazeRenderer(self.canvas, self.env, cell_size=37)
        self._draw()
        self.after(100, self._process_worker_queue)

    def _build_ui(self) -> None:
        controls = ttk.Frame(self, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)
        display = ttk.Frame(self, padding=10)
        display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(controls, text="Configuration", font=("Arial", 13, "bold")).pack(anchor="w", pady=(0, 8))
        self.algorithm_var = tk.StringVar(value="Q-Learning")
        self.environment_var = tk.StringVar(value="Source")
        self.mode_var = tk.StringVar(value="Evaluation")
        self.show_policy_var = tk.BooleanVar(value=True)
        self.speed_var = tk.DoubleVar(value=0.20)

        for label, variable, values in (
            ("Algorithm", self.algorithm_var, ("Value Iteration", "Q-Learning", "SARSA(lambda)")),
            ("Environment", self.environment_var, tuple(MAP_FILES)),
            ("Mode", self.mode_var, ("Training", "Evaluation")),
        ):
            ttk.Label(controls, text=label).pack(anchor="w")
            box = ttk.Combobox(controls, textvariable=variable, values=values, state="readonly", width=22)
            box.pack(anchor="w", pady=(0, 8))
            if label == "Environment":
                box.bind("<<ComboboxSelected>>", lambda _event: self._change_environment())

        ttk.Checkbutton(controls, text="Show policy", variable=self.show_policy_var, command=self._draw).pack(anchor="w", pady=4)
        ttk.Label(controls, text="Animation delay (seconds)").pack(anchor="w", pady=(8, 0))
        ttk.Scale(controls, from_=0.03, to=0.8, variable=self.speed_var, orient=tk.HORIZONTAL, length=190).pack(anchor="w")

        button_frame = ttk.Frame(controls)
        button_frame.pack(anchor="w", pady=12)
        ttk.Button(button_frame, text="Start", command=self.start).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="Pause", command=self.pause).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(button_frame, text="Continue", command=self.resume).grid(row=1, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="Stop", command=self.stop).grid(row=1, column=1, padx=2, pady=2)
        ttk.Button(button_frame, text="Reset", command=self.reset).grid(row=2, column=0, padx=2, pady=2)
        ttk.Button(button_frame, text="Run again", command=self.rerun).grid(row=2, column=1, padx=2, pady=2)

        ttk.Separator(controls).pack(fill=tk.X, pady=8)
        ttk.Label(controls, text="Live information", font=("Arial", 12, "bold")).pack(anchor="w")
        self.info_vars = {name: tk.StringVar(value="-") for name in ("Episode", "Step", "Reward", "Epsilon", "Key", "Recent success", "Gate")}
        for name, variable in self.info_vars.items():
            row = ttk.Frame(controls)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{name}:", width=14).pack(side=tk.LEFT)
            ttk.Label(row, textvariable=variable, width=12).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(controls, textvariable=self.status_var, wraplength=220).pack(anchor="w", pady=(12, 0))

        self.canvas = tk.Canvas(display, background="white", highlightthickness=1, highlightbackground="#B0BEC5")
        self.canvas.pack(expand=True)
        legend = ttk.Label(display, text="S=start  K=key  D=door  T=target  !=penalty  G=periodic gate  A=agent")
        legend.pack(pady=5)

    def _change_environment(self) -> None:
        self.stop()
        self.env = DynamicMazeEnv(MAP_FILES[self.environment_var.get()], reward_mode="shaping", gamma=0.95, seed=5)
        self.state, _ = self.env.reset()
        self.policy = None
        self.q_values = None
        self.renderer = MazeRenderer(self.canvas, self.env, cell_size=37)
        self.status_var.set("Environment changed; load/train a policy.")
        self._draw()

    def _draw(self, message: str = "") -> None:
        self.renderer.draw(self.state, self.policy, self.show_policy_var.get(), message=message)
        self.info_vars["Step"].set(str(self.env.steps))
        self.info_vars["Reward"].set(f"{self.env.episode_return:.2f}")
        self.info_vars["Key"].set("yes" if self.state[2] else "no")
        self.info_vars["Gate"].set("open" if self.env.is_gate_open(self.state[3]) else "closed")
        recent = np.mean(self.recent_success[-50:]) if self.recent_success else 0.0
        self.info_vars["Recent success"].set(f"{recent:.1%}")

    def _candidate_model_path(self) -> Path | None:
        algorithm = self.algorithm_var.get()
        env_name = self.environment_var.get()
        if env_name != "Source":
            return None
        if algorithm == "Q-Learning":
            candidates = sorted((ROOT / "results" / "models").glob("q_learning_*_seed_5.npz"))
            return candidates[0] if candidates else None
        if algorithm == "SARSA(lambda)":
            candidates = sorted((ROOT / "results" / "models").glob("sarsa_lambda_*_seed_5.npz"))
            return candidates[0] if candidates else None
        return None

    def _load_or_compute_policy(self) -> bool:
        algorithm = self.algorithm_var.get()
        if algorithm == "Value Iteration":
            result = value_iteration(self.env, gamma=0.95, tolerance=1e-8)
            self.q_values = result.q_values
            self.policy = result.policy
            self.status_var.set(f"Value Iteration converged in {result.iterations} iterations.")
            return True
        model = self._candidate_model_path()
        if model and model.exists():
            self.q_values, _ = load_q(model)
            self.policy = policy_from_q(self.q_values)
            self.status_var.set(f"Loaded {model.name}")
            return True
        self.status_var.set("No saved model for this selection; use Training mode first.")
        return False

    def start(self) -> None:
        if self.running:
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.paused = False
        if self.mode_var.get() == "Training":
            self._start_training()
        else:
            if not self._load_or_compute_policy():
                return
            self.state, _ = self.env.reset(phase=0)
            self.running = True
            self._animate_step()

    def _start_training(self) -> None:
        self.running = True
        self.status_var.set("Training in progress...")
        algorithm = self.algorithm_var.get()

        def callback(episode: int, row: dict[str, object]) -> None:
            if self.stop_event.is_set():
                raise TrainingStopped()
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.05)
            self.worker_queue.put(("progress", (episode, row)))

        def worker() -> None:
            try:
                if algorithm == "Value Iteration":
                    result = value_iteration(self.env.clone(seed=5), gamma=0.95, tolerance=1e-8)
                    self.worker_queue.put(("complete", (result.q_values, result.policy, f"Converged in {result.iterations} Bellman iterations")))
                elif algorithm == "Q-Learning":
                    config = QLearningConfig(episodes=700, alpha=0.22, gamma=0.95, epsilon=EpsilonSchedule("exponential", 1.0, 0.05, 0.88), seed=5, collect_events=False, max_update_trace_rows=0)
                    result = train_q_learning(self.env.clone(seed=5), config, episode_callback=callback)
                    self.worker_queue.put(("complete", (result.q_values, policy_from_q(result.q_values), f"Q-Learning completed in {result.runtime_seconds:.1f}s")))
                else:
                    config = SarsaLambdaConfig(episodes=700, alpha=0.04, gamma=0.95, lambda_value=0.7, epsilon=EpsilonSchedule("exponential", 1.0, 0.05, 0.88), seed=5)
                    result = train_sarsa_lambda(self.env.clone(seed=5), config, episode_callback=callback)
                    self.worker_queue.put(("complete", (result.q_values, policy_from_q(result.q_values), f"SARSA(lambda) completed in {result.runtime_seconds:.1f}s")))
            except TrainingStopped:
                self.worker_queue.put(("stopped", None))
            except Exception as exc:
                self.worker_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _process_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "progress":
                    episode, row = payload
                    self.info_vars["Episode"].set(str(episode))
                    self.info_vars["Step"].set(str(row["steps"]))
                    self.info_vars["Reward"].set(f"{row['return']:.2f}")
                    self.info_vars["Epsilon"].set(f"{row['epsilon']:.3f}")
                    self.recent_success.append(int(row["success"]))
                    self.info_vars["Recent success"].set(f"{np.mean(self.recent_success[-50:]):.1%}")
                elif kind == "complete":
                    self.q_values, self.policy, message = payload
                    self.running = False
                    self.status_var.set(message)
                    self._draw("Training complete")
                elif kind == "stopped":
                    self.running = False
                    self.status_var.set("Training stopped")
                elif kind == "error":
                    self.running = False
                    self.status_var.set("Training failed")
                    messagebox.showerror("Error", str(payload))
        except queue.Empty:
            pass
        self.after(100, self._process_worker_queue)

    def _animate_step(self) -> None:
        if not self.running or self.paused or self.policy is None:
            return
        action = int(self.policy[self.state])
        self.state, _reward, terminated, truncated, info = self.env.step(action)
        self._draw(str(info["event"]))
        if terminated or truncated or self.stop_event.is_set():
            self.running = False
            self.recent_success.append(int(terminated))
            self.status_var.set("Successful episode" if terminated else "Episode ended without success")
            self._draw("SUCCESS" if terminated else "TRUNCATED")
            return
        self.animation_job = self.after(max(20, int(self.speed_var.get() * 1000)), self._animate_step)

    def pause(self) -> None:
        if not self.running:
            return
        self.paused = True
        self.pause_event.set()
        self.status_var.set("Paused")

    def resume(self) -> None:
        if not self.running:
            return
        self.paused = False
        self.pause_event.clear()
        self.status_var.set("Running")
        if self.mode_var.get() == "Evaluation":
            self._animate_step()

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.paused = False
        self.running = False
        if self.animation_job is not None:
            try:
                self.after_cancel(self.animation_job)
            except tk.TclError:
                pass
            self.animation_job = None
        self.status_var.set("Stopped")

    def reset(self) -> None:
        self.stop()
        self.state, _ = self.env.reset(phase=0)
        self.info_vars["Episode"].set("-")
        self.info_vars["Epsilon"].set("-")
        self.status_var.set("Reset")
        self._draw()

    def rerun(self) -> None:
        self.reset()
        self.start()


def launch() -> None:
    app = RLProjectApp()
    app.mainloop()


if __name__ == "__main__":
    launch()
