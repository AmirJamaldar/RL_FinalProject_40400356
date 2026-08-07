"""From-scratch off-policy tabular Q-Learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class EpsilonSchedule:
    mode: str = "linear"
    start: float = 1.0
    end: float = 0.05
    decay_episodes: int = 400

    def __post_init__(self) -> None:
        if self.mode not in {"linear", "exponential"}:
            raise ValueError("epsilon mode must be 'linear' or 'exponential'")
        if not 0 <= self.end <= self.start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if self.decay_episodes <= 0:
            raise ValueError("decay_episodes must be positive")

    def value(self, episode: int) -> float:
        fraction = min(max(episode, 0) / self.decay_episodes, 1.0)
        if self.mode == "linear":
            return float(self.start + fraction * (self.end - self.start))
        if self.start == 0:
            return 0.0
        if self.end == 0:
            # A finite, monotone approximation that reaches exactly zero at the end.
            return 0.0 if fraction >= 1.0 else float(self.start * (1e-6 ** fraction))
        return float(self.start * ((self.end / self.start) ** fraction))


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 500
    alpha: float = 0.15
    gamma: float = 0.95
    epsilon: EpsilonSchedule = EpsilonSchedule()
    trace_updates: int = 20

    def __post_init__(self) -> None:
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")


class QLearningAgent:
    def __init__(
        self,
        shape: tuple[int, int, int, int],
        *,
        alpha: float,
        gamma: float,
        rng: np.random.Generator,
        initial_q: np.ndarray | None = None,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0 <= gamma < 1:
            raise ValueError("gamma must lie in [0, 1)")
        self.q = np.zeros(shape, dtype=np.float64) if initial_q is None else np.array(initial_q, dtype=np.float64, copy=True)
        if self.q.shape != shape or not np.all(np.isfinite(self.q)):
            raise ValueError("initial_q has the wrong shape or non-finite values")
        self.alpha = alpha
        self.gamma = gamma
        self.rng = rng

    def select_action(self, state: tuple[int, int, int], epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(4))
        return exploratory_greedy_action(q_at(self.q, state), self.rng)

    def update(
        self,
        state: tuple[int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int],
        *,
        terminated: bool,
    ) -> dict:
        key_status, row, col = state_index(state)
        old_value = float(self.q[key_status, row, col, action])
        bootstrap = 0.0 if terminated else float(np.max(q_at(self.q, next_state)))
        target = float(reward + self.gamma * bootstrap)
        td_error = target - old_value
        new_value = old_value + self.alpha * td_error
        self.q[key_status, row, col, action] = new_value
        return {
            "state": list(state),
            "action": action,
            "old_q": old_value,
            "reward": float(reward),
            "next_state": list(next_state),
            "next_max_q": bootstrap,
            "terminated": int(terminated),
            "target": target,
            "td_error": td_error,
            "new_q": new_value,
        }


def train_q_learning(
    env: DynamicMazeEnv,
    config: QLearningConfig,
    *,
    seed: int,
    initial_q: np.ndarray | None = None,
) -> TrainingResult:
    if env.reward_mode == "shaped" and not np.isclose(env.gamma, config.gamma):
        raise ValueError("Potential shaping gamma must match Q-Learning gamma")
    rng = np.random.default_rng(seed)
    agent = QLearningAgent(
        env.q_shape,
        alpha=config.alpha,
        gamma=config.gamma,
        rng=rng,
        initial_q=initial_q,
    )
    visits = np.zeros(env.value_shape, dtype=np.int64)
    metrics: list[dict] = []
    update_trace: list[dict] = []
    total_samples = 0
    started = time.perf_counter()
    for episode in range(config.episodes):
        epsilon = config.epsilon.value(episode)
        environment_seed = episode_seed(seed, episode, stream=1)
        state, _ = env.reset(seed=environment_seed)
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
            has_key, row, col = state_index(state)
            visits[has_key, row, col] += 1
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            update = agent.update(
                state,
                action,
                reward,
                next_state,
                terminated=terminated,
            )
            total_samples += 1
            if len(update_trace) < config.trace_updates:
                update_trace.append(
                    {
                        "episode": episode,
                        "step": env.step_count,
                        "epsilon": epsilon,
                        **update,
                    }
                )
            events = set(info["events"])
            counts["wall_collisions"] += int("wall_collision" in events)
            counts["penalty_entries"] += int("penalty_entry" in events)
            counts["closed_door_attempts"] += int("closed_door_attempt" in events)
            counts["door_passages"] += int("door_passed" in events)
            counts["teleports"] += int("teleport" in events)
            total_reward += reward
            state = next_state
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
    if not np.all(np.isfinite(agent.q)):
        raise FloatingPointError("Q-Learning produced non-finite Q values")
    return TrainingResult(
        algorithm="q_learning",
        q_table=agent.q,
        visits=visits,
        episode_metrics=metrics,
        update_trace=update_trace,
        runtime_seconds=runtime,
        total_samples=total_samples,
        config={
            **asdict(config),
            "epsilon": asdict(config.epsilon),
            "seed": seed,
        },
    )
