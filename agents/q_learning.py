"""From-scratch off-policy Q-Learning implementation."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from agents.common import EpsilonSchedule, empty_q, epsilon_greedy, write_jsonl
from environments.maze import DynamicMazeEnv, State


@dataclass
class QLearningConfig:
    episodes: int = 1800
    alpha: float = 0.18
    gamma: float = 0.95
    epsilon: EpsilonSchedule = EpsilonSchedule()
    seed: int = 5
    log_all_events_for_first_n_episodes: int = 3
    collect_events: bool = True
    max_update_trace_rows: int = 50


@dataclass
class QLearningResult:
    q_values: np.ndarray
    history: pd.DataFrame
    runtime_seconds: float
    update_trace: pd.DataFrame
    event_records: list[dict[str, object]]
    watch_trace: pd.DataFrame
    visitation_counts: np.ndarray


def q_learning_update(
    q: np.ndarray,
    state: State,
    action: int,
    reward: float,
    next_state: State,
    terminated: bool,
    *,
    alpha: float,
    gamma: float,
) -> dict[str, float]:
    before = float(q[state + (action,)])
    bootstrap = 0.0 if terminated else float(np.max(q[next_state]))
    target = float(reward + gamma * bootstrap)
    td_error = target - before
    after = before + alpha * td_error
    q[state + (action,)] = after
    return {
        "q_before": before,
        "bootstrap": bootstrap,
        "target": target,
        "td_error": td_error,
        "q_after": float(after),
    }


def train_q_learning(
    env: DynamicMazeEnv,
    config: QLearningConfig,
    *,
    initial_q: np.ndarray | None = None,
    watch_state: State | None = None,
    watch_interval: int = 50,
    episode_callback: Callable[[int, dict[str, object]], None] | None = None,
) -> QLearningResult:
    if not np.isclose(env.gamma, config.gamma):
        env = env.clone(gamma=config.gamma)
    q = empty_q(env) if initial_q is None else np.array(initial_q, dtype=np.float64, copy=True)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, object]] = []
    updates: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    watched: list[dict[str, object]] = []
    visitation_counts = np.zeros((env.size, env.size), dtype=np.int64)
    started = time.perf_counter()

    for episode in range(config.episodes):
        epsilon = config.epsilon.value(episode, config.episodes)
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)), phase=episode % env.gate_period)
        total_reward = 0.0
        counters = {
            "wall_hits": 0,
            "penalty_entries": 0,
            "locked_door_attempts": 0,
            "gate_blocks": 0,
            "key_collected": 0,
            "door_crossings": 0,
        }
        terminated = truncated = False
        while not (terminated or truncated):
            visitation_counts[state[0], state[1]] += 1
            action = epsilon_greedy(q, state, epsilon, rng)
            next_state, reward, terminated, truncated, info = env.step(action)
            update = q_learning_update(
                q,
                state,
                action,
                reward,
                next_state,
                terminated,
                alpha=config.alpha,
                gamma=config.gamma,
            )
            if len(updates) < config.max_update_trace_rows:
                updates.append(
                    {
                        "episode": episode,
                        "step": env.steps,
                        "state": str(state),
                        "action": action,
                        "reward": reward,
                        "next_state": str(next_state),
                        "terminated": terminated,
                        **update,
                    }
                )
            event = str(info["event"])
            counters["wall_hits"] += int(event == "wall_collision")
            counters["penalty_entries"] += int(event == "penalty_cell_entered")
            counters["locked_door_attempts"] += int(event == "locked_door_attempt")
            counters["gate_blocks"] += int(event == "periodic_gate_blocked")
            counters["key_collected"] += int(event == "key_collected")
            counters["door_crossings"] += int(event == "door_crossed")
            if config.collect_events and (episode < config.log_all_events_for_first_n_episodes or event != "normal_move"):
                events.append(
                    {
                        "algorithm": "q_learning",
                        "episode": episode,
                        "step": env.steps,
                        "state": state,
                        "chosen_action": action,
                        "reward": reward,
                        "next_state": next_state,
                        **info,
                    }
                )
            total_reward += reward
            state = next_state

        if config.collect_events:
            events.append(
                {
                "algorithm": "q_learning",
                "episode": episode,
                "step": env.steps,
                "event": "episode_end",
                "success": bool(terminated),
                "truncated": bool(truncated),
                "return": total_reward,
            }
        )
        row = {
            "algorithm": "q_learning",
            "episode": episode,
            "return": total_reward,
            "steps": env.steps,
            "success": int(terminated),
            "epsilon": epsilon,
            **counters,
        }
        history.append(row)
        if watch_state is not None and (episode % watch_interval == 0 or episode == config.episodes - 1):
            watched.append({"episode": episode, **{f"q_action_{a}": float(q[watch_state + (a,)]) for a in range(4)}})
        if episode_callback is not None:
            episode_callback(episode, row)

    return QLearningResult(
        q_values=q,
        history=pd.DataFrame(history),
        runtime_seconds=float(time.perf_counter() - started),
        update_trace=pd.DataFrame(updates),
        event_records=events,
        watch_trace=pd.DataFrame(watched),
        visitation_counts=visitation_counts,
    )
