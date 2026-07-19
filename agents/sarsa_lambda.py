"""On-policy SARSA(lambda) with replacing eligibility traces."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from agents.common import EpsilonSchedule, empty_q, epsilon_greedy
from environments.maze import DynamicMazeEnv, State


@dataclass
class SarsaLambdaConfig:
    episodes: int = 1800
    alpha: float = 0.14
    gamma: float = 0.95
    lambda_value: float = 0.7
    epsilon: EpsilonSchedule = EpsilonSchedule()
    seed: int = 5
    trace_type: str = "replacing"
    trace_prune_threshold: float = 1e-8


@dataclass
class SarsaLambdaResult:
    q_values: np.ndarray
    history: pd.DataFrame
    runtime_seconds: float
    trace_log: pd.DataFrame


def train_sarsa_lambda(
    env: DynamicMazeEnv,
    config: SarsaLambdaConfig,
    *,
    initial_q: np.ndarray | None = None,
    trace_episode: int = 0,
    trace_steps: int = 16,
    episode_callback: Callable[[int, dict[str, object]], None] | None = None,
) -> SarsaLambdaResult:
    if config.trace_type not in {"replacing", "accumulating"}:
        raise ValueError("trace_type must be replacing or accumulating")
    if not np.isclose(env.gamma, config.gamma):
        env = env.clone(gamma=config.gamma)
    q = empty_q(env) if initial_q is None else np.array(initial_q, dtype=np.float64, copy=True)
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, object]] = []
    trace_records: list[dict[str, object]] = []
    started = time.perf_counter()

    for episode in range(config.episodes):
        epsilon = config.epsilon.value(episode, config.episodes)
        state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)), phase=episode % env.gate_period)
        action = epsilon_greedy(q, state, epsilon, rng)
        traces: dict[tuple[int, int, int, int, int], float] = {}
        total_reward = 0.0
        counters = {"wall_hits": 0, "penalty_entries": 0, "gate_blocks": 0}
        terminated = truncated = False

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = None if terminated or truncated else epsilon_greedy(q, next_state, epsilon, rng)
            current_key = state + (action,)
            q_before = float(q[current_key])
            bootstrap = 0.0 if terminated or truncated else float(q[next_state + (int(next_action),)])
            delta = float(reward + config.gamma * bootstrap - q_before)

            if config.trace_type == "replacing":
                traces[current_key] = 1.0
            else:
                traces[current_key] = traces.get(current_key, 0.0) + 1.0

            active_before = len(traces)
            for key, eligibility in list(traces.items()):
                q[key] += config.alpha * delta * eligibility
                new_value = config.gamma * config.lambda_value * eligibility
                if abs(new_value) < config.trace_prune_threshold:
                    del traces[key]
                else:
                    traces[key] = new_value

            if episode == trace_episode and env.steps <= trace_steps:
                top = sorted(traces.items(), key=lambda item: abs(item[1]), reverse=True)[:6]
                trace_records.append(
                    {
                        "episode": episode,
                        "step": env.steps,
                        "state": str(state),
                        "action": action,
                        "reward": reward,
                        "next_state": str(next_state),
                        "next_action": -1 if next_action is None else int(next_action),
                        "q_before": q_before,
                        "bootstrap": bootstrap,
                        "delta": delta,
                        "active_traces_before_decay": active_before,
                        "active_traces_after_decay": len(traces),
                        "top_traces": str([(str(k), round(v, 8)) for k, v in top]),
                    }
                )

            event = str(info["event"])
            counters["wall_hits"] += int(event == "wall_collision")
            counters["penalty_entries"] += int(event == "penalty_cell_entered")
            counters["gate_blocks"] += int(event == "periodic_gate_blocked")
            total_reward += reward
            state = next_state
            if next_action is not None:
                action = int(next_action)

        row = {
            "algorithm": "sarsa_lambda",
            "lambda": config.lambda_value,
            "episode": episode,
            "return": total_reward,
            "steps": env.steps,
            "success": int(terminated),
            "epsilon": epsilon,
            **counters,
        }
        history.append(row)
        if episode_callback is not None:
            episode_callback(episode, row)

    return SarsaLambdaResult(
        q_values=q,
        history=pd.DataFrame(history),
        runtime_seconds=float(time.perf_counter() - started),
        trace_log=pd.DataFrame(trace_records),
    )
