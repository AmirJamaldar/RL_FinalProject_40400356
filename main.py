"""Main GUI entry point with a deterministic headless fallback."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from agents.common import evaluate_q_policy
from agents.q_learning import EpsilonSchedule, QLearningConfig, train_q_learning
from agents.sarsa_lambda import SarsaLambdaConfig, train_sarsa_lambda
from agents.value_iteration import bellman_residual, value_iteration
from environments.generator import load_map
from environments.maze import DynamicMazeEnv, RewardSpec
from gui.app import ensure_maps, launch_gui


def headless_demo(algorithm: str, episodes: int, map_name: str) -> dict:
    paths = ensure_maps()
    maze = load_map(paths[map_name])
    env_kwargs = dict(
        maze=maze,
        reward_mode="shaped",
        reward_spec=RewardSpec(),
        gamma=0.95,
        max_steps=3 * maze.walkable_count,
        seed=5,
    )
    if algorithm == "vi":
        env = DynamicMazeEnv(**env_kwargs)
        result = value_iteration(env, threshold=1e-8)
        evaluation = evaluate_q_policy(DynamicMazeEnv(**env_kwargs), result.q_values, episodes=max(5, episodes), seed=5005)
        return {
            "algorithm": "value_iteration",
            "converged": result.converged,
            "iterations": result.iterations,
            "bellman_residual": bellman_residual(env, result),
            "evaluation": evaluation.metrics,
        }
    schedule = EpsilonSchedule("linear", 1.0, 0.05, max(1, int(episodes * 0.8)))
    if algorithm == "q":
        result = train_q_learning(
            DynamicMazeEnv(**env_kwargs),
            QLearningConfig(episodes=episodes, alpha=0.15, gamma=0.95, epsilon=schedule),
            seed=5,
        )
    else:
        result = train_sarsa_lambda(
            DynamicMazeEnv(**env_kwargs),
            SarsaLambdaConfig(episodes=episodes, alpha=0.12, gamma=0.95, lambda_value=0.7, epsilon=schedule),
            seed=5,
        )
    evaluation = evaluate_q_policy(DynamicMazeEnv(**env_kwargs), result.q_table, episodes=max(5, min(50, episodes)), seed=6005)
    return {
        "algorithm": result.algorithm,
        "training_episodes": episodes,
        "samples": result.total_samples,
        "runtime_seconds": result.runtime_seconds,
        "final_training_success": result.episode_metrics[-1]["success"],
        "evaluation": evaluation.metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Run a terminal smoke demonstration")
    parser.add_argument("--gui", action="store_true", help="Force GUI launch even if DISPLAY is absent")
    parser.add_argument("--algorithm", choices=("vi", "q", "sarsa"), default="q")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--map", dest="map_name", choices=("source", "similar", "different"), default="source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    display_missing = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")
    if args.headless or (display_missing and not args.gui):
        if display_missing and not args.headless:
            print("No graphical display detected; running the headless demo. Use --gui on a desktop.")
        print(json.dumps(headless_demo(args.algorithm, args.episodes, args.map_name), indent=2, sort_keys=True))
    else:
        launch_gui()


if __name__ == "__main__":
    main()
