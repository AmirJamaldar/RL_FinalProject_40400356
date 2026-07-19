"""Primary entry point for the RL final project."""
from __future__ import annotations

import argparse
from pathlib import Path

from agents.common import greedy_action
from agents.value_iteration import value_iteration
from environments.generator import generate_and_save_all
from environments.maze import DynamicMazeEnv

ROOT = Path(__file__).resolve().parent


def ascii_demo(episodes: int = 1) -> None:
    env = DynamicMazeEnv(ROOT / "environments" / "maps" / "source_map.json", reward_mode="shaping", gamma=0.95, seed=5)
    result = value_iteration(env, gamma=0.95, tolerance=1e-8)
    for episode in range(episodes):
        state, _ = env.reset(seed=5 + episode, phase=episode % env.gate_period)
        terminated = truncated = False
        print(f"\nEpisode {episode + 1}")
        while not (terminated or truncated):
            action = int(result.policy[state])
            state, reward, terminated, truncated, info = env.step(action)
            print(f"step={env.steps:03d} action={action} event={info['event']:<24} reward={reward:7.3f}")
        print(f"success={terminated} steps={env.steps} return={env.episode_return:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic Maze RL Final Project")
    parser.add_argument("--gui", action="store_true", help="launch the Tkinter GUI")
    parser.add_argument("--demo", action="store_true", help="run a terminal demonstration")
    parser.add_argument("--generate-maps", action="store_true", help="regenerate deterministic maps")
    parser.add_argument("--episodes", type=int, default=1, help="number of demo episodes")
    args = parser.parse_args()

    if args.generate_maps:
        paths = generate_and_save_all("40400356", ROOT / "environments" / "maps")
        for name, path in paths.items():
            print(f"{name}: {path}")
        return
    if args.demo:
        ascii_demo(args.episodes)
        return
    from gui.app import launch
    launch()


if __name__ == "__main__":
    main()
