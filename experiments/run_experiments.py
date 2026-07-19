"""Reproduce every experiment and visual artifact used in the final report."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.common import EpsilonSchedule, evaluate_q_policy, save_q, write_jsonl
from agents.q_learning import QLearningConfig, train_q_learning
from agents.sarsa_lambda import SarsaLambdaConfig, train_sarsa_lambda
from agents.value_iteration import save_result, value_iteration
from environments.generator import generate_and_save_all, load_map
from environments.maze import DynamicMazeEnv
from experiments.analysis import (
    auc_early_success,
    episodes_to_threshold,
    plot_learning_curves,
    plot_policy_disagreement,
    plot_q_change_heatmap,
    plot_static_map,
    plot_trajectory,
    plot_visitation_heatmap,
    plot_policy_map,
    plot_value_heatmap,
    policy_agreement,
    policy_from_q,
    training_summary,
    write_json,
)
from transfer.transfer_learning import find_negative_transfer_example, initialize_transfer_q


RAW = ROOT / "results" / "raw_data"
MODELS = ROOT / "results" / "models"
FIGURES = ROOT / "results" / "figures"
MAPS = ROOT / "environments" / "maps"


def ensure_dirs() -> None:
    for path in (RAW, MODELS, FIGURES, MAPS):
        path.mkdir(parents=True, exist_ok=True)


def configured_env(map_name: str, reward_mode: str, gamma: float, seed: int) -> DynamicMazeEnv:
    return DynamicMazeEnv(MAPS / map_name, reward_mode=reward_mode, gamma=gamma, seed=seed)


def add_metadata(frame: pd.DataFrame, **metadata: object) -> pd.DataFrame:
    result = frame.copy()
    for key, value in metadata.items():
        result[key] = value
    return result


def run(profile: str) -> None:
    ensure_dirs()
    generate_and_save_all("40400356", MAPS)
    if profile == "quick":
        episodes, seeds, transfer_episodes, transfer_seeds = 350, [5, 15], 250, [5, 15]
        eval_episodes = 60
    else:
        episodes, seeds, transfer_episodes, transfer_seeds = 1200, [5, 15, 25, 35, 45], 800, [5, 25, 45]
        eval_episodes = 250

    gamma = 0.95
    alpha_q = 0.22
    alpha_sarsa = 0.04
    exp_schedule = EpsilonSchedule("exponential", 1.0, 0.05, 0.88)
    linear_schedule = EpsilonSchedule("linear", 1.0, 0.05, 0.88)
    experiment_started = time.perf_counter()

    run_manifest = {
        "profile": profile,
        "student_id": "40400356",
        "base_seed": 5,
        "maze_size": 16,
        "episodes": episodes,
        "seeds": seeds,
        "transfer_episodes": transfer_episodes,
        "transfer_seeds": transfer_seeds,
        "evaluation_episodes": eval_episodes,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "started_unix": time.time(),
    }
    write_json(run_manifest, RAW / "run_manifest.json")

    # 1) Exact model-based baseline and gamma sensitivity.
    vi_rows = []
    vi_main = None
    source_shaping = configured_env("source_map.json", "shaping", gamma, 5)
    plot_static_map(source_shaping, FIGURES / "source_map.png", "Source dynamic maze")
    plot_static_map(configured_env("target_similar_map.json", "shaping", gamma, 5), FIGURES / "target_similar_map.png", "Similar target maze")
    plot_static_map(configured_env("target_different_map.json", "shaping", gamma, 5), FIGURES / "target_different_map.png", "Different target maze")
    for tested_gamma in (0.85, 0.95, 0.99):
        env = configured_env("source_map.json", "shaping", tested_gamma, 5)
        result = value_iteration(env, gamma=tested_gamma, tolerance=1e-8)
        save_result(result, MODELS / f"value_iteration_gamma_{tested_gamma:.2f}.npz")
        vi_rows.append(
            {
                "gamma": tested_gamma,
                "iterations": result.iterations,
                "converged": result.converged,
                "final_delta": result.final_delta,
                "runtime_seconds": result.runtime_seconds,
                "start_value": float(result.values[env.start_pos[0], env.start_pos[1], 0, 0]),
                "memory_bytes_values_q_policy": int(result.values.nbytes + result.q_values.nbytes + result.policy.nbytes),
            }
        )
        if np.isclose(tested_gamma, gamma):
            vi_main = result
    assert vi_main is not None
    pd.DataFrame(vi_rows).to_csv(RAW / "value_iteration_gamma.csv", index=False)
    plot_value_heatmap(source_shaping, vi_main.values, FIGURES / "vi_value_before_key.png", has_key=0, phase=0, title="Value Iteration: before key, phase 0")
    plot_value_heatmap(source_shaping, vi_main.values, FIGURES / "vi_value_after_key.png", has_key=1, phase=0, title="Value Iteration: after key, phase 0")
    plot_policy_map(source_shaping, vi_main.policy, FIGURES / "vi_policy_before_key.png", has_key=0, phase=0, title="Value Iteration policy: before key")
    plot_policy_map(source_shaping, vi_main.policy, FIGURES / "vi_policy_after_key.png", has_key=1, phase=0, title="Value Iteration policy: after key")

    # 2) Q-Learning: two epsilon schedules on the same shaped reward.
    q_schedule_frames = []
    q_models: dict[tuple[str, int], np.ndarray] = {}
    q_visitations: dict[tuple[str, int], np.ndarray] = {}
    q_runtime_rows = []
    for schedule_name, schedule in (("linear", linear_schedule), ("exponential", exp_schedule)):
        for seed in seeds:
            env = configured_env("source_map.json", "shaping", gamma, seed)
            config = QLearningConfig(
                episodes=episodes,
                alpha=alpha_q,
                gamma=gamma,
                epsilon=schedule,
                seed=seed,
                collect_events=False,
                max_update_trace_rows=0,
            )
            result = train_q_learning(env, config)
            frame = add_metadata(result.history, seed=seed, epsilon_schedule=schedule_name, reward_mode="shaping")
            q_schedule_frames.append(frame)
            q_models[(schedule_name, seed)] = result.q_values
            q_visitations[(schedule_name, seed)] = result.visitation_counts
            q_runtime_rows.append({"algorithm": "q_learning", "variant": schedule_name, "seed": seed, "runtime_seconds": result.runtime_seconds, "memory_bytes": result.q_values.nbytes})
            save_q(result.q_values, MODELS / f"q_learning_{schedule_name}_seed_{seed}.npz", {"config": asdict(config)})
    q_schedule_data = pd.concat(q_schedule_frames, ignore_index=True)
    q_schedule_data.to_csv(RAW / "q_learning_epsilon_schedules.csv", index=False)
    plot_learning_curves(q_schedule_data, group_column="epsilon_schedule", value="success", output_path=FIGURES / "q_epsilon_success.png", ylabel="Rolling success rate", title="Q-Learning epsilon schedules")
    plot_learning_curves(q_schedule_data, group_column="epsilon_schedule", value="return", output_path=FIGURES / "q_epsilon_return.png", ylabel="Rolling episodic return", title="Q-Learning epsilon schedules")

    schedule_rank_rows = []
    threshold_window = min(100, max(20, episodes // 4))
    for (schedule_name, seed), group in q_schedule_data.groupby(["epsilon_schedule", "seed"]):
        row = training_summary(group, tail=min(200, episodes // 4))
        row.update(
            {
                "epsilon_schedule": schedule_name,
                "seed": int(seed),
                "episodes_to_80pct": episodes_to_threshold(group, threshold=0.80, window=threshold_window),
            }
        )
        schedule_rank_rows.append(row)
    schedule_scores = pd.DataFrame(schedule_rank_rows)
    schedule_scores.to_csv(RAW / "q_epsilon_schedule_summary.csv", index=False)
    schedule_aggregate = schedule_scores.groupby("epsilon_schedule").agg(
        final_success=("final_success_rate", "mean"),
        median_episodes_to_80pct=("episodes_to_80pct", "median"),
    )
    max_final_success = float(schedule_aggregate["final_success"].max())
    practically_tied = schedule_aggregate[schedule_aggregate["final_success"] >= max_final_success - 0.01]
    best_schedule = str(
        practically_tied.sort_values(
            ["median_episodes_to_80pct", "final_success"], ascending=[True, False]
        ).index[0]
    )
    best_schedule_obj = linear_schedule if best_schedule == "linear" else exp_schedule
    representative_seed = seeds[0]
    plot_visitation_heatmap(
        source_shaping,
        q_visitations[(best_schedule, representative_seed)],
        FIGURES / "q_training_visitation.png",
        "Q-Learning state visitation during training",
    )

    # 3) Sparse versus potential-based shaping under the selected schedule.
    reward_frames = []
    reward_models: dict[tuple[str, int], np.ndarray] = {}
    for reward_mode in ("sparse", "shaping"):
        for seed in seeds:
            if reward_mode == "shaping" and (best_schedule, seed) in q_models:
                q_values = q_models[(best_schedule, seed)]
                frame = q_schedule_data[(q_schedule_data["epsilon_schedule"] == best_schedule) & (q_schedule_data["seed"] == seed)].copy()
            else:
                env = configured_env("source_map.json", reward_mode, gamma, seed)
                config = QLearningConfig(episodes=episodes, alpha=alpha_q, gamma=gamma, epsilon=best_schedule_obj, seed=seed, collect_events=False, max_update_trace_rows=0)
                result = train_q_learning(env, config)
                q_values = result.q_values
                frame = add_metadata(result.history, seed=seed, epsilon_schedule=best_schedule, reward_mode=reward_mode)
                q_runtime_rows.append({"algorithm": "q_learning", "variant": reward_mode, "seed": seed, "runtime_seconds": result.runtime_seconds, "memory_bytes": result.q_values.nbytes})
            reward_models[(reward_mode, seed)] = q_values
            reward_frames.append(frame)
    reward_data = pd.concat(reward_frames, ignore_index=True)
    reward_data.to_csv(RAW / "reward_shaping_comparison.csv", index=False)
    plot_learning_curves(reward_data, group_column="reward_mode", value="success", output_path=FIGURES / "reward_success.png", ylabel="Rolling success rate", title="Sparse versus shaped reward")
    plot_learning_curves(reward_data, group_column="reward_mode", value="return", output_path=FIGURES / "reward_return.png", ylabel="Rolling episodic return", title="Sparse versus shaped reward")

    # Dedicated auditable Q update and event log.
    trace_env = configured_env("source_map.json", "shaping", gamma, 505)
    trace_config = QLearningConfig(episodes=min(40, episodes), alpha=alpha_q, gamma=gamma, epsilon=best_schedule_obj, seed=505, collect_events=True, log_all_events_for_first_n_episodes=4, max_update_trace_rows=50)
    trace_result = train_q_learning(trace_env, trace_config)
    trace_result.update_trace.to_csv(RAW / "q_learning_manual_update_trace.csv", index=False)
    write_jsonl(trace_result.event_records, RAW / "q_learning_event_log.jsonl")

    # Guaranteed examples of every mandatory event, extracted from the exact
    # transition model and a TimeLimit-style truncation demonstration.
    required_events = {
        "normal_move",
        "wall_collision",
        "penalty_cell_entered",
        "key_collected",
        "locked_door_attempt",
        "door_crossed",
        "goal_reached",
    }
    event_examples: list[dict[str, object]] = []
    found: set[str] = set()
    for state in trace_env.states():
        if trace_env.is_terminal(state):
            continue
        for action in range(4):
            for transition in trace_env.transition_outcomes(state, action):
                if transition.event in required_events and transition.event not in found:
                    event_examples.append({
                        "event": transition.event,
                        "state": state,
                        "chosen_action": action,
                        "realized_action": transition.realized_action,
                        "probability": transition.probability,
                        "next_state": transition.next_state,
                        "reward": transition.reward,
                        "terminated": transition.terminated,
                    })
                    found.add(transition.event)
        if found == required_events:
            break
    trunc_env = DynamicMazeEnv(MAPS / "source_map.json", reward_mode="shaping", gamma=gamma, seed=123, max_steps=1)
    trunc_state, _ = trunc_env.reset(seed=123)
    next_state, reward, terminated, truncated, info = trunc_env.step(0)
    event_examples.append({
        "event": "step_limit_reached",
        "state": trunc_state,
        "chosen_action": 0,
        "next_state": next_state,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    })
    write_jsonl(event_examples, RAW / "required_event_examples.jsonl")

    # 4) SARSA(lambda) ablation.
    sarsa_frames = []
    sarsa_models: dict[tuple[float, int], np.ndarray] = {}
    sarsa_runtime_rows = []
    for lambda_value in (0.0, 0.3, 0.7, 0.9):
        for seed in seeds:
            env = configured_env("source_map.json", "shaping", gamma, seed)
            config = SarsaLambdaConfig(episodes=episodes, alpha=alpha_sarsa, gamma=gamma, lambda_value=lambda_value, epsilon=best_schedule_obj, seed=seed, trace_type="replacing")
            result = train_sarsa_lambda(env, config, trace_episode=0 if (lambda_value == 0.7 and seed == seeds[0]) else -1)
            frame = add_metadata(result.history, seed=seed, lambda_label=f"lambda={lambda_value:.1f}")
            sarsa_frames.append(frame)
            sarsa_models[(lambda_value, seed)] = result.q_values
            sarsa_runtime_rows.append({"algorithm": "sarsa_lambda", "variant": lambda_value, "seed": seed, "runtime_seconds": result.runtime_seconds, "memory_bytes": result.q_values.nbytes})
            save_q(result.q_values, MODELS / f"sarsa_lambda_{lambda_value:.1f}_seed_{seed}.npz", {"config": asdict(config)})
            if lambda_value == 0.7 and seed == seeds[0]:
                result.trace_log.to_csv(RAW / "sarsa_lambda_trace_log.csv", index=False)
    sarsa_data = pd.concat(sarsa_frames, ignore_index=True)
    sarsa_data.to_csv(RAW / "sarsa_lambda_ablation.csv", index=False)
    plot_learning_curves(sarsa_data, group_column="lambda_label", value="success", output_path=FIGURES / "sarsa_lambda_success.png", ylabel="Rolling success rate", title="SARSA(lambda) ablation")
    plot_learning_curves(sarsa_data, group_column="lambda_label", value="return", output_path=FIGURES / "sarsa_lambda_return.png", ylabel="Rolling episodic return", title="SARSA(lambda) ablation")

    lambda_summary_rows = []
    for (lambda_label, seed), group in sarsa_data.groupby(["lambda_label", "seed"]):
        summary = training_summary(group, tail=min(200, episodes // 4))
        summary.update({"lambda_label": lambda_label, "seed": seed, "episodes_to_80pct": episodes_to_threshold(group, 0.8, min(100, episodes // 4))})
        lambda_summary_rows.append(summary)
    lambda_summary = pd.DataFrame(lambda_summary_rows)
    lambda_summary.to_csv(RAW / "sarsa_lambda_summary.csv", index=False)
    # Rank by final success first and finite speed second.
    rank = lambda_summary.groupby("lambda_label").agg(final_success=("final_success_rate", "mean"), speed=("episodes_to_80pct", "median"))
    best_lambda_label = str(rank.sort_values(["final_success", "speed"], ascending=[False, True]).index[0])
    best_lambda = float(best_lambda_label.split("=")[1])

    # 5) Common-policy comparison against Value Iteration.
    q_best = q_models[(best_schedule, representative_seed)]
    sarsa_best = sarsa_models[(best_lambda, representative_seed)]
    trajectory_env = configured_env("source_map.json", "shaping", gamma, 7000)
    trajectory_positions: list[tuple[int, int]] = []
    trajectory_success = False
    for rollout_seed in range(7000, 7050):
        state, _ = trajectory_env.reset(seed=rollout_seed, phase=rollout_seed % trajectory_env.gate_period)
        positions = [(state[0], state[1])]
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(np.argmax(q_best[state]))
            state, _reward, terminated, truncated, _info = trajectory_env.step(action)
            positions.append((state[0], state[1]))
        if terminated:
            trajectory_positions = positions
            trajectory_success = True
            break
    if not trajectory_positions:
        trajectory_positions = positions
    plot_trajectory(
        trajectory_env,
        trajectory_positions,
        FIGURES / "q_final_trajectory.png",
        f"Q-Learning final trajectory (success={trajectory_success})",
    )
    q_policy = policy_from_q(q_best)
    sarsa_policy = policy_from_q(sarsa_best)
    q_agreement = policy_agreement(source_shaping, q_policy, vi_main.policy)
    sarsa_agreement = policy_agreement(source_shaping, sarsa_policy, vi_main.policy)
    q_agreement["comparisons"].to_csv(RAW / "q_policy_comparison.csv", index=False)
    sarsa_agreement["comparisons"].to_csv(RAW / "sarsa_policy_comparison.csv", index=False)
    plot_policy_disagreement(source_shaping, q_agreement["comparisons"], FIGURES / "q_policy_disagreement.png", "Q-Learning disagreement with Value Iteration")
    plot_policy_disagreement(source_shaping, sarsa_agreement["comparisons"], FIGURES / "sarsa_policy_disagreement.png", "SARSA(lambda) disagreement with Value Iteration")
    plot_policy_map(source_shaping, q_policy, FIGURES / "q_policy_before_key.png", has_key=0, phase=0, title="Q-Learning policy: before key")
    plot_policy_map(source_shaping, sarsa_policy, FIGURES / "sarsa_policy_before_key.png", has_key=0, phase=0, title="SARSA(lambda) policy: before key")

    main_eval_rows = []
    vi_eval = evaluate_q_policy(configured_env("source_map.json", "shaping", gamma, 900), vi_main.q_values, episodes=eval_episodes, seed=900)
    main_eval_rows.append({"algorithm": "Value Iteration", "variant": "gamma=0.95", **vi_eval, "runtime_seconds": float(pd.DataFrame(vi_rows).query("gamma == 0.95")["runtime_seconds"].iloc[0]), "memory_bytes": int(vi_main.values.nbytes + vi_main.q_values.nbytes + vi_main.policy.nbytes), "policy_agreement": 1.0})
    for seed in seeds:
        q_eval = evaluate_q_policy(configured_env("source_map.json", "shaping", gamma, 1000 + seed), q_models[(best_schedule, seed)], episodes=eval_episodes, seed=1000 + seed)
        main_eval_rows.append({"algorithm": "Q-Learning", "variant": best_schedule, "seed": seed, **q_eval, "runtime_seconds": next(row["runtime_seconds"] for row in q_runtime_rows if row["variant"] == best_schedule and row["seed"] == seed), "memory_bytes": q_models[(best_schedule, seed)].nbytes, "policy_agreement": policy_agreement(source_shaping, policy_from_q(q_models[(best_schedule, seed)]), vi_main.policy)["agreement_rate"]})
        s_eval = evaluate_q_policy(configured_env("source_map.json", "shaping", gamma, 2000 + seed), sarsa_models[(best_lambda, seed)], episodes=eval_episodes, seed=2000 + seed)
        main_eval_rows.append({"algorithm": "SARSA(lambda)", "variant": best_lambda, "seed": seed, **s_eval, "runtime_seconds": next(row["runtime_seconds"] for row in sarsa_runtime_rows if np.isclose(row["variant"], best_lambda) and row["seed"] == seed), "memory_bytes": sarsa_models[(best_lambda, seed)].nbytes, "policy_agreement": policy_agreement(source_shaping, policy_from_q(sarsa_models[(best_lambda, seed)]), vi_main.policy)["agreement_rate"]})
    pd.DataFrame(main_eval_rows).to_csv(RAW / "three_algorithm_comparison.csv", index=False)

    # Three concrete disagreement states with local data.
    disagreement_rows = q_agreement["comparisons"].query("agree == 0").copy()
    examples = []
    for row in disagreement_rows.head(3).itertuples(index=False):
        state = row.state
        if isinstance(state, str):
            state = tuple(int(x.strip()) for x in state.strip("()").split(","))
        examples.append({"state": state, "candidate_action": row.candidate_action, "reference_action": row.reference_action, "tile": str(source_shaping.grid[state[0], state[1]]), "local_signature": source_shaping.local_signature(state[0], state[1]), "q_values": q_best[state].tolist(), "vi_q_values": vi_main.q_values[state].tolist()})
    write_json({"examples": examples}, RAW / "policy_disagreement_examples.json")

    # 6) Transfer learning on similar and different targets.
    source_q_for_transfer = q_best
    transfer_frames = []
    transfer_summary_rows = []
    negative_example = None
    transfer_q_change_pair: tuple[np.ndarray, np.ndarray, DynamicMazeEnv] | None = None
    for target_kind, map_file in (("similar", "target_similar_map.json"), ("different", "target_different_map.json")):
        target_env_model = configured_env(map_file, "shaping", gamma, 3000)
        target_vi = value_iteration(target_env_model, gamma=gamma, tolerance=1e-8)
        save_result(target_vi, MODELS / f"value_iteration_target_{target_kind}.npz")
        scenarios = [("scratch", None), ("full", None), ("scaled", 0.25), ("scaled", 0.50), ("scaled", 0.75), ("selective", None)]
        for mode, beta in scenarios:
            label = mode if beta is None else f"scaled_beta_{beta:.2f}"
            initial_q, transfer_meta = initialize_transfer_q(source_shaping, target_env_model, source_q_for_transfer, mode=mode, beta=0.5 if beta is None else beta)
            initial_eval = evaluate_q_policy(configured_env(map_file, "shaping", gamma, 4000), initial_q, episodes=max(80, eval_episodes // 2), seed=4000)
            if target_kind == "different" and mode == "full" and negative_example is None:
                negative_example = find_negative_transfer_example(target_env_model, initial_q, target_vi.q_values)
            for seed in transfer_seeds:
                env = configured_env(map_file, "shaping", gamma, seed)
                watch_state = tuple(negative_example["state"]) if negative_example and target_kind == "different" and mode == "full" else None
                config = QLearningConfig(episodes=transfer_episodes, alpha=alpha_q, gamma=gamma, epsilon=best_schedule_obj, seed=seed, collect_events=False, max_update_trace_rows=0)
                result = train_q_learning(env, config, initial_q=initial_q, watch_state=watch_state, watch_interval=max(20, transfer_episodes // 20))
                frame = add_metadata(result.history, seed=seed, target=target_kind, transfer_scenario=label)
                transfer_frames.append(frame)
                final_eval = evaluate_q_policy(configured_env(map_file, "shaping", gamma, 5000 + seed), result.q_values, episodes=eval_episodes, seed=5000 + seed)
                transfer_summary_rows.append({"target": target_kind, "scenario": label, "seed": seed, "initial_success_rate": initial_eval["success_rate"], "initial_mean_return": initial_eval["mean_return"], "early_success_auc": auc_early_success(result.history), "episodes_to_80pct": episodes_to_threshold(result.history, 0.8, min(100, transfer_episodes // 4)), "final_success_rate": final_eval["success_rate"], "final_mean_return": final_eval["mean_return"], "final_mean_steps": final_eval["mean_steps"], "runtime_seconds": result.runtime_seconds, "copied_fraction": transfer_meta.copied_fraction})
                save_q(result.q_values, MODELS / f"transfer_{target_kind}_{label}_seed_{seed}.npz", {"target": target_kind, "scenario": label, "transfer": asdict(transfer_meta)})
                if target_kind == "different" and mode == "full" and seed == transfer_seeds[0]:
                    transfer_q_change_pair = (initial_q.copy(), result.q_values.copy(), target_env_model)
                if watch_state is not None and seed == transfer_seeds[0]:
                    result.watch_trace.to_csv(RAW / "negative_transfer_q_correction.csv", index=False)
        
    transfer_data = pd.concat(transfer_frames, ignore_index=True)
    transfer_data.to_csv(RAW / "transfer_episode_data.csv", index=False)
    pd.DataFrame(transfer_summary_rows).to_csv(RAW / "transfer_summary.csv", index=False)
    plot_learning_curves(transfer_data.query("target == 'similar'"), group_column="transfer_scenario", value="success", output_path=FIGURES / "transfer_similar_success.png", ylabel="Rolling success rate", title="Transfer to similar target", window=min(80, max(20, transfer_episodes // 8)))
    plot_learning_curves(transfer_data.query("target == 'different'"), group_column="transfer_scenario", value="success", output_path=FIGURES / "transfer_different_success.png", ylabel="Rolling success rate", title="Transfer to different target", window=min(80, max(20, transfer_episodes // 8)))
    if negative_example is not None:
        write_json(negative_example, RAW / "negative_transfer_example.json")

    # Visualize how transferred Q values were corrected by target training.
    if transfer_q_change_pair is not None:
        q_before, q_after, q_env = transfer_q_change_pair
        plot_q_change_heatmap(
            q_env,
            q_before,
            q_after,
            FIGURES / "transfer_q_change_different.png",
            "Q-value correction after full transfer to different target",
        )
        np.save(RAW / "full_transfer_q_change.npy", np.max(np.abs(q_after - q_before), axis=(2, 3, 4)))

    final_summary = {
        "best_epsilon_schedule": best_schedule,
        "best_lambda": best_lambda,
        "q_policy_agreement": q_agreement["agreement_rate"],
        "sarsa_policy_agreement": sarsa_agreement["agreement_rate"],
        "elapsed_seconds": time.perf_counter() - experiment_started,
        "profile": profile,
    }
    write_json(final_summary, RAW / "final_summary.json")
    print(json.dumps(final_summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "submission"), default="submission")
    args = parser.parse_args()
    run(args.profile)


if __name__ == "__main__":
    main()
