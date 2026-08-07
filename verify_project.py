"""Strict integrity verification for persisted experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from agents.common import evaluate_q_policy
from environments.generator import load_map, maze_fingerprint
from environments.maze import DynamicMazeEnv, REQUIRED_EVENTS, RewardSpec
from experiments.run_experiments import _sha256, code_hash, validate_model_archives
from gui.app import GUI_CHECKPOINTS, load_gui_checkpoint


ROOT = Path(__file__).resolve().parent


def verify_project(root: str | Path = ROOT) -> dict:
    root = Path(root)
    raw = root / "results/raw_data"
    models = root / "results/models"
    figures = root / "results/figures"
    manifest = json.loads((raw / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise AssertionError("Experiment manifest is not completed")
    if manifest.get("code_sha256") != code_hash(root):
        raise AssertionError("Manifest code hash does not match the delivered source")

    config_path = Path(manifest["config_path"])
    if not config_path.is_absolute():
        config_path = root / config_path
    if manifest.get("config_sha256") != _sha256(config_path):
        raise AssertionError("Executed configuration hash mismatch")
    config = json.loads((raw / "executed_config.json").read_text(encoding="utf-8"))
    if config != json.loads(config_path.read_text(encoding="utf-8")):
        raise AssertionError("executed_config.json differs from the hashed configuration")

    for name, expected_hash in manifest["map_sha256"].items():
        path = root / f"environments/maps/{name}_map.json"
        if _sha256(path) != expected_hash:
            raise AssertionError(f"Map hash mismatch: {name}")
        if maze_fingerprint(load_map(path)) != manifest["map_fingerprints"][name]:
            raise AssertionError(f"Map semantic fingerprint mismatch: {name}")

    missing_raw = [name for name in manifest["raw_files"] if not (raw / name).is_file()]
    if missing_raw:
        raise AssertionError(f"Manifest references missing raw files: {missing_raw}")
    for filename, expected_hash in manifest["raw_sha256"].items():
        if _sha256(raw / filename) != expected_hash:
            raise AssertionError(f"Raw artifact checksum mismatch: {filename}")
    for filename, expected_hash in manifest["model_sha256"].items():
        if _sha256(models / filename) != expected_hash:
            raise AssertionError(f"Model artifact checksum mismatch: {filename}")
    for filename, expected_hash in manifest["figure_sha256"].items():
        if _sha256(figures / filename) != expected_hash:
            raise AssertionError(f"Figure artifact checksum mismatch: {filename}")
    model_validation = validate_model_archives(models)
    if model_validation["archive_count"] != int(manifest["model_count"]):
        raise AssertionError("Model archive count does not match the manifest")

    training = pd.read_csv(raw / "training_episodes.csv")
    evaluation = pd.read_csv(raw / "evaluation_episodes.csv")
    transfer = pd.read_csv(raw / "transfer_episodes.csv")
    run_summary = pd.read_csv(raw / "run_summary.csv")
    transfer_summary = pd.read_csv(raw / "transfer_summary.csv")
    observed_counts = {
        "training_runs": int(run_summary.shape[0]),
        "transfer_runs": int(transfer_summary.shape[0]),
        "training_episode_rows": int(training.shape[0]),
        "transfer_episode_rows": int(transfer.shape[0]),
    }
    if observed_counts != manifest["run_counts"]:
        raise AssertionError(f"CSV counts differ from manifest: {observed_counts}")
    expected_without_models = {
        key: int(value)
        for key, value in manifest["expected_counts"].items()
        if key != "model_archives"
    }
    if observed_counts != expected_without_models:
        raise AssertionError("CSV counts do not cover the complete configured matrix")
    if training.duplicated(["run_id", "episode"]).any():
        raise AssertionError("Duplicate source-training episode rows exist")
    if transfer.duplicated(["run_id", "episode"]).any():
        raise AssertionError("Duplicate transfer-training episode rows exist")
    source_episode_counts = training.groupby("run_id")["episode"].agg(["size", "min", "max"])
    if not (
        (source_episode_counts["size"] == source_episode_counts["max"] + 1)
        & (source_episode_counts["min"] == 0)
    ).all():
        raise AssertionError("At least one source-training run has a non-contiguous episode range")
    transfer_episode_counts = transfer.groupby("run_id")["episode"].agg(["size", "min", "max"])
    if not (
        (transfer_episode_counts["size"] == transfer_episode_counts["max"] + 1)
        & (transfer_episode_counts["min"] == 0)
    ).all():
        raise AssertionError("At least one transfer-training run has a non-contiguous episode range")
    required_numeric_columns = ["environment_seed", "return", "steps", "success", "terminated", "truncated"]
    if training[required_numeric_columns].isna().any().any() or transfer[required_numeric_columns].isna().any().any():
        raise AssertionError("A grading-critical episode field contains NaN")
    if int(evaluation.groupby("episode")["environment_seed"].nunique().max()) != 1:
        raise AssertionError("Source policies were not evaluated with common environment seeds")
    if int(transfer_summary["evaluation_base_seed"].nunique()) != 1:
        raise AssertionError("Transfer scenarios were not evaluated with a common base seed")
    if int(transfer.groupby(["seed", "episode"])["environment_seed"].nunique().max()) != 1:
        raise AssertionError("Transfer scenarios were not trained with matched environment-noise streams")

    q_trace = pd.read_csv(raw / "q_update_trace.csv")
    expected_targets = q_trace["reward"] + float(config["gamma"]) * q_trace["next_max_q"] * (1 - q_trace["terminated"])
    expected_new_q = q_trace["old_q"] + float(config["q_alpha"]) * (expected_targets - q_trace["old_q"])
    if not np.allclose(q_trace["target"], expected_targets, atol=1e-12, rtol=0):
        raise AssertionError("Persisted Q-Learning trace violates its Bellman target equation")
    if not np.allclose(q_trace["new_q"], expected_new_q, atol=1e-12, rtol=0):
        raise AssertionError("Persisted Q-Learning trace violates its update equation")

    sarsa_trace = pd.read_csv(raw / "sarsa_trace.csv")
    expected_delta = sarsa_trace["reward"] + float(config["gamma"]) * sarsa_trace["bootstrap_q"] - sarsa_trace["old_q"]
    if not np.allclose(sarsa_trace["td_error_delta"], expected_delta, atol=1e-12, rtol=0):
        raise AssertionError("Persisted SARSA trace violates its TD-error equation")
    decay = float(config["gamma"]) * sarsa_trace["lambda"]
    if not np.allclose(
        sarsa_trace["trace_l1_after_decay"],
        sarsa_trace["trace_l1_before_decay"] * decay,
        atol=1e-12,
        rtol=0,
    ):
        raise AssertionError("Persisted SARSA eligibility trace violates gamma-lambda decay")

    event_rows = [
        json.loads(line)
        for line in (raw / "sampled_event_log.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    observed_events = {event for row in event_rows for event in row.get("events", [])}
    if not REQUIRED_EVENTS.issubset(observed_events):
        raise AssertionError("Sampled event log does not cover every required event")
    reward_check = json.loads((raw / "reward_design_verification.json").read_text(encoding="utf-8"))
    if not np.isclose(float(reward_check["sparse_vs_shaped_policy_agreement"]), 1.0):
        raise AssertionError("Sparse and shaped optimal policies do not agree")
    witnesses = json.loads((raw / "negative_transfer_witnesses.json").read_text(encoding="utf-8"))
    if not witnesses or not any(
        row.get("behavior_corrected") and float(row.get("exact_one_state_regret", 0.0)) > 0
        for row in witnesses
    ):
        raise AssertionError("No corrected negative-transfer witness is persisted")

    figure_manifest = json.loads((raw / "figure_manifest.json").read_text(encoding="utf-8"))
    if len(figure_manifest) != int(manifest["figure_count"]):
        raise AssertionError("Figure count does not match the manifest")
    for filename in figure_manifest:
        with Image.open(figures / filename) as image:
            image.verify()
    temporary_artifacts = [
        str(path.relative_to(root))
        for path in (root / "results").rglob("*")
        if path.is_file() and ".tmp" in path.name
    ]
    if temporary_artifacts:
        raise AssertionError(f"Temporary result artifacts remain: {temporary_artifacts}")

    gui_models_checked = 0
    reward = RewardSpec.from_dict(config["reward"])
    for algorithm, environment in GUI_CHECKPOINTS:
        maze = load_map(root / f"environments/maps/{environment}_map.json")
        env = DynamicMazeEnv(
            maze,
            reward_mode="shaped",
            reward_spec=reward,
            gamma=float(config["gamma"]),
            max_steps=int(config["max_steps_multiplier"] * maze.walkable_count),
            seed=5,
        )
        loaded = load_gui_checkpoint(algorithm, environment, env, model_dir=models)
        before = loaded.q_table.copy()
        evaluation = evaluate_q_policy(env, loaded.q_table, episodes=3, seed=90_005)
        if evaluation.metrics["success_rate"] <= 0:
            raise AssertionError(f"Packaged GUI model never succeeds: {algorithm}/{environment}")
        np.testing.assert_array_equal(before, loaded.q_table)
        gui_models_checked += 1

    return {
        "status": "verified",
        "profile": manifest["profile"],
        "code_sha256": manifest["code_sha256"],
        "model_archives": model_validation["archive_count"],
        "numeric_model_arrays": model_validation["numeric_array_count"],
        "figures": len(figure_manifest),
        "gui_models_checked": gui_models_checked,
        **observed_counts,
    }


def main() -> None:
    print(json.dumps(verify_project(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
