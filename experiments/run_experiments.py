"""Run every required experiment and persist raw, reproducible evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HASHED_SOURCE_DIRECTORIES = (
    "agents",
    "environments",
    "experiments",
    "gui",
    "tests",
    "transfer",
)
HASHED_SOURCE_FILES = ("main.py", "verify_project.py")

import numpy as np
import pandas as pd

from agents.common import evaluate_q_policy, q_at, save_q_checkpoint, state_index
from agents.q_learning import EpsilonSchedule, QLearningConfig, train_q_learning
from agents.sarsa_lambda import SarsaLambdaConfig, train_sarsa_lambda
from agents.value_iteration import bellman_residual, save_value_iteration, value_iteration
from environments.generator import (
    WALL,
    generate_source_map,
    generate_transfer_map,
    local_signature,
    maze_fingerprint,
    validate_map,
)
from environments.maze import (
    ACTION_NAMES,
    PERPENDICULAR_ACTIONS,
    REQUIRED_EVENTS,
    DynamicMazeEnv,
    EventLogger,
    RewardSpec,
)
from transfer.transfer_learning import (
    find_negative_transfer_witness,
    initialize_transfer,
    policy_agreement,
)


def progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path, *, attempts: int = 3) -> None:
    """Copy a completed local artifact and prove the persisted bytes are identical.

    Some synchronized filesystems can acknowledge a large write yet expose a
    truncated destination.  The experiment manifest must never claim success in
    that situation, so every generated table is staged outside the result tree,
    flushed, copied, and compared by both byte count and SHA-256.
    """

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = source.stat().st_size
    expected_hash = _sha256(source)
    last_error: Exception | None = None
    for _attempt in range(attempts):
        try:
            with source.open("rb") as input_stream, destination.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            if destination.stat().st_size != expected_size or _sha256(destination) != expected_hash:
                raise OSError(f"persisted byte verification failed for {destination}")
            return
        except Exception as error:
            last_error = error
    raise OSError(f"failed to persist a verified artifact after {attempts} attempts: {destination}") from last_error


def write_bytes_verified(path: Path, payload: bytes) -> None:
    """Persist bytes via an OS-temporary staging file and verify the result."""

    descriptor, temporary_name = tempfile.mkstemp(prefix="rl-final-artifact-")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _copy_verified(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    write_bytes_verified(path, payload)


def write_dataframe_verified(path: Path, frame: pd.DataFrame) -> None:
    """Write a CSV from a completed temporary file, then verify exact bytes."""

    descriptor, temporary_name = tempfile.mkstemp(prefix="rl-final-table-", suffix=".csv")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        # Windows maps os.fsync() to _commit(), which rejects read-only
        # descriptors even though POSIX platforms commonly accept them.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        _copy_verified(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def csv_row_count(path: Path) -> int:
    """Parse a CSV in bounded memory and return its persisted data-row count."""

    return sum(len(chunk) for chunk in pd.read_csv(path, chunksize=50_000))


def code_hash(root: str | Path = ROOT) -> str:
    """Hash the delivered source deterministically across platforms.

    Only project-owned source/configuration locations are included.  This keeps
    local virtual environments, bytecode caches, VCS metadata, IDE settings,
    and generated results from contaminating the source identity.  POSIX path
    names and length framing make the byte stream identical on Windows and
    POSIX systems and unambiguous between adjacent path/content records.
    """

    root = Path(root).resolve()
    digest = hashlib.sha256()
    candidates = [root / filename for filename in HASHED_SOURCE_FILES]
    for directory_name in HASHED_SOURCE_DIRECTORIES:
        directory = root / directory_name
        for pattern in ("*.py", "*.json"):
            candidates.extend(directory.rglob(pattern) if directory.is_dir() else ())
    records = sorted(
        (path.relative_to(root).as_posix(), path)
        for path in candidates
        if path.is_file()
    )
    for relative_name, path in records:
        encoded_name = relative_name.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(encoded_name).to_bytes(8, byteorder="big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def remove_generated_files(directory: Path, suffixes: set[str]) -> list[str]:
    """Remove only known generated files while preserving unrelated user files."""

    removed: list[str] = []
    if not directory.exists():
        return removed
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            path.unlink()
            removed.append(path.name)
    return removed


def validate_config(config: dict) -> None:
    """Fail before a costly run when a grading-critical matrix is incomplete."""

    student_id = str(config.get("student_id", ""))
    if len(student_id) < 2 or not student_id.isdigit():
        raise ValueError("student_id must contain at least two decimal digits")
    expected_seed = int(student_id[-2])
    expected_size = 15 + expected_seed % 4
    if student_id != "40400356" or expected_seed != 5 or expected_size != 16:
        raise ValueError("Configuration does not match student ID 40400356 -> seed 5 -> size 16")
    seeds = [int(value) for value in config.get("seeds", [])]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds) or seeds[0] != expected_seed:
        raise ValueError("seeds must be unique, include at least three runs, and start with base seed 5")
    gammas = {float(value) for value in config.get("vi_gammas", [])}
    if len(gammas) < 3:
        raise ValueError("At least three distinct Value Iteration gamma values are required")
    if not {0.0, 0.3, 0.7, 0.9}.issubset(
        {round(float(value), 10) for value in config.get("sarsa_lambdas", [])}
    ):
        raise ValueError("SARSA lambda sweep must include 0, 0.3, 0.7, and 0.9")
    if not {0.25, 0.5, 0.75}.issubset(
        {round(float(value), 10) for value in config.get("transfer_betas", [])}
    ):
        raise ValueError("Transfer beta sweep must include 0.25, 0.50, and 0.75")
    for key in ("q_episodes", "sarsa_episodes", "transfer_episodes", "evaluation_episodes"):
        if int(config.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be positive")
    gamma = float(config.get("gamma", -1))
    if not 0 <= gamma < 1:
        raise ValueError("gamma must lie in [0, 1)")
    if not 0 < float(config.get("q_alpha", 0)) <= 1:
        raise ValueError("q_alpha must lie in (0, 1]")
    if not 0 < float(config.get("sarsa_alpha", 0)) <= 1:
        raise ValueError("sarsa_alpha must lie in (0, 1]")
    for schedule_name in ("epsilon", "transfer_epsilon"):
        values = config.get(schedule_name, {})
        for mode in ("linear", "exponential"):
            EpsilonSchedule(
                mode=mode,
                start=float(values.get("start", -1)),
                end=float(values.get("end", -1)),
                decay_episodes=int(values.get("decay_episodes", 0)),
            )
    if int(config.get("success_window", 0)) <= 0:
        raise ValueError("success_window must be positive")
    if not 0 <= float(config.get("success_threshold", -1)) <= 1:
        raise ValueError("success_threshold must lie in [0, 1]")
    if float(config.get("vi_threshold", 0)) <= 0 or int(config.get("vi_max_iterations", 0)) <= 0:
        raise ValueError("Value Iteration threshold and iteration cap must be positive")
    if float(config.get("max_steps_multiplier", 0)) <= 0:
        raise ValueError("max_steps_multiplier must be positive")
    RewardSpec.from_dict(config.get("reward"))


def load_config(path_value: str | None) -> tuple[dict, Path]:
    path = ROOT / "experiments/configs/full.json" if path_value is None else Path(path_value)
    if not path.is_absolute():
        candidate = Path.cwd() / path
        path = candidate if candidate.exists() else ROOT / path
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config, path.resolve()


def make_env(maze, config: dict, *, reward_mode: str, gamma: float | None = None, max_steps: int | None = None):
    discount = float(config["gamma"] if gamma is None else gamma)
    return DynamicMazeEnv(
        maze,
        reward_mode=reward_mode,
        reward_spec=RewardSpec.from_dict(config["reward"]),
        gamma=discount,
        max_steps=(
            int(config["max_steps_multiplier"] * maze.walkable_count)
            if max_steps is None
            else max_steps
        ),
        seed=int(config["seeds"][0]),
    )


def epsilon_schedule(config: dict, mode: str, *, transfer: bool = False) -> EpsilonSchedule:
    values = config["transfer_epsilon" if transfer else "epsilon"]
    return EpsilonSchedule(
        mode=mode,
        start=float(values["start"]),
        end=float(values["end"]),
        decay_episodes=int(values["decay_episodes"]),
    )


def learning_speed(metrics: list[dict], window: int, threshold: float) -> tuple[int | None, int | None]:
    successes = pd.Series([row["success"] for row in metrics], dtype=float)
    rolling = successes.rolling(window=window, min_periods=window).mean()
    matches = np.flatnonzero(rolling.to_numpy() >= threshold)
    if len(matches) == 0:
        return None, None
    index = int(matches[0])
    samples = int(sum(row["steps"] for row in metrics[: index + 1]))
    return index + 1, samples


def append_training_rows(destination: list[dict], result, *, run_id: str, algorithm: str, variant: str, seed: int, reward_mode: str, map_kind: str) -> None:
    for row in result.episode_metrics:
        destination.append(
            {
                "run_id": run_id,
                "algorithm": algorithm,
                "variant": variant,
                "seed": seed,
                "reward_mode": reward_mode,
                "map_kind": map_kind,
                **row,
            }
        )


def append_evaluation_rows(destination: list[dict], evaluation, *, run_id: str, phase: str, algorithm: str, variant: str, seed: int, reward_mode: str, map_kind: str) -> None:
    for row in evaluation.episodes:
        destination.append(
            {
                "run_id": run_id,
                "phase": phase,
                "algorithm": algorithm,
                "variant": variant,
                "seed": seed,
                "reward_mode": reward_mode,
                "map_kind": map_kind,
                **row,
            }
        )


def model_free_summary(
    result,
    evaluation,
    agreement: dict,
    config: dict,
    *,
    run_id: str,
    algorithm: str,
    variant: str,
    seed: int,
    reward_mode: str,
    map_kind: str,
) -> dict:
    episode_to_threshold, samples_to_threshold = learning_speed(
        result.episode_metrics,
        int(config["success_window"]),
        float(config["success_threshold"]),
    )
    final_window = result.episode_metrics[-int(config["success_window"]) :]
    return {
        "run_id": run_id,
        "algorithm": algorithm,
        "variant": variant,
        "seed": seed,
        "reward_mode": reward_mode,
        "map_kind": map_kind,
        "runtime_seconds": result.runtime_seconds,
        "total_samples": result.total_samples,
        "memory_bytes": result.memory_bytes,
        "episodes_to_success_threshold": episode_to_threshold,
        "samples_to_success_threshold": samples_to_threshold,
        "final_window_success_rate": float(np.mean([row["success"] for row in final_window])),
        "final_window_return_mean": float(np.mean([row["return"] for row in final_window])),
        "final_window_return_std": float(np.std([row["return"] for row in final_window])),
        "policy_agreement_percent": agreement["agreement_percent"],
        **{f"eval_{key}": value for key, value in evaluation.metrics.items()},
    }


def required_event_witnesses(env: DynamicMazeEnv, path: Path) -> list[dict]:
    witnesses: dict[str, dict] = {}
    for state in env.reachable_states(include_terminal=False):
        for selected_action in range(4):
            for outcome in env.transition_distribution(state, selected_action):
                for event in outcome.events:
                    if event in REQUIRED_EVENTS and event not in witnesses:
                        witnesses[event] = {
                            "event": event,
                            "source": "exact_transition_model",
                            "state": list(state),
                            "selected_action": selected_action,
                            "selected_action_name": ACTION_NAMES[selected_action],
                            "executed_action": outcome.executed_action,
                            "executed_action_name": ACTION_NAMES[outcome.executed_action],
                            "outcome_probability": outcome.probability,
                            "next_state": list(outcome.next_state),
                            "reward": outcome.reward,
                            "terminated": outcome.terminated,
                            "truncated": False,
                        }
    truncation_env = DynamicMazeEnv(
        env.maze,
        reward_mode=env.reward_mode,
        reward_spec=env.reward_spec,
        gamma=env.gamma,
        max_steps=1,
        seed=5,
    )
    state, _ = truncation_env.reset(seed=5)
    next_state, reward, terminated, truncated, info = truncation_env.step(3)
    if truncated:
        witnesses["max_steps_truncation"] = {
            "event": "max_steps_truncation",
            "source": "sampled_step_with_one_step_limit",
            "state": list(state),
            "selected_action": 3,
            "selected_action_name": "right",
            "executed_action": info["executed_action"],
            "executed_action_name": info["executed_action_name"],
            "outcome_probability": None,
            "next_state": list(next_state),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
    missing = REQUIRED_EVENTS - set(witnesses)
    if missing:
        raise AssertionError(f"No real transition witness found for required events: {sorted(missing)}")
    ordered = [witnesses[event] for event in sorted(witnesses)]
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered).encode("utf-8")
    write_bytes_verified(path, payload)
    return ordered


def write_sampled_event_log(maze, config: dict, q_table: np.ndarray, path: Path) -> dict:
    """Log natural greedy episodes, then actual-step probes for rare events."""

    with EventLogger(path) as logger:
        natural_env = DynamicMazeEnv(
            maze,
            reward_mode="shaped",
            reward_spec=RewardSpec.from_dict(config["reward"]),
            gamma=float(config["gamma"]),
            max_steps=int(config["max_steps_multiplier"] * maze.walkable_count),
            seed=5,
            event_logger=logger,
        )
        evaluate_q_policy(natural_env, q_table, episodes=10, seed=55_005)

    def observed_events() -> set[str]:
        events: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            events.update(record.get("events", []))
        return events

    missing = REQUIRED_EVENTS - observed_events()
    witnesses = {
        row["event"]: row
        for row in (
            json.loads(line)
            for line in (ROOT / "results/raw_data/required_event_witnesses.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    with EventLogger(path, mode="a") as logger:
        for event in sorted(missing):
            witness = witnesses[event]
            max_steps = 1 if event == "max_steps_truncation" else int(config["max_steps_multiplier"] * maze.walkable_count)
            probe = DynamicMazeEnv(
                maze,
                reward_mode="shaped",
                reward_spec=RewardSpec.from_dict(config["reward"]),
                gamma=float(config["gamma"]),
                max_steps=max_steps,
                seed=5,
                event_logger=logger,
            )
            probe.reset(seed=5)
            if event != "max_steps_truncation":
                probe.state = tuple(witness["state"])
                selected = int(witness["selected_action"])
                desired = int(witness["executed_action"])
                choices = (selected, *PERPENDICULAR_ACTIONS[selected])
                for candidate_seed in range(10_000):
                    candidate_rng = np.random.default_rng(candidate_seed)
                    if int(candidate_rng.choice(choices, p=(0.8, 0.1, 0.1))) == desired:
                        probe.rng = np.random.default_rng(candidate_seed)
                        break
                else:
                    raise AssertionError(f"Could not sample required executed action for {event}")
                probe.step(selected)
            else:
                probe.step(3)
    final_events = observed_events()
    still_missing = REQUIRED_EVENTS - final_events
    if still_missing:
        raise AssertionError(f"Sampled event log lacks required events: {sorted(still_missing)}")
    return {
        "record_count": len(path.read_text(encoding="utf-8").splitlines()),
        "covered_events": sorted(REQUIRED_EVENTS),
        "natural_evaluation_episodes": 10,
        "rare_event_probes": len(missing),
    }


def mismatch_examples(q_table, vi_result, env: DynamicMazeEnv, algorithm: str, limit: int = 3) -> list[dict]:
    candidates: list[tuple[float, tuple[int, int, int], int, int]] = []
    for state in env.reachable_states(include_terminal=False):
        has_key, row, col = state_index(state)
        learned = int(np.argmax(q_at(q_table, state)))
        reference = int(vi_result.policy[has_key, row, col])
        if learned == reference or reference < 0:
            continue
        exact_q = vi_result.q_values[has_key, row, col]
        regret = float(exact_q[reference] - exact_q[learned])
        candidates.append((regret, state, learned, reference))
    result: list[dict] = []
    for regret, state, learned, reference in sorted(candidates, reverse=True)[:limit]:
        has_key, row, col = state_index(state)
        result.append(
            {
                "algorithm": algorithm,
                "state": list(state),
                "tile": env.maze.tile((row, col)),
                "local_neighborhood": list(local_signature(env.maze, (row, col))),
                "learned_action": learned,
                "learned_action_name": ACTION_NAMES[learned],
                "value_iteration_action": reference,
                "value_iteration_action_name": ACTION_NAMES[reference],
                "learned_q_values": q_at(q_table, state).tolist(),
                "model_q_values": vi_result.q_values[has_key, row, col].tolist(),
                "exact_action_regret": regret,
            }
        )
    return result


def aggregate_summaries(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "runtime_seconds",
        "total_samples",
        "memory_bytes",
        "final_window_success_rate",
        "final_window_return_mean",
        "final_window_return_std",
        "policy_agreement_percent",
        "eval_success_rate",
        "eval_mean_return",
        "eval_std_return",
        "eval_mean_steps",
        "eval_mean_wall_collisions",
        "eval_mean_penalty_entries",
    ]
    existing = [column for column in numeric if column in frame.columns]
    grouped = frame.groupby(["algorithm", "variant", "reward_mode", "map_kind"], dropna=False)[existing]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=0).add_suffix("_across_seed_std")
    return mean.join(std).reset_index()


def validate_model_archives(model_dir: Path) -> dict:
    """Decode every model and reject object or non-finite numeric arrays."""

    archive_count = 0
    numeric_array_count = 0
    for path in sorted(model_dir.glob("*.npz")):
        archive_count += 1
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise AssertionError(f"Model lacks metadata_json: {path.name}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            if not isinstance(metadata, dict):
                raise AssertionError(f"Model metadata is not an object: {path.name}")
            if not isinstance(metadata.get("map_fingerprint"), str):
                raise AssertionError(f"Model lacks a map fingerprint: {path.name}")
            for name in archive.files:
                if name == "metadata_json":
                    continue
                values = archive[name]
                if not np.issubdtype(values.dtype, np.number):
                    raise AssertionError(f"Non-numeric array {name} in {path.name}")
                if not np.all(np.isfinite(values)):
                    raise AssertionError(f"Non-finite array {name} in {path.name}")
                numeric_array_count += 1
    return {
        "archive_count": archive_count,
        "numeric_array_count": numeric_array_count,
        "all_numeric_arrays_finite": True,
    }


def run_all(config: dict, config_path: Path, *, run_analysis: bool = True) -> dict:
    validate_config(config)
    raw_dir = ROOT / "results/raw_data"
    model_dir = ROOT / "results/models"
    figure_dir = ROOT / "results/figures"
    map_dir = ROOT / "environments/maps"
    for directory in (raw_dir, model_dir, figure_dir, map_dir, ROOT / "results/videos"):
        directory.mkdir(parents=True, exist_ok=True)
    removed_outputs = {
        "raw_data": remove_generated_files(raw_dir, {".csv", ".json", ".jsonl"}),
        "models": remove_generated_files(model_dir, {".npz"}),
        "figures": remove_generated_files(figure_dir, {".png"}),
        "videos": remove_generated_files(ROOT / "results/videos", {".gif", ".mp4", ".webm"}),
    }
    started_at = datetime.now(timezone.utc).isoformat()
    write_json(
        raw_dir / "run_manifest.json",
        {
            "status": "running",
            "started_at_utc": started_at,
            "profile": config["profile"],
            "student_id": config["student_id"],
            "config_path": str(config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path),
            "code_sha256": code_hash(),
            "removed_stale_generated_file_counts": {
                key: len(values) for key, values in removed_outputs.items()
            },
        },
    )

    source = generate_source_map(config["student_id"])
    similar = generate_transfer_map(source, "similar", seed=106)
    different = generate_transfer_map(source, "different", seed=207)
    maps = {"source": source, "similar": similar, "different": different}
    for name, maze in maps.items():
        maze.save(map_dir / f"{name}_map.json")
    map_fingerprints = {name: maze_fingerprint(maze) for name, maze in maps.items()}
    progress("generated and BFS-validated source/similar/different maps")
    validation = {name: validate_map(maze) for name, maze in maps.items()}
    validation["similar"]["actual_all_obstacle_relocation_fraction"] = similar.metadata["actual_all_obstacle_relocation_fraction"]
    validation["different"]["actual_all_obstacle_relocation_fraction"] = different.metadata["actual_all_obstacle_relocation_fraction"]
    validation["different"]["key_relocated"] = different.key != source.key
    validation["different"]["new_penalty_count"] = len(different.metadata["new_penalties"])
    write_json(raw_dir / "map_validation.json", validation)
    write_json(raw_dir / "executed_config.json", config)

    default_gamma = float(config["gamma"])
    reward_spec = RewardSpec.from_dict(config["reward"])
    vi_rows: list[dict] = []
    sparse_vi: dict[float, Any] = {}
    for gamma in map(float, config["vi_gammas"]):
        env = make_env(source, config, reward_mode="sparse", gamma=gamma)
        result = value_iteration(
            env,
            gamma=gamma,
            threshold=float(config["vi_threshold"]),
            max_iterations=int(config["vi_max_iterations"]),
        )
        if not result.converged:
            raise AssertionError(f"Value Iteration failed to converge for gamma={gamma}")
        residual = bellman_residual(env, result)
        sparse_vi[gamma] = result
        token = str(gamma).replace(".", "p")
        save_value_iteration(
            model_dir / f"vi_sparse_gamma_{token}.npz",
            result,
            {
                "map_kind": "source",
                "map_fingerprint": map_fingerprints["source"],
                "reward_mode": "sparse",
            },
        )
        vi_rows.append(
            {
                "gamma": gamma,
                "iterations": result.iterations,
                "runtime_seconds": result.runtime_seconds,
                "bellman_backups": result.bellman_backups,
                "memory_bytes": result.memory_bytes,
                "final_delta": result.delta_history[-1],
                "bellman_residual": residual,
            }
        )
        progress(f"Value Iteration gamma={gamma} converged in {result.iterations} iterations")
    shaped_env = make_env(source, config, reward_mode="shaped")
    shaped_vi = value_iteration(
        shaped_env,
        threshold=float(config["vi_threshold"]),
        max_iterations=int(config["vi_max_iterations"]),
    )
    if not shaped_vi.converged:
        raise AssertionError("Shaped-reward Value Iteration did not converge")
    save_value_iteration(
        model_dir / "vi_shaped_reference.npz",
        shaped_vi,
        {
            "map_kind": "source",
            "map_fingerprint": map_fingerprints["source"],
            "reward_mode": "shaped",
        },
    )
    closest_gamma = min(sparse_vi, key=lambda value: abs(value - default_gamma))
    if not np.isclose(closest_gamma, default_gamma):
        sparse_default_env = make_env(source, config, reward_mode="sparse")
        sparse_default = value_iteration(sparse_default_env, threshold=float(config["vi_threshold"]))
    else:
        sparse_default = sparse_vi[closest_gamma]
    comparison_mask = (sparse_default.policy >= 0) & (shaped_vi.policy >= 0)
    reward_policy_agreement = float(np.mean(sparse_default.policy[comparison_mask] == shaped_vi.policy[comparison_mask]))
    if reward_policy_agreement < 1.0:
        raise AssertionError("Potential-based shaping changed the converged optimal policy")
    write_json(
        raw_dir / "reward_design_verification.json",
        {
            "method": "F(s,s') = shaping_scale * (gamma * Phi(s') - Phi(s))",
            "potential": "negative deterministic BFS mission distance",
            "sparse_vs_shaped_policy_agreement": reward_policy_agreement,
            "compared_reachable_states": int(np.count_nonzero(comparison_mask)),
            "repeatable_door_bonus": reward_spec.door_passed,
        },
    )
    progress("verified sparse/shaped optimal-policy invariance and event coverage")
    write_dataframe_verified(raw_dir / "vi_gamma_sensitivity.csv", pd.DataFrame(vi_rows))
    required_event_witnesses(shaped_env, raw_dir / "required_event_witnesses.jsonl")

    training_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    summary_rows: list[dict] = []
    q_runs: dict[tuple[str, int], Any] = {}
    sarsa_runs: dict[tuple[float, int], Any] = {}
    representative_paths: dict[str, list[list[int]]] = {}
    seeds = [int(value) for value in config["seeds"]]
    base_seed = seeds[0]
    common_evaluation_seed = base_seed + 9_000
    common_transfer_evaluation_seed = base_seed + 20_000

    # Reference model-based policy evaluation uses its exact one-step Q values.
    vi_eval = evaluate_q_policy(
        make_env(source, config, reward_mode="shaped"),
        shaped_vi.q_values,
        episodes=int(config["evaluation_episodes"]),
        seed=common_evaluation_seed,
    )
    summary_rows.append(
        {
            "run_id": "vi_shaped_reference",
            "algorithm": "value_iteration",
            "variant": "gamma_0.95",
            "seed": -1,
            "reward_mode": "shaped",
            "map_kind": "source",
            "runtime_seconds": shaped_vi.runtime_seconds,
            "total_samples": 0,
            "bellman_backups": shaped_vi.bellman_backups,
            "memory_bytes": shaped_vi.memory_bytes,
            "final_window_success_rate": np.nan,
            "final_window_return_mean": np.nan,
            "final_window_return_std": np.nan,
            "policy_agreement_percent": 100.0,
            **{f"eval_{key}": value for key, value in vi_eval.metrics.items()},
        }
    )
    append_evaluation_rows(
        evaluation_rows,
        vi_eval,
        run_id="vi_shaped_reference",
        phase="final",
        algorithm="value_iteration",
        variant="gamma_0.95",
        seed=-1,
        reward_mode="shaped",
        map_kind="source",
    )
    representative_paths["value_iteration"] = [list(state) for state in vi_eval.representative_path]

    for schedule_mode in ("linear", "exponential"):
        schedule = epsilon_schedule(config, schedule_mode)
        q_config = QLearningConfig(
            episodes=int(config["q_episodes"]),
            alpha=float(config["q_alpha"]),
            gamma=default_gamma,
            epsilon=schedule,
            trace_updates=20,
        )
        for seed in seeds:
            run_id = f"q_shaped_{schedule_mode}_seed_{seed}"
            result = train_q_learning(make_env(source, config, reward_mode="shaped"), q_config, seed=seed)
            evaluation = evaluate_q_policy(
                make_env(source, config, reward_mode="shaped"),
                result.q_table,
                episodes=int(config["evaluation_episodes"]),
                seed=common_evaluation_seed,
            )
            agreement = policy_agreement(result.q_table, shaped_vi.policy, shaped_env)
            q_runs[(schedule_mode, seed)] = result
            append_training_rows(training_rows, result, run_id=run_id, algorithm="q_learning", variant=schedule_mode, seed=seed, reward_mode="shaped", map_kind="source")
            append_evaluation_rows(evaluation_rows, evaluation, run_id=run_id, phase="final", algorithm="q_learning", variant=schedule_mode, seed=seed, reward_mode="shaped", map_kind="source")
            summary_rows.append(model_free_summary(result, evaluation, agreement, config, run_id=run_id, algorithm="q_learning", variant=schedule_mode, seed=seed, reward_mode="shaped", map_kind="source"))
            save_q_checkpoint(model_dir / f"{run_id}.npz", result.q_table, metadata={"algorithm": "q_learning", "variant": schedule_mode, "seed": seed, "reward_mode": "shaped", "map_kind": "source", "map_fingerprint": map_fingerprints["source"], "config": result.config}, visits=result.visits)
            if seed == base_seed and schedule_mode == "linear":
                write_dataframe_verified(raw_dir / "q_update_trace.csv", pd.DataFrame(result.update_trace))
                representative_paths["q_learning"] = [list(state) for state in evaluation.representative_path]
            progress(f"Q-Learning shaped schedule={schedule_mode} seed={seed} complete")

    # Sparse-versus-shaped training comparison. Shaped/linear above is reused;
    # no successful or failed run is discarded.
    sparse_schedule = epsilon_schedule(config, "linear")
    sparse_q_config = QLearningConfig(
        episodes=int(config["q_episodes"]),
        alpha=float(config["q_alpha"]),
        gamma=default_gamma,
        epsilon=sparse_schedule,
        trace_updates=0,
    )
    sparse_env_for_agreement = make_env(source, config, reward_mode="sparse")
    for seed in seeds:
        run_id = f"q_sparse_linear_seed_{seed}"
        result = train_q_learning(make_env(source, config, reward_mode="sparse"), sparse_q_config, seed=seed)
        evaluation = evaluate_q_policy(make_env(source, config, reward_mode="sparse"), result.q_table, episodes=int(config["evaluation_episodes"]), seed=common_evaluation_seed)
        agreement = policy_agreement(result.q_table, sparse_default.policy, sparse_env_for_agreement)
        append_training_rows(training_rows, result, run_id=run_id, algorithm="q_learning", variant="linear", seed=seed, reward_mode="sparse", map_kind="source")
        append_evaluation_rows(evaluation_rows, evaluation, run_id=run_id, phase="final", algorithm="q_learning", variant="linear", seed=seed, reward_mode="sparse", map_kind="source")
        summary_rows.append(model_free_summary(result, evaluation, agreement, config, run_id=run_id, algorithm="q_learning", variant="linear", seed=seed, reward_mode="sparse", map_kind="source"))
        save_q_checkpoint(model_dir / f"{run_id}.npz", result.q_table, metadata={"algorithm": "q_learning", "variant": "linear", "seed": seed, "reward_mode": "sparse", "map_kind": "source", "map_fingerprint": map_fingerprints["source"], "config": result.config}, visits=result.visits)
        progress(f"Q-Learning sparse seed={seed} complete")

    for lambda_value in map(float, config["sarsa_lambdas"]):
        schedule = epsilon_schedule(config, "linear")
        sarsa_config = SarsaLambdaConfig(
            episodes=int(config["sarsa_episodes"]),
            alpha=float(config["sarsa_alpha"]),
            gamma=default_gamma,
            lambda_value=lambda_value,
            epsilon=schedule,
            trace_type=str(config["sarsa_trace_type"]),
            trace_steps=30,
        )
        lambda_token = str(lambda_value).replace(".", "p")
        for seed in seeds:
            run_id = f"sarsa_shaped_lambda_{lambda_token}_seed_{seed}"
            result = train_sarsa_lambda(make_env(source, config, reward_mode="shaped"), sarsa_config, seed=seed)
            evaluation = evaluate_q_policy(make_env(source, config, reward_mode="shaped"), result.q_table, episodes=int(config["evaluation_episodes"]), seed=common_evaluation_seed)
            agreement = policy_agreement(result.q_table, shaped_vi.policy, shaped_env)
            sarsa_runs[(lambda_value, seed)] = result
            append_training_rows(training_rows, result, run_id=run_id, algorithm="sarsa_lambda", variant=f"lambda_{lambda_value}", seed=seed, reward_mode="shaped", map_kind="source")
            append_evaluation_rows(evaluation_rows, evaluation, run_id=run_id, phase="final", algorithm="sarsa_lambda", variant=f"lambda_{lambda_value}", seed=seed, reward_mode="shaped", map_kind="source")
            summary_rows.append(model_free_summary(result, evaluation, agreement, config, run_id=run_id, algorithm="sarsa_lambda", variant=f"lambda_{lambda_value}", seed=seed, reward_mode="shaped", map_kind="source"))
            save_q_checkpoint(model_dir / f"{run_id}.npz", result.q_table, metadata={"algorithm": "sarsa_lambda", "variant": f"lambda_{lambda_value}", "seed": seed, "reward_mode": "shaped", "map_kind": "source", "map_fingerprint": map_fingerprints["source"], "config": result.config}, visits=result.visits)
            if seed == base_seed and np.isclose(lambda_value, 0.7):
                write_dataframe_verified(raw_dir / "sarsa_trace.csv", pd.DataFrame(result.update_trace))
                representative_paths["sarsa_lambda_0.7"] = [list(state) for state in evaluation.representative_path]
            progress(f"SARSA lambda={lambda_value} seed={seed} complete")

    mismatch_data = []
    mismatch_data.extend(mismatch_examples(q_runs[("linear", base_seed)].q_table, shaped_vi, shaped_env, "q_learning_linear"))
    mismatch_data.extend(mismatch_examples(sarsa_runs[(0.7, base_seed)].q_table, shaped_vi, shaped_env, "sarsa_lambda_0.7"))
    if len([row for row in mismatch_data if row["algorithm"] == "q_learning_linear"]) < 3 or len([row for row in mismatch_data if row["algorithm"] == "sarsa_lambda_0.7"]) < 3:
        raise AssertionError("Fewer than three model-free/reference policy mismatch examples were found")
    write_json(raw_dir / "policy_mismatch_examples.json", mismatch_data)
    sampled_event_evidence = write_sampled_event_log(
        source,
        config,
        q_runs[("linear", base_seed)].q_table,
        raw_dir / "sampled_event_log.jsonl",
    )
    write_json(raw_dir / "sampled_event_log_summary.json", sampled_event_evidence)

    transfer_episode_rows: list[dict] = []
    transfer_summary_rows: list[dict] = []
    negative_witnesses: list[dict] = []
    target_vi_results: dict[str, Any] = {}
    for target_name, target_map in (("similar", similar), ("different", different)):
        target_env = make_env(target_map, config, reward_mode="shaped")
        target_vi = value_iteration(target_env, threshold=float(config["vi_threshold"]), max_iterations=int(config["vi_max_iterations"]))
        if not target_vi.converged:
            raise AssertionError(f"Target Value Iteration failed for {target_name}")
        target_vi_results[target_name] = target_vi
        save_value_iteration(model_dir / f"vi_shaped_{target_name}.npz", target_vi, {"map_kind": target_name, "map_fingerprint": map_fingerprints[target_name], "reward_mode": "shaped"})
        scenarios: list[tuple[str, str, float | None]] = [
            ("scratch", "scratch", None),
            ("full", "full", None),
            *[(f"scaled_beta_{beta:.2f}", "scaled", float(beta)) for beta in config["transfer_betas"]],
            ("selective", "selective", None),
        ]
        for scenario, mode, beta in scenarios:
            transfer_schedule = epsilon_schedule(config, "linear", transfer=True)
            target_config = QLearningConfig(
                episodes=int(config["transfer_episodes"]),
                alpha=float(config["q_alpha"]),
                gamma=default_gamma,
                epsilon=transfer_schedule,
                trace_updates=0,
            )
            for seed in seeds:
                source_q = q_runs[("linear", seed)].q_table
                initialization = initialize_transfer(source_q, source, target_map, mode=mode, beta=beta)
                run_id = f"transfer_{target_name}_{scenario}_seed_{seed}".replace(".", "p")
                initial_evaluation = evaluate_q_policy(make_env(target_map, config, reward_mode="shaped"), initialization.q_table, episodes=int(config["initial_transfer_evaluation_episodes"]), seed=common_transfer_evaluation_seed)
                result = train_q_learning(make_env(target_map, config, reward_mode="shaped"), target_config, seed=seed + 30_000, initial_q=initialization.q_table)
                final_evaluation = evaluate_q_policy(make_env(target_map, config, reward_mode="shaped"), result.q_table, episodes=int(config["evaluation_episodes"]), seed=common_transfer_evaluation_seed)
                agreement = policy_agreement(result.q_table, target_vi.policy, target_env)
                episode_to_threshold, samples_to_threshold = learning_speed(result.episode_metrics, int(config["success_window"]), float(config["success_threshold"]))
                for row in result.episode_metrics:
                    transfer_episode_rows.append({"run_id": run_id, "target": target_name, "scenario": scenario, "mode": mode, "beta": beta, "seed": seed, **row})
                transfer_summary_rows.append(
                    {
                        "run_id": run_id,
                        "target": target_name,
                        "scenario": scenario,
                        "mode": mode,
                        "beta": beta,
                        "seed": seed,
                        "transferred_state_count": initialization.transferred_state_count,
                        "evaluation_base_seed": common_transfer_evaluation_seed,
                        "initial_success_rate": initial_evaluation.metrics["success_rate"],
                        "initial_mean_return": initial_evaluation.metrics["mean_return"],
                        "episodes_to_success_threshold": episode_to_threshold,
                        "samples_to_success_threshold": samples_to_threshold,
                        "final_success_rate": final_evaluation.metrics["success_rate"],
                        "final_mean_return": final_evaluation.metrics["mean_return"],
                        "final_mean_steps": final_evaluation.metrics["mean_steps"],
                        "runtime_seconds": result.runtime_seconds,
                        "total_samples": result.total_samples,
                        "memory_bytes": result.memory_bytes,
                        "policy_agreement_percent": agreement["agreement_percent"],
                    }
                )
                save_q_checkpoint(model_dir / f"{run_id}.npz", result.q_table, metadata={"algorithm": "q_learning_transfer", "target": target_name, "map_fingerprint": map_fingerprints[target_name], "reward_mode": "shaped", "scenario": scenario, "mode": mode, "beta": beta, "seed": seed, "config": result.config}, visits=result.visits)
                if mode == "full" and seed == base_seed:
                    witness = find_negative_transfer_witness(source_q, source, target_env, target_vi.policy, target_vi.q_values, trained_target_q=result.q_table)
                    if witness is not None:
                        negative_witnesses.append({"target": target_name, "scenario": scenario, "seed": seed, **witness})
                    representative_paths[f"transfer_{target_name}_full"] = [list(state) for state in final_evaluation.representative_path]
                progress(f"transfer target={target_name} scenario={scenario} seed={seed} complete")
    if not negative_witnesses:
        raise AssertionError("No negative-transfer witness was found; transfer claim would be unsupported")
    write_json(raw_dir / "negative_transfer_witnesses.json", negative_witnesses)
    write_json(raw_dir / "representative_paths.json", representative_paths)

    training_frame = pd.DataFrame(training_rows)
    evaluation_frame = pd.DataFrame(evaluation_rows)
    summary_frame = pd.DataFrame(summary_rows)
    transfer_episode_frame = pd.DataFrame(transfer_episode_rows)
    transfer_summary_frame = pd.DataFrame(transfer_summary_rows)
    write_dataframe_verified(raw_dir / "training_episodes.csv", training_frame)
    write_dataframe_verified(raw_dir / "evaluation_episodes.csv", evaluation_frame)
    write_dataframe_verified(raw_dir / "run_summary.csv", summary_frame)
    write_dataframe_verified(raw_dir / "aggregate_summary.csv", aggregate_summaries(summary_frame))
    write_dataframe_verified(raw_dir / "transfer_episodes.csv", transfer_episode_frame)
    write_dataframe_verified(raw_dir / "transfer_summary.csv", transfer_summary_frame)
    transfer_aggregate = transfer_summary_frame.groupby(["target", "scenario", "mode", "beta"], dropna=False).agg(
        initial_success_rate_mean=("initial_success_rate", "mean"),
        initial_success_rate_std=("initial_success_rate", lambda values: values.std(ddof=0)),
        final_success_rate_mean=("final_success_rate", "mean"),
        final_success_rate_std=("final_success_rate", lambda values: values.std(ddof=0)),
        final_mean_return_mean=("final_mean_return", "mean"),
        policy_agreement_percent_mean=("policy_agreement_percent", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
        total_samples_mean=("total_samples", "mean"),
    ).reset_index()
    write_dataframe_verified(raw_dir / "transfer_aggregate_summary.csv", transfer_aggregate)

    configured_seed_count = len(seeds)
    source_model_free_runs_per_seed = 3 + len(config["sarsa_lambdas"])
    transfer_scenarios_per_target = 3 + len(config["transfer_betas"])
    expected_counts = {
        "training_runs": 1 + source_model_free_runs_per_seed * configured_seed_count,
        "transfer_runs": 2 * transfer_scenarios_per_target * configured_seed_count,
        "training_episode_rows": configured_seed_count
        * (
            3 * int(config["q_episodes"])
            + len(config["sarsa_lambdas"]) * int(config["sarsa_episodes"])
        ),
        "transfer_episode_rows": 2
        * transfer_scenarios_per_target
        * configured_seed_count
        * int(config["transfer_episodes"]),
        "model_archives": len(config["vi_gammas"])
        + 3
        + source_model_free_runs_per_seed * configured_seed_count
        + 2 * transfer_scenarios_per_target * configured_seed_count,
    }
    observed_counts = {
        "training_runs": int(summary_frame.shape[0]),
        "transfer_runs": int(transfer_summary_frame.shape[0]),
        "training_episode_rows": int(training_frame.shape[0]),
        "transfer_episode_rows": int(transfer_episode_frame.shape[0]),
        "model_archives": len(list(model_dir.glob("*.npz"))),
    }
    if observed_counts != expected_counts:
        raise AssertionError(
            f"Experiment output counts do not match the configured matrix: expected {expected_counts}, "
            f"observed {observed_counts}"
        )
    model_validation = validate_model_archives(model_dir)

    manifest = {
        "status": "running",
        "started_at_utc": started_at,
        "profile": config["profile"],
        "student_id": config["student_id"],
        "base_seed": int(config["student_id"][-2]),
        "maze_size": 15 + int(config["student_id"][-2]) % 4,
        "config_path": str(config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path),
        "config_sha256": _sha256(config_path),
        "code_sha256": code_hash(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": package_version("matplotlib"),
        "pillow": package_version("Pillow"),
        "map_sha256": {name: _sha256(map_dir / f"{name}_map.json") for name in maps},
        "map_fingerprints": map_fingerprints,
        "raw_files": sorted(path.name for path in raw_dir.iterdir() if path.is_file()),
        "model_count": observed_counts["model_archives"],
        "run_counts": {key: value for key, value in observed_counts.items() if key != "model_archives"},
        "expected_counts": expected_counts,
        "model_validation": model_validation,
        "memory_metric": "tabular array bytes: retained Q/visits plus peak SARSA eligibility workspace",
        "evaluation_rng_design": "common environment episode seeds across algorithms and transfer scenarios",
        "all_configured_runs_retained": True,
        "removed_stale_generated_file_counts": {
            key: len(values) for key, values in removed_outputs.items()
        },
        "reward_policy_agreement": reward_policy_agreement,
        "negative_transfer_witness_count": len(negative_witnesses),
    }

    if run_analysis:
        from experiments.analysis import generate_all_figures

        figure_manifest = generate_all_figures(ROOT)
        manifest["figure_count"] = len(figure_manifest)
        progress(f"generated {len(figure_manifest)} figures from persisted raw data")
    else:
        manifest["figure_count"] = 0

    # Re-read the durable files rather than trusting the in-memory frames.  This
    # is deliberately performed after analysis, because analysis is another
    # consumer that must see the complete data before the run can be completed.
    persisted_counts = {
        "training_runs": csv_row_count(raw_dir / "run_summary.csv"),
        "transfer_runs": csv_row_count(raw_dir / "transfer_summary.csv"),
        "training_episode_rows": csv_row_count(raw_dir / "training_episodes.csv"),
        "transfer_episode_rows": csv_row_count(raw_dir / "transfer_episodes.csv"),
        "model_archives": len(list(model_dir.glob("*.npz"))),
    }
    if persisted_counts != expected_counts:
        raise AssertionError(
            f"Persisted artifact counts do not match the configured matrix: expected {expected_counts}, "
            f"persisted {persisted_counts}"
        )
    manifest["run_counts"] = {
        key: value for key, value in persisted_counts.items() if key != "model_archives"
    }
    manifest["raw_sha256"] = {
        path.name: _sha256(path)
        for path in sorted(raw_dir.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    manifest["model_sha256"] = {
        path.name: _sha256(path) for path in sorted(model_dir.glob("*.npz"))
    }
    manifest["figure_sha256"] = {
        path.name: _sha256(path) for path in sorted(figure_dir.glob("*.png"))
    }
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["generated_at_utc"] = manifest["completed_at_utc"]
    manifest["raw_files"] = sorted(path.name for path in raw_dir.iterdir() if path.is_file())
    write_json(raw_dir / "run_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config path; defaults to the reproducible full.json profile")
    parser.add_argument("--skip-analysis", action="store_true", help="Run experiments without regenerating figures")
    parser.add_argument("--analysis-only", action="store_true", help="Regenerate figures from existing raw files/models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.analysis_only:
        from experiments.analysis import generate_all_figures

        manifest = generate_all_figures(ROOT)
        print(json.dumps({"status": "analysis_completed", "figure_count": len(manifest)}, indent=2))
        return
    config, config_path = load_config(args.config)
    manifest = run_all(config, config_path, run_analysis=not args.skip_analysis)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
