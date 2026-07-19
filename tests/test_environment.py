from pathlib import Path

import numpy as np

from environments.generator import generate_source_map, validate_map
from environments.maze import Action, DynamicMazeEnv

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "environments" / "maps" / "source_map.json"


def test_student_seed_and_map_size() -> None:
    data = generate_source_map("40400356")
    assert data["metadata"]["base_seed"] == 5
    assert data["metadata"]["size"] == 16
    assert validate_map(data["grid"])["valid"] is True


def test_transition_probabilities_sum_to_one() -> None:
    env = DynamicMazeEnv(MAP, reward_mode="sparse", seed=5)
    state, _ = env.reset()
    for action in range(4):
        outcomes = env.transition_outcomes(state, action)
        assert np.isclose(sum(item.probability for item in outcomes), 1.0)
        assert set(round(item.probability, 10) for item in outcomes).issubset({0.1, 0.2, 0.8, 0.9, 1.0})


def test_key_and_locked_door_logic() -> None:
    env = DynamicMazeEnv(MAP, reward_mode="sparse", seed=5)
    next_state, reward, terminated, event = env._deterministic_transition((11, 3, 0, 0), Action.RIGHT)
    assert event == "key_collected"
    assert next_state[2] == 1
    assert reward > 0
    assert not terminated

    next_state, reward, _, event = env._deterministic_transition((10, 6, 0, 0), Action.RIGHT)
    assert event == "locked_door_attempt"
    assert next_state[:2] == (10, 6)
    assert reward < 0

    next_state, _, _, event = env._deterministic_transition((10, 6, 1, 0), Action.RIGHT)
    assert event == "door_crossed"
    assert next_state[:2] == env.door_pos


def test_periodic_gate_is_part_of_transition() -> None:
    env = DynamicMazeEnv(MAP, reward_mode="sparse", seed=5)
    blocked_state, _, _, blocked_event = env._deterministic_transition((8, 10, 1, 2), Action.UP)
    assert blocked_event == "periodic_gate_blocked"
    assert blocked_state[:2] == (8, 10)
    open_state, _, _, open_event = env._deterministic_transition((8, 10, 1, 0), Action.UP)
    assert open_event == "periodic_gate_crossed"
    assert open_state[:2] == env.gate_pos


def test_time_limit_is_reported_as_truncation() -> None:
    env = DynamicMazeEnv(MAP, reward_mode="sparse", seed=5, max_steps=1)
    env.reset(seed=1)
    _, _, terminated, truncated, info = env.step(Action.UP)
    assert not terminated
    assert truncated
    assert info["event"] == "step_limit_reached"
