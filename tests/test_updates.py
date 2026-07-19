from pathlib import Path

import numpy as np

from agents.common import EpsilonSchedule, empty_q
from agents.q_learning import q_learning_update
from agents.sarsa_lambda import SarsaLambdaConfig, train_sarsa_lambda
from agents.value_iteration import value_iteration
from environments.maze import DynamicMazeEnv

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "environments" / "maps" / "source_map.json"


def test_q_learning_update_matches_manual_calculation() -> None:
    env = DynamicMazeEnv(MAP, reward_mode="sparse", seed=5)
    q = empty_q(env)
    state = (14, 2, 0, 0)
    next_state = (13, 2, 0, 1)
    q[next_state + (0,)] = 4.0
    result = q_learning_update(q, state, 0, reward=2.0, next_state=next_state, terminated=False, alpha=0.5, gamma=0.9)
    expected_target = 2.0 + 0.9 * 4.0
    expected_after = 0.5 * expected_target
    assert np.isclose(result["target"], expected_target)
    assert np.isclose(result["q_after"], expected_after)
    assert np.isclose(q[state + (0,)], expected_after)


def test_lambda_zero_runs_as_one_step_sarsa() -> None:
    env = DynamicMazeEnv(MAP, reward_mode="shaping", gamma=0.95, seed=5)
    result = train_sarsa_lambda(
        env,
        SarsaLambdaConfig(
            episodes=3,
            alpha=0.1,
            gamma=0.95,
            lambda_value=0.0,
            epsilon=EpsilonSchedule("constant", 0.2, 0.2, 1.0),
            seed=5,
        ),
        trace_episode=0,
        trace_steps=4,
    )
    assert len(result.history) == 3
    assert np.isfinite(result.q_values).all()
    assert not result.trace_log.empty
    # With lambda=0 traces are removed immediately after the current update.
    assert (result.trace_log["active_traces_after_decay"] == 0).all()


def test_value_iteration_converges_and_goal_is_terminal() -> None:
    env = DynamicMazeEnv(MAP, reward_mode="shaping", gamma=0.95, seed=5)
    result = value_iteration(env, gamma=0.95, tolerance=1e-8)
    assert result.converged
    assert result.iterations > 0
    for phase in range(env.gate_period):
        assert result.values[env.goal_pos + (1, phase)] == 0.0
        assert result.policy[env.goal_pos + (1, phase)] == -1
