"""Stochastic Markov environment for the key-door-goal maze."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .generator import (
    ACTION_DELTAS,
    DOOR,
    FLOOR,
    GOAL,
    KEY,
    PENALTY,
    PORTAL_A,
    PORTAL_B,
    WALL,
    MazeMap,
    shortest_mission_distance,
)


ACTION_NAMES = ("up", "down", "left", "right")
PERPENDICULAR_ACTIONS = {
    0: (2, 3),
    1: (2, 3),
    2: (0, 1),
    3: (0, 1),
}
REQUIRED_EVENTS = {
    "normal_move",
    "wall_collision",
    "penalty_entry",
    "key_collected",
    "closed_door_attempt",
    "door_passed",
    "goal_reached",
    "max_steps_truncation",
}

State = tuple[int, int, int]


@dataclass(frozen=True)
class RewardSpec:
    step: float = -1.0
    wall_collision: float = -4.0
    penalty_entry: float = -10.0
    key_collected: float = 30.0
    closed_door_attempt: float = -7.0
    # Door passage is logged but deliberately carries no repeatable bonus;
    # otherwise oscillating across the door becomes a reward-farming exploit.
    door_passed: float = 0.0
    goal_reached: float = 100.0
    # Keeps ordinary shaped moves non-positive on this map while retaining the
    # policy-invariance guarantee of potential-based shaping.
    shaping_scale: float = 0.25

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.step,
                self.wall_collision,
                self.penalty_entry,
                self.key_collected,
                self.closed_door_attempt,
                self.door_passed,
                self.goal_reached,
                self.shaping_scale,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("All reward parameters must be finite")
        if self.shaping_scale < 0:
            raise ValueError("shaping_scale must be non-negative")

    @classmethod
    def from_dict(cls, values: dict | None) -> "RewardSpec":
        return cls(**(values or {}))


class Transition(NamedTuple):
    probability: float
    next_state: State
    reward: float
    terminated: bool
    events: tuple[str, ...]
    executed_action: int


class EventLogger:
    """Append-only JSONL logger for actual sampled transitions."""

    def __init__(self, path: str | Path, *, mode: str = "w"):
        if mode not in {"w", "a"}:
            raise ValueError("EventLogger mode must be 'w' or 'a'")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open(mode, encoding="utf-8")

    def write(self, record: dict) -> None:
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class DynamicMazeEnv:
    """A lightweight Gym-like environment with explicit termination semantics.

    State is ``(row, column, has_key)``.  The teleporter is static and fully
    encoded by the current position and map, so it adds no hidden state.
    """

    action_space_n = 4

    def __init__(
        self,
        maze: MazeMap,
        *,
        reward_mode: str = "shaped",
        reward_spec: RewardSpec | dict | None = None,
        gamma: float = 0.95,
        max_steps: int | None = None,
        seed: int | None = None,
        event_logger: EventLogger | None = None,
    ) -> None:
        if reward_mode not in {"sparse", "shaped"}:
            raise ValueError("reward_mode must be 'sparse' or 'shaped'")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1)")
        self.maze = maze
        self.reward_mode = reward_mode
        self.reward_spec = (
            reward_spec if isinstance(reward_spec, RewardSpec) else RewardSpec.from_dict(reward_spec)
        )
        self.gamma = float(gamma)
        self.max_steps = int(3 * maze.walkable_count if max_steps is None else max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.rng = np.random.default_rng(seed)
        self.event_logger = event_logger
        self.episode_index = -1
        self.state: State = (maze.start[0], maze.start[1], 0)
        self.step_count = 0
        self.total_reward = 0.0
        self._episode_done = False
        self._potential_cache: dict[State, float] = {}

    @property
    def start_state(self) -> State:
        return self.maze.start[0], self.maze.start[1], 0

    @property
    def terminal_state(self) -> State:
        return self.maze.goal[0], self.maze.goal[1], 1

    @property
    def q_shape(self) -> tuple[int, int, int, int]:
        return (2, self.maze.height, self.maze.width, 4)

    @property
    def value_shape(self) -> tuple[int, int, int]:
        return (2, self.maze.height, self.maze.width)

    def reset(self, *, seed: int | None = None) -> tuple[State, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_index += 1
        self.state = self.start_state
        self.step_count = 0
        self.total_reward = 0.0
        self._episode_done = False
        return self.state, {
            "episode": self.episode_index,
            "max_steps": self.max_steps,
            "has_key": False,
        }

    def is_terminal(self, state: State) -> bool:
        return (state[0], state[1]) == self.maze.goal and state[2] == 1

    def _validate_state(self, state: State) -> None:
        if not isinstance(state, tuple) or len(state) != 3:
            raise ValueError("State must be a (row, column, has_key) tuple")
        row, col, has_key = state
        if not all(isinstance(value, (int, np.integer)) for value in state):
            raise ValueError("State entries must be integers")
        if has_key not in (0, 1):
            raise ValueError("State has_key must be 0 or 1")
        if not (0 <= row < self.maze.height and 0 <= col < self.maze.width):
            raise ValueError("State position is outside the maze")
        if self.maze.tile((int(row), int(col))) == WALL:
            raise ValueError("State cannot occupy a wall")

    def valid_states(self, *, include_terminal: bool = True) -> list[State]:
        states = [
            (row, col, key_status)
            for key_status in (0, 1)
            for row in range(self.maze.height)
            for col in range(self.maze.width)
            if self.maze.grid[row][col] != WALL
        ]
        return states if include_terminal else [state for state in states if not self.is_terminal(state)]

    def reachable_states(self, *, include_terminal: bool = True) -> list[State]:
        """All states reachable with nonzero probability from reset."""

        queue: deque[State] = deque([self.start_state])
        seen = {self.start_state}
        while queue:
            state = queue.popleft()
            if self.is_terminal(state):
                continue
            for action in range(4):
                for outcome in self.transition_distribution(state, action):
                    if outcome.next_state not in seen:
                        seen.add(outcome.next_state)
                        queue.append(outcome.next_state)
        ordered = sorted(seen)
        return ordered if include_terminal else [state for state in ordered if not self.is_terminal(state)]

    def potential(self, state: State) -> float:
        """Negative shortest remaining mission length (zero at goal)."""

        if self.is_terminal(state):
            return 0.0
        if state not in self._potential_cache:
            distance = shortest_mission_distance(self.maze, state)
            if distance is None:
                distance = 2 * self.maze.height * self.maze.width
            self._potential_cache[state] = -float(distance)
        return self._potential_cache[state]

    def _base_transition(self, state: State, executed_action: int) -> tuple[State, float, bool, tuple[str, ...]]:
        if executed_action not in range(4):
            raise ValueError(f"Invalid action {executed_action}; expected 0..3")
        self._validate_state(state)
        if self.is_terminal(state):
            return state, 0.0, True, ("terminal_absorbing",)
        row, col, has_key_int = state
        has_key = bool(has_key_int)
        dr, dc = ACTION_DELTAS[executed_action]
        candidate = (row + dr, col + dc)
        reward = self.reward_spec.step

        if not (0 <= candidate[0] < self.maze.height and 0 <= candidate[1] < self.maze.width):
            return state, reward + self.reward_spec.wall_collision, False, ("wall_collision",)
        tile = self.maze.tile(candidate)
        if tile == WALL:
            return state, reward + self.reward_spec.wall_collision, False, ("wall_collision",)
        if tile == DOOR and not has_key:
            return state, reward + self.reward_spec.closed_door_attempt, False, ("closed_door_attempt",)

        events: tuple[str, ...]
        if tile in (PORTAL_A, PORTAL_B):
            candidate = self.maze.portals[candidate]
            tile = self.maze.tile(candidate)
            events = ("teleport",)
        elif tile == PENALTY:
            reward += self.reward_spec.penalty_entry
            events = ("penalty_entry",)
        elif tile == KEY and not has_key:
            reward += self.reward_spec.key_collected
            has_key = True
            events = ("key_collected",)
        elif tile == DOOR:
            reward += self.reward_spec.door_passed
            events = ("door_passed",)
        elif tile == GOAL and has_key:
            reward += self.reward_spec.goal_reached
            events = ("goal_reached",)
        else:
            events = ("normal_move",)

        next_state = candidate[0], candidate[1], int(has_key)
        terminated = self.is_terminal(next_state)
        return next_state, reward, terminated, events

    def transition_for_executed_action(self, state: State, executed_action: int) -> Transition:
        next_state, sparse_reward, terminated, events = self._base_transition(state, executed_action)
        reward = sparse_reward
        if self.reward_mode == "shaped":
            reward += self.reward_spec.shaping_scale * (
                self.gamma * self.potential(next_state) - self.potential(state)
            )
        return Transition(1.0, next_state, float(reward), terminated, events, executed_action)

    def transition_distribution(self, state: State, selected_action: int) -> list[Transition]:
        """Exact P(s', r | s, a), including 0.8/0.1/0.1 action noise."""

        if selected_action not in range(4):
            raise ValueError(f"Invalid action {selected_action}; expected 0..3")
        executed = (selected_action, *PERPENDICULAR_ACTIONS[selected_action])
        probabilities = (0.8, 0.1, 0.1)
        aggregate: dict[tuple, float] = defaultdict(float)
        examples: dict[tuple, Transition] = {}
        for probability, actual_action in zip(probabilities, executed):
            outcome = self.transition_for_executed_action(state, actual_action)
            key = (
                outcome.next_state,
                outcome.reward,
                outcome.terminated,
                outcome.events,
                outcome.executed_action,
            )
            aggregate[key] += probability
            examples[key] = outcome
        return [
            Transition(
                probability,
                examples[key].next_state,
                examples[key].reward,
                examples[key].terminated,
                examples[key].events,
                examples[key].executed_action,
            )
            for key, probability in aggregate.items()
        ]

    def step(self, selected_action: int) -> tuple[State, float, bool, bool, dict]:
        if self._episode_done:
            raise RuntimeError("Episode is done; call reset() before step()")
        if selected_action not in range(4):
            raise ValueError(f"Invalid action {selected_action}; expected 0..3")
        previous_state = self.state
        perpendicular = PERPENDICULAR_ACTIONS[selected_action]
        actual_action = int(
            self.rng.choice((selected_action, *perpendicular), p=(0.8, 0.1, 0.1))
        )
        outcome = self.transition_for_executed_action(previous_state, actual_action)
        self.state = outcome.next_state
        self.step_count += 1
        self.total_reward += outcome.reward
        terminated = outcome.terminated
        truncated = self.step_count >= self.max_steps and not terminated
        events = list(outcome.events)
        if truncated:
            events.append("max_steps_truncation")
        self._episode_done = terminated or truncated
        info = {
            "episode": self.episode_index,
            "step": self.step_count,
            "selected_action": selected_action,
            "selected_action_name": ACTION_NAMES[selected_action],
            "executed_action": actual_action,
            "executed_action_name": ACTION_NAMES[actual_action],
            "events": events,
            "has_key": bool(self.state[2]),
            "total_reward": self.total_reward,
        }
        if self.event_logger is not None:
            self.event_logger.write(
                {
                    **info,
                    "state": list(previous_state),
                    "next_state": list(self.state),
                    "reward": outcome.reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "reward_mode": self.reward_mode,
                }
            )
        return self.state, outcome.reward, terminated, truncated, info

    def configuration(self) -> dict:
        return {
            "reward_mode": self.reward_mode,
            "reward_spec": asdict(self.reward_spec),
            "gamma": self.gamma,
            "max_steps": self.max_steps,
            "slip_probabilities": [0.8, 0.1, 0.1],
            "state_definition": ["row", "column", "has_key"],
            "action_names": list(ACTION_NAMES),
        }
