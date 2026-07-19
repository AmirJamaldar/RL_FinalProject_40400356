"""Stochastic dynamic maze environment and exact transition model."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np

from environments.generator import DOOR, GATE, GOAL, KEY, NORMAL, PENALTY, START, WALL, load_map


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}
ACTION_NAMES = {Action.UP: "up", Action.DOWN: "down", Action.LEFT: "left", Action.RIGHT: "right"}
PERPENDICULAR = {
    Action.UP: (Action.LEFT, Action.RIGHT),
    Action.DOWN: (Action.LEFT, Action.RIGHT),
    Action.LEFT: (Action.UP, Action.DOWN),
    Action.RIGHT: (Action.UP, Action.DOWN),
}

State = tuple[int, int, int, int]  # row, column, has_key, gate_phase


@dataclass(frozen=True)
class RewardConfig:
    step: float = -0.50
    wall: float = -1.50
    penalty: float = -7.00
    key: float = 25.00
    locked_door: float = -4.00
    goal: float = 120.00
    shaping_scale: float = 0.20


class Transition(NamedTuple):
    probability: float
    next_state: State
    reward: float
    terminated: bool
    event: str
    realized_action: int


class DynamicMazeEnv:
    """A reproducible, model-queryable stochastic maze.

    The stationary MDP state is (row, column, has_key, gate_phase).  The
    episode step cap is implemented as a TimeLimit-style truncation wrapper;
    it is deliberately not part of the stationary state used by Value
    Iteration.
    """

    action_space_n = 4

    def __init__(
        self,
        map_path: str | Path,
        *,
        reward_mode: str = "sparse",
        gamma: float = 0.95,
        reward_config: RewardConfig | None = None,
        seed: int = 0,
        max_steps: int | None = None,
    ) -> None:
        if reward_mode not in {"sparse", "shaping"}:
            raise ValueError("reward_mode must be 'sparse' or 'shaping'")
        self.map_path = Path(map_path)
        self.map_data = load_map(self.map_path)
        self.grid = np.array([list(row) for row in self.map_data["grid"]], dtype="U1")
        self.size = int(self.map_data["metadata"]["size"])
        self.gate_period = int(self.map_data["metadata"].get("gate_period", 4))
        self.gate_open_phases = tuple(int(x) for x in self.map_data["metadata"].get("gate_open_phases", [0, 1]))
        self.reward_mode = reward_mode
        self.gamma = float(gamma)
        self.rewards = reward_config or RewardConfig()
        self.rng = np.random.default_rng(seed)

        self.start_pos = self._find(START)
        self.key_pos = self._find(KEY)
        self.door_pos = self._find(DOOR)
        self.goal_pos = self._find(GOAL)
        self.gate_pos = self._find(GATE)
        self.valid_positions = [
            (r, c) for r in range(self.size) for c in range(self.size) if self.grid[r, c] != WALL
        ]
        self.max_steps = max_steps or (3 * len(self.valid_positions))
        self._distance_to_key = self._all_distances(self.key_pos)
        self._distance_to_goal = self._all_distances(self.goal_pos)
        self._key_to_goal = self._distance_to_goal.get(self.key_pos, self.size * self.size)
        self.state: State = (self.start_pos[0], self.start_pos[1], 0, 0)
        self.steps = 0
        self.episode_return = 0.0
        self.last_info: dict[str, object] = {}

    def _find(self, symbol: str) -> tuple[int, int]:
        positions = np.argwhere(self.grid == symbol)
        if len(positions) != 1:
            raise ValueError(f"expected one {symbol}, found {len(positions)}")
        return tuple(int(x) for x in positions[0])

    def clone(self, *, seed: int | None = None, reward_mode: str | None = None, gamma: float | None = None) -> "DynamicMazeEnv":
        return DynamicMazeEnv(
            self.map_path,
            reward_mode=reward_mode or self.reward_mode,
            gamma=self.gamma if gamma is None else gamma,
            reward_config=self.rewards,
            seed=0 if seed is None else seed,
            max_steps=self.max_steps,
        )

    def _all_distances(self, target: tuple[int, int]) -> dict[tuple[int, int], int]:
        """Static shortest-path distances with the door/gate treated as traversable."""
        q: deque[tuple[int, int]] = deque([target])
        dist = {target: 0}
        while q:
            r, c = q.popleft()
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.size and 0 <= nc < self.size):
                    continue
                if self.grid[nr, nc] == WALL or (nr, nc) in dist:
                    continue
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
        return dist

    def potential(self, state: State) -> float:
        r, c, has_key, _phase = state
        if (r, c) == self.goal_pos and has_key:
            return 0.0
        fallback = self.size * self.size
        if has_key:
            return -float(self._distance_to_goal.get((r, c), fallback))
        return -float(self._distance_to_key.get((r, c), fallback) + self._key_to_goal)

    def is_gate_open(self, phase: int) -> bool:
        return int(phase) % self.gate_period in self.gate_open_phases

    def is_terminal(self, state: State) -> bool:
        r, c, has_key, _phase = state
        return bool(has_key and (r, c) == self.goal_pos)

    def reset(self, *, seed: int | None = None, phase: int = 0) -> tuple[State, dict[str, object]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = (self.start_pos[0], self.start_pos[1], 0, int(phase) % self.gate_period)
        self.steps = 0
        self.episode_return = 0.0
        self.last_info = {
            "event": "episode_start",
            "has_key": False,
            "gate_open": self.is_gate_open(phase),
            "terminated": False,
            "truncated": False,
        }
        return self.state, dict(self.last_info)

    def _deterministic_transition(self, state: State, realized_action: Action) -> tuple[State, float, bool, str]:
        if self.is_terminal(state):
            return state, 0.0, True, "terminal_self_loop"

        r, c, has_key, phase = state
        dr, dc = ACTION_DELTAS[realized_action]
        nr, nc = r + dr, c + dc
        next_phase = (phase + 1) % self.gate_period
        event = "normal_move"
        base_reward = self.rewards.step

        tile = WALL
        if 0 <= nr < self.size and 0 <= nc < self.size:
            tile = self.grid[nr, nc]

        blocked = False
        if tile == WALL:
            blocked = True
            event = "wall_collision"
            base_reward += self.rewards.wall
        elif tile == DOOR and not has_key:
            blocked = True
            event = "locked_door_attempt"
            base_reward += self.rewards.locked_door
        elif tile == GATE and not self.is_gate_open(phase):
            blocked = True
            event = "periodic_gate_blocked"
            base_reward += self.rewards.wall

        if blocked:
            next_state = (r, c, has_key, next_phase)
        else:
            next_has_key = has_key
            if tile == KEY and not has_key:
                next_has_key = 1
                event = "key_collected"
                base_reward += self.rewards.key
            elif tile == PENALTY:
                event = "penalty_cell_entered"
                base_reward += self.rewards.penalty
            elif tile == DOOR and has_key:
                event = "door_crossed"
            elif tile == GATE:
                event = "periodic_gate_crossed"
            elif tile == GOAL and has_key:
                event = "goal_reached"
                base_reward += self.rewards.goal
            next_state = (nr, nc, next_has_key, next_phase)

        terminated = self.is_terminal(next_state)
        if self.reward_mode == "shaping":
            shaping = self.rewards.shaping_scale * (
                self.gamma * self.potential(next_state) - self.potential(state)
            )
            base_reward += shaping
        return next_state, float(base_reward), terminated, event

    def transition_outcomes(self, state: State, action: int | Action) -> list[Transition]:
        """Return the exact P(s', r | s, a), merging identical outcomes."""
        action = Action(int(action))
        branches = [(action, 0.8), (PERPENDICULAR[action][0], 0.1), (PERPENDICULAR[action][1], 0.1)]
        merged: dict[tuple[State, float, bool, str, int], float] = defaultdict(float)
        for realized, probability in branches:
            ns, reward, terminated, event = self._deterministic_transition(state, realized)
            key = (ns, reward, terminated, event, int(realized))
            merged[key] += probability
        return [
            Transition(prob, ns, reward, terminated, event, realized)
            for (ns, reward, terminated, event, realized), prob in merged.items()
        ]

    def step(self, action: int | Action) -> tuple[State, float, bool, bool, dict[str, object]]:
        action = Action(int(action))
        draw = self.rng.random()
        if draw < 0.8:
            realized = action
        elif draw < 0.9:
            realized = PERPENDICULAR[action][0]
        else:
            realized = PERPENDICULAR[action][1]
        next_state, reward, terminated, event = self._deterministic_transition(self.state, realized)
        self.state = next_state
        self.steps += 1
        self.episode_return += reward
        truncated = bool(not terminated and self.steps >= self.max_steps)
        if truncated:
            event = "step_limit_reached"
        self.last_info = {
            "event": event,
            "chosen_action": int(action),
            "chosen_action_name": ACTION_NAMES[action],
            "realized_action": int(realized),
            "realized_action_name": ACTION_NAMES[realized],
            "has_key": bool(next_state[2]),
            "gate_open": self.is_gate_open(next_state[3]),
            "steps": self.steps,
            "episode_return": self.episode_return,
            "terminated": terminated,
            "truncated": truncated,
            "success": terminated,
        }
        return next_state, reward, terminated, truncated, dict(self.last_info)

    def states(self) -> list[State]:
        return [
            (r, c, has_key, phase)
            for r, c in self.valid_positions
            for has_key in (0, 1)
            for phase in range(self.gate_period)
        ]

    def reachable_states(self) -> set[State]:
        """Reachable support under all actions from all possible start phases."""
        seeds = {(self.start_pos[0], self.start_pos[1], 0, p) for p in range(self.gate_period)}
        q: deque[State] = deque(seeds)
        seen = set(seeds)
        while q:
            state = q.popleft()
            if self.is_terminal(state):
                continue
            for action in Action:
                for transition in self.transition_outcomes(state, action):
                    if transition.next_state not in seen:
                        seen.add(transition.next_state)
                        q.append(transition.next_state)
        return seen

    def local_signature(self, row: int, col: int, radius: int = 1) -> tuple[str, ...]:
        signature: list[str] = []
        for r in range(row - radius, row + radius + 1):
            for c in range(col - radius, col + radius + 1):
                if 0 <= r < self.size and 0 <= c < self.size:
                    signature.append(str(self.grid[r, c]))
                else:
                    signature.append("OUT")
        return tuple(signature)

    def render_ascii(self, state: State | None = None) -> str:
        state = self.state if state is None else state
        r_agent, c_agent, has_key, phase = state
        rows: list[str] = []
        for r in range(self.size):
            chars: list[str] = []
            for c in range(self.size):
                if (r, c) == (r_agent, c_agent):
                    chars.append("A")
                elif (r, c) == self.key_pos and has_key:
                    chars.append(NORMAL)
                elif (r, c) == self.gate_pos:
                    chars.append("g" if self.is_gate_open(phase) else "G")
                else:
                    chars.append(str(self.grid[r, c]))
            rows.append("".join(chars))
        return "\n".join(rows)
