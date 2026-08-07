"""Reproducible maze generation and BFS validation.

The generated source map contains a structural wall with a single locked door,
so collecting the key is not merely cosmetic.  The extra project mechanic is a
bidirectional teleporter pair (A/B).  Teleportation is part of the transition,
not an animation-only feature.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


WALL = "#"
FLOOR = "."
START = "S"
KEY = "K"
DOOR = "D"
GOAL = "G"
PENALTY = "X"
PORTAL_A = "A"
PORTAL_B = "B"

ACTION_DELTAS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
Position = tuple[int, int]
State = tuple[int, int, int]


@dataclass(frozen=True)
class MazeMap:
    """Immutable serializable map specification."""

    grid: tuple[str, ...]
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.grid:
            raise ValueError("Maze grid cannot be empty")
        width = len(self.grid[0])
        if width == 0 or any(len(row) != width for row in self.grid):
            raise ValueError("Maze grid must be non-empty and rectangular")

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0])

    @property
    def size(self) -> int:
        if self.height != self.width:
            raise ValueError("Project maze must be square")
        return self.height

    def tile(self, position: Position) -> str:
        row, col = position
        return self.grid[row][col]

    def positions(self, tile: str) -> list[Position]:
        return [
            (row, col)
            for row, line in enumerate(self.grid)
            for col, value in enumerate(line)
            if value == tile
        ]

    def unique_position(self, tile: str) -> Position:
        matches = self.positions(tile)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {tile!r} tile, found {len(matches)}")
        return matches[0]

    @property
    def start(self) -> Position:
        return self.unique_position(START)

    @property
    def key(self) -> Position:
        return self.unique_position(KEY)

    @property
    def door(self) -> Position:
        return self.unique_position(DOOR)

    @property
    def goal(self) -> Position:
        return self.unique_position(GOAL)

    @property
    def portals(self) -> dict[Position, Position]:
        a = self.unique_position(PORTAL_A)
        b = self.unique_position(PORTAL_B)
        return {a: b, b: a}

    @property
    def wall_positions(self) -> set[Position]:
        return set(self.positions(WALL))

    @property
    def walkable_count(self) -> int:
        return self.height * self.width - len(self.wall_positions)

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "grid": list(self.grid),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(destination)
        return destination


def load_map(path: str | Path) -> MazeMap:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("format_version") != 1:
        raise ValueError(f"Unsupported map format: {data.get('format_version')!r}")
    maze = MazeMap(tuple(data["grid"]), dict(data.get("metadata", {})))
    validate_map(maze)
    return maze


def maze_fingerprint(maze: MazeMap) -> str:
    """Stable semantic hash independent of JSON whitespace or file location."""

    payload = json.dumps(
        maze.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _line_path(start: Position, end: Position) -> list[Position]:
    """A deterministic row-first Manhattan path, including both endpoints."""

    row, col = start
    result = [(row, col)]
    while row != end[0]:
        row += 1 if end[0] > row else -1
        result.append((row, col))
    while col != end[1]:
        col += 1 if end[1] > col else -1
        result.append((row, col))
    return result


def geometry_step(maze: MazeMap, state: State, action: int) -> State:
    """Apply one deterministic intended action without rewards or episode limits."""

    if action not in range(4):
        raise ValueError(f"Invalid action {action}; expected 0..3")
    row, col, has_key_int = state
    has_key = bool(has_key_int)
    dr, dc = ACTION_DELTAS[action]
    candidate = (row + dr, col + dc)
    if not (0 <= candidate[0] < maze.height and 0 <= candidate[1] < maze.width):
        return state
    tile = maze.tile(candidate)
    if tile == WALL or (tile == DOOR and not has_key):
        return state
    if tile in (PORTAL_A, PORTAL_B):
        candidate = maze.portals[candidate]
        tile = maze.tile(candidate)
    next_has_key = has_key or tile == KEY
    return candidate[0], candidate[1], int(next_has_key)


def bfs_state_path(
    maze: MazeMap,
    initial_state: State,
    goal_test: Callable[[State], bool],
) -> list[State] | None:
    """Return a shortest deterministic state path, or ``None`` if unreachable."""

    queue: deque[State] = deque([initial_state])
    parent: dict[State, State | None] = {initial_state: None}
    while queue:
        state = queue.popleft()
        if goal_test(state):
            path: list[State] = []
            cursor: State | None = state
            while cursor is not None:
                path.append(cursor)
                cursor = parent[cursor]
            return list(reversed(path))
        for action in range(4):
            successor = geometry_step(maze, state, action)
            if successor not in parent:
                parent[successor] = state
                queue.append(successor)
    return None


def mission_path(maze: MazeMap, initial_state: State | None = None) -> list[State] | None:
    start = initial_state or (maze.start[0], maze.start[1], 0)
    return bfs_state_path(
        maze,
        start,
        lambda state: (state[0], state[1]) == maze.goal and state[2] == 1,
    )


def shortest_mission_distance(maze: MazeMap, state: State) -> int | None:
    path = mission_path(maze, state)
    return None if path is None else len(path) - 1


def _as_grid(rows: Sequence[Sequence[str]]) -> tuple[str, ...]:
    return tuple("".join(row) for row in rows)


def generate_source_map(student_id: str = "40400356") -> MazeMap:
    """Generate the student's deterministic 16x16 source maze."""

    if len(student_id) < 2 or not student_id.isdigit():
        raise ValueError("student_id must contain at least two decimal digits")
    base_seed = int(student_id[-2])
    size = 15 + (base_seed % 4)
    rng = np.random.default_rng(base_seed)

    grid = [[WALL for _ in range(size)] for _ in range(size)]
    for row in range(1, size - 1):
        for col in range(1, size - 1):
            grid[row][col] = FLOOR

    start = (1, 1)
    key = (size - 3, 3)
    barrier_col = size // 2
    door = (size - 3, barrier_col)
    goal = (size - 2, size - 2)
    portal_a = (2, 3)
    portal_b = (size - 5, barrier_col - 2)

    # A full-height internal barrier makes the locked door operationally required.
    structural_walls: set[Position] = set()
    for row in range(1, size - 1):
        if (row, barrier_col) != door:
            grid[row][barrier_col] = WALL
            structural_walls.add((row, barrier_col))
    structural_walls.update(
        (row, col)
        for row in range(size)
        for col in range(size)
        if row in (0, size - 1) or col in (0, size - 1)
    )

    protected: set[Position] = {start, key, door, goal, portal_a, portal_b}
    protected.update(_line_path(start, (key[0], start[1])))
    protected.update(_line_path((key[0], start[1]), key))
    protected.update(_line_path(key, door))
    protected.update(_line_path(door, (door[0], goal[1])))
    protected.update(_line_path((door[0], goal[1]), goal))
    # Keep access to both teleporter endpoints open.
    protected.update(_line_path(start, portal_a))
    protected.update(_line_path(portal_b, key))

    candidates = [
        (row, col)
        for row in range(1, size - 1)
        for col in range(1, size - 1)
        if grid[row][col] == FLOOR and (row, col) not in protected
    ]
    rng.shuffle(candidates)
    # Keep enough interior obstacles mutable that transfer maps can relocate the
    # required fraction of *all* source obstacles, even under the strictest
    # interpretation that counts immutable boundary walls in the denominator.
    target_mutable_walls = min(56, len(candidates))
    if target_mutable_walls < 56:
        raise RuntimeError("Map layout does not provide enough mutable obstacle candidates")
    mutable_walls = set(candidates[:target_mutable_walls])
    for row, col in mutable_walls:
        grid[row][col] = WALL

    grid[start[0]][start[1]] = START
    grid[key[0]][key[1]] = KEY
    grid[door[0]][door[1]] = DOOR
    grid[goal[0]][goal[1]] = GOAL
    grid[portal_a[0]][portal_a[1]] = PORTAL_A
    grid[portal_b[0]][portal_b[1]] = PORTAL_B

    penalty_candidates = [
        (row, col)
        for row in range(1, size - 1)
        for col in range(1, size - 1)
        if grid[row][col] == FLOOR
    ]
    rng.shuffle(penalty_candidates)
    penalties = penalty_candidates[:7]
    for row, col in penalties:
        grid[row][col] = PENALTY

    metadata = {
        "student_id": student_id,
        "base_seed": base_seed,
        "maze_size": size,
        "map_kind": "source",
        "feature": "bidirectional_teleporter",
        "slip_probabilities": [0.8, 0.1, 0.1],
        "barrier_column": barrier_col,
        "mutable_walls": [list(position) for position in sorted(mutable_walls)],
        "protected_mission_positions": [list(position) for position in sorted(protected)],
        "structural_wall_count": len(structural_walls),
        "generator_version": 1,
    }
    maze = MazeMap(_as_grid(grid), metadata)
    validate_map(maze)
    return maze


def _replace_tile(grid: list[list[str]], old: str, new: str) -> None:
    matches = [
        (row, col)
        for row, line in enumerate(grid)
        for col, value in enumerate(line)
        if value == old
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {old!r} while mutating map")
    row, col = matches[0]
    grid[row][col] = new


def generate_transfer_map(
    source: MazeMap,
    kind: str,
    seed: int | None = None,
) -> MazeMap:
    """Create a BFS-valid similar or different target map.

    Boundary and locked-door barrier walls remain structural, but the requested
    relocation fraction is conservatively measured against *all* source walls.
    """

    if kind not in {"similar", "different"}:
        raise ValueError("kind must be 'similar' or 'different'")
    source_mutable = {tuple(item) for item in source.metadata["mutable_walls"]}
    fraction = 0.175 if kind == "similar" else 0.40
    source_wall_count = len(source.wall_positions)
    move_count = max(1, math.ceil(fraction * source_wall_count))
    if move_count > len(source_mutable):
        raise RuntimeError(
            f"Need {move_count} mutable walls for {kind} transfer, only {len(source_mutable)} available"
        )
    effective_seed = int(seed if seed is not None else source.metadata["base_seed"] + (101 if kind == "similar" else 202))
    rng = np.random.default_rng(effective_seed)
    barrier_col = int(source.metadata["barrier_column"])
    protected_mission = {tuple(item) for item in source.metadata["protected_mission_positions"]}

    for attempt in range(1, 501):
        grid = [list(row) for row in source.grid]
        removed_array = rng.choice(sorted(source_mutable), size=move_count, replace=False)
        removed = {tuple(position) for position in removed_array.tolist()}
        for row, col in removed:
            grid[row][col] = FLOOR

        special_positions = {
            source.start,
            source.key,
            source.door,
            source.goal,
            *source.portals.keys(),
        }
        destinations = [
            (row, col)
            for row in range(1, source.height - 1)
            for col in range(1, source.width - 1)
            if grid[row][col] == FLOOR
            and (row, col) not in special_positions
            and col != barrier_col
            and (row, col) not in source_mutable
            and (row, col) not in protected_mission
        ]
        if len(destinations) < move_count:
            raise RuntimeError("Not enough free cells to relocate obstacles")
        added_array = rng.choice(destinations, size=move_count, replace=False)
        added = {tuple(position) for position in added_array.tolist()}
        for row, col in added:
            grid[row][col] = WALL

        relocated_key: Position | None = None
        added_penalties: list[Position] = []
        if kind == "different":
            _replace_tile(grid, KEY, FLOOR)
            key_candidates = [
                (row, col)
                for row, col in sorted(protected_mission)
                if 1 <= row < source.height - 1
                and 1 <= col < barrier_col
                and grid[row][col] == FLOOR
                and (row, col) != source.key
            ]
            if not key_candidates:
                continue
            relocated_key = tuple(key_candidates[int(rng.integers(len(key_candidates)))])
            grid[relocated_key[0]][relocated_key[1]] = KEY

            penalty_candidates = [
                (row, col)
                for row in range(1, source.height - 1)
                for col in range(1, source.width - 1)
                if grid[row][col] == FLOOR
            ]
            if len(penalty_candidates) < 3:
                continue
            chosen = rng.choice(penalty_candidates, size=3, replace=False)
            added_penalties = [tuple(position) for position in chosen.tolist()]
            for row, col in added_penalties:
                grid[row][col] = PENALTY

        new_mutable = (source_mutable - removed) | added
        metadata = dict(source.metadata)
        metadata.update(
            {
                "map_kind": kind,
                "parent_map_kind": source.metadata.get("map_kind", "source"),
                "mutation_seed": effective_seed,
                "mutation_attempt": attempt,
                "requested_mutable_obstacle_fraction": fraction,
                "moved_mutable_obstacle_count": move_count,
                "mutable_obstacle_count": len(source_mutable),
                "actual_mutable_obstacle_fraction": move_count / len(source_mutable),
                "source_total_obstacle_count": source_wall_count,
                "actual_all_obstacle_relocation_fraction": move_count / source_wall_count,
                "mutable_walls": [list(position) for position in sorted(new_mutable)],
                "removed_walls": [list(position) for position in sorted(removed)],
                "added_walls": [list(position) for position in sorted(added)],
                "relocated_key": list(relocated_key) if relocated_key else None,
                "new_penalties": [list(position) for position in sorted(added_penalties)],
            }
        )
        candidate = MazeMap(_as_grid(grid), metadata)
        try:
            validate_map(candidate)
        except ValueError:
            continue
        return candidate
    raise RuntimeError(f"Unable to construct a valid {kind} target map after 500 attempts")


def validate_map(maze: MazeMap) -> dict:
    """Validate all hard map requirements and return measurable evidence."""

    if not 15 <= maze.size <= 18:
        raise ValueError(f"Maze size {maze.size} is outside the required [15, 18]")
    allowed = {WALL, FLOOR, START, KEY, DOOR, GOAL, PENALTY, PORTAL_A, PORTAL_B}
    unknown = set("".join(maze.grid)) - allowed
    if unknown:
        raise ValueError(f"Unknown map tiles: {sorted(unknown)}")
    for required in (START, KEY, DOOR, GOAL, PORTAL_A, PORTAL_B):
        maze.unique_position(required)
    wall_ratio = len(maze.wall_positions) / (maze.height * maze.width)
    if wall_ratio < 0.15:
        raise ValueError(f"Wall ratio {wall_ratio:.3f} is below 0.15")
    penalty_count = len(maze.positions(PENALTY))
    if penalty_count < 5:
        raise ValueError(f"Only {penalty_count} penalty cells; at least five required")
    path = mission_path(maze)
    if path is None:
        raise ValueError("BFS found no valid start -> key -> door -> goal mission path")
    positions = [(row, col) for row, col, _ in path]
    if maze.key not in positions:
        raise ValueError("BFS mission path reaches goal without collecting key")
    if maze.door not in positions:
        raise ValueError("BFS mission path does not pass through the locked door")
    if positions.index(maze.key) > positions.index(maze.door):
        raise ValueError("BFS mission path passes door before collecting key")
    return {
        "size": maze.size,
        "wall_count": len(maze.wall_positions),
        "wall_ratio": wall_ratio,
        "penalty_count": penalty_count,
        "walkable_count": maze.walkable_count,
        "mission_path_length": len(path) - 1,
        "mission_path": [list(state) for state in path],
    }


def local_signature(maze: MazeMap, position: Position, radius: int = 1) -> tuple[str, ...]:
    """Semantic local neighborhood used by selective Q transfer."""

    row, col = position
    values: list[str] = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr, cc = row + dr, col + dc
            values.append(maze.grid[rr][cc] if 0 <= rr < maze.height and 0 <= cc < maze.width else WALL)
    return tuple(values)
