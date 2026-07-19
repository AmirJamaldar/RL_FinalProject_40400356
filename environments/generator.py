"""Deterministic maze generation and validation utilities.

The project seed is the penultimate digit of the student ID.  The generated
map is saved as JSON and then reused by every algorithm so that comparisons
are fair and reproducible.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

WALL = "#"
NORMAL = "."
START = "S"
KEY = "K"
DOOR = "D"
GOAL = "T"
PENALTY = "P"
GATE = "G"

PASSABLE = {NORMAL, START, KEY, DOOR, GOAL, PENALTY, GATE}


@dataclass(frozen=True)
class MapMetadata:
    student_id: str
    base_seed: int
    size: int
    gate_period: int = 4
    gate_open_phases: tuple[int, ...] = (0, 1)


def project_seed(student_id: str) -> int:
    if not (student_id.isdigit() and len(student_id) >= 2):
        raise ValueError("student_id must contain at least two digits")
    return int(student_id[-2])


def project_size(student_id: str) -> int:
    return 15 + (project_seed(student_id) % 4)


def _carve_segment(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    """Carve a Manhattan segment and return its cells."""
    r, c = a
    tr, tc = b
    cells: set[tuple[int, int]] = set()
    step_r = 1 if tr >= r else -1
    while r != tr:
        grid[r, c] = NORMAL
        cells.add((r, c))
        r += step_r
    step_c = 1 if tc >= c else -1
    while c != tc:
        grid[r, c] = NORMAL
        cells.add((r, c))
        c += step_c
    grid[r, c] = NORMAL
    cells.add((r, c))
    return cells


def _find(grid: Sequence[Sequence[str]], symbol: str) -> tuple[int, int]:
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == symbol:
                return r, c
    raise ValueError(f"symbol {symbol!r} not found")


def bfs_path_exists(
    grid: Sequence[Sequence[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    has_key: bool,
) -> bool:
    """Static reachability check used only for map validation.

    The periodic gate is treated as passable because it opens recurrently.
    The door is passable only after the key has been collected.
    """
    n = len(grid)
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n and 0 <= nc < n) or (nr, nc) in seen:
                continue
            tile = grid[nr][nc]
            if tile == WALL or (tile == DOOR and not has_key):
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
    return False


def validate_map(grid: Sequence[Sequence[str]]) -> dict[str, object]:
    n = len(grid)
    if n < 15 or n > 18 or any(len(row) != n for row in grid):
        raise ValueError("map must be square and have size between 15 and 18")
    counts = {symbol: sum(row.count(symbol) for row in grid) for symbol in (START, KEY, DOOR, GOAL, GATE)}
    for symbol, count in counts.items():
        if count != 1:
            raise ValueError(f"map must contain exactly one {symbol}; found {count}")
    penalty_count = sum(row.count(PENALTY) for row in grid)
    if penalty_count < 5:
        raise ValueError("map must contain at least five penalty cells")
    wall_count = sum(row.count(WALL) for row in grid)
    if wall_count / (n * n) < 0.15:
        raise ValueError("fewer than 15% of cells are walls")

    start, key, goal = _find(grid, START), _find(grid, KEY), _find(grid, GOAL)
    if not bfs_path_exists(grid, start, key, has_key=False):
        raise ValueError("no valid path from start to key")
    if not bfs_path_exists(grid, key, goal, has_key=True):
        raise ValueError("no valid path from key to goal")

    return {
        "size": n,
        "wall_count": wall_count,
        "wall_ratio": wall_count / (n * n),
        "penalty_count": penalty_count,
        "start": start,
        "key": key,
        "goal": goal,
        "valid": True,
    }


def _protected_source_paths(grid: np.ndarray, n: int) -> set[tuple[int, int]]:
    start = (n - 2, 2)
    key = (2, 3)
    door = (10 if n >= 16 else 9, 7)
    gate = (7, 10)
    goal = (2, n - 3)
    detour_gap = (7, n - 2)

    protected: set[tuple[int, int]] = set()
    for a, b in (
        (start, (2, 2)),
        ((2, 2), key),
        (key, (2, 6)),
        ((2, 6), (door[0], 6)),
        ((door[0], 6), door),
        (door, (door[0], 10)),
        ((door[0], 10), gate),
        (gate, (2, 10)),
        ((2, 10), goal),
        ((door[0], 10), (door[0], n - 2)),
        ((door[0], n - 2), detour_gap),
        (detour_gap, (2, n - 2)),
        ((2, n - 2), goal),
    ):
        protected |= _carve_segment(grid, a, b)
    return protected


def generate_source_map(student_id: str = "40400356") -> dict[str, object]:
    """Generate the deterministic source map for the supplied student ID."""
    seed = project_seed(student_id)
    n = project_size(student_id)
    rng = np.random.default_rng(seed)

    grid = np.full((n, n), NORMAL, dtype="U1")
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    # Two structural barriers produce a genuine key-door-stage and a periodic
    # gate choice: waiting for the gate versus taking a longer detour.
    barrier_col = 7
    door = (10 if n >= 16 else 9, barrier_col)
    grid[1 : n - 1, barrier_col] = WALL
    grid[door] = DOOR

    barrier_row = 7
    gate = (barrier_row, 10)
    detour_gap = (barrier_row, n - 2)
    grid[barrier_row, 8 : n - 1] = WALL
    grid[gate] = GATE
    grid[detour_gap] = NORMAL

    protected = _protected_source_paths(grid, n)
    protected.update({door, gate, detour_gap})

    start = (n - 2, 2)
    key = (2, 3)
    goal = (2, n - 3)

    # Add seeded internal obstacles while preserving the guaranteed routes.
    candidates = [
        (r, c)
        for r in range(1, n - 1)
        for c in range(1, n - 1)
        if (r, c) not in protected and grid[r, c] == NORMAL
    ]
    rng.shuffle(candidates)
    target_internal_walls = max(28, int(0.16 * (n - 2) ** 2))
    added = 0
    for r, c in candidates:
        if added >= target_internal_walls:
            break
        # Avoid isolated one-cell pockets and keep map visually navigable.
        local_walls = sum(
            grid[r + dr, c + dc] == WALL for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if local_walls >= 2:
            continue
        grid[r, c] = WALL
        added += 1

    penalty_candidates = [
        (n - 3, 2),
        (n - 5, 4),
        (6, 3),
        (door[0], 9),
        (6, 11),
        (3, n - 4),
        (n - 4, n - 3),
    ]
    for pos in penalty_candidates:
        if grid[pos] == NORMAL and pos not in {start, key, door, gate, goal}:
            grid[pos] = PENALTY

    grid[start] = START
    grid[key] = KEY
    grid[door] = DOOR
    grid[gate] = GATE
    grid[goal] = GOAL

    rows = ["".join(row.tolist()) for row in grid]
    validation = validate_map(rows)
    metadata = MapMetadata(student_id=student_id, base_seed=seed, size=n)
    return {
        "metadata": {
            "student_id": metadata.student_id,
            "base_seed": metadata.base_seed,
            "size": metadata.size,
            "gate_period": metadata.gate_period,
            "gate_open_phases": list(metadata.gate_open_phases),
            "feature": "periodic_gate",
            "description": "The gate opens at phases 0 and 1 and closes at phases 2 and 3.",
        },
        "grid": rows,
        "validation": validation,
        "legend": {
            WALL: "wall",
            NORMAL: "normal",
            START: "start",
            KEY: "key",
            DOOR: "locked_door",
            GOAL: "goal",
            PENALTY: "penalty",
            GATE: "periodic_gate",
        },
    }


def _mutate_walls(
    source_rows: Sequence[str],
    rng: np.random.Generator,
    fraction: float,
    protected: set[tuple[int, int]],
) -> list[str]:
    grid = np.array([list(row) for row in source_rows], dtype="U1")
    n = len(grid)
    interior_walls = [
        (r, c)
        for r in range(1, n - 1)
        for c in range(1, n - 1)
        if grid[r, c] == WALL and (r, c) not in protected
    ]
    count = max(1, round(fraction * len(interior_walls)))
    rng.shuffle(interior_walls)
    selected = interior_walls[:count]
    for pos in selected:
        grid[pos] = NORMAL

    empty_candidates = [
        (r, c)
        for r in range(1, n - 1)
        for c in range(1, n - 1)
        if grid[r, c] == NORMAL and (r, c) not in protected
    ]
    rng.shuffle(empty_candidates)
    for pos in empty_candidates[:count]:
        grid[pos] = WALL
    return ["".join(row.tolist()) for row in grid]


def generate_target_map(source: dict[str, object], kind: str) -> dict[str, object]:
    """Create a similar or substantially different target map.

    Generation is deterministic and repeats until BFS validity is satisfied.
    """
    if kind not in {"similar", "different"}:
        raise ValueError("kind must be 'similar' or 'different'")
    rows = list(source["grid"])
    n = len(rows)
    source_seed = int(source["metadata"]["base_seed"])
    seed = source_seed + (101 if kind == "similar" else 202)
    start = _find(rows, START)
    old_key = _find(rows, KEY)
    old_goal = _find(rows, GOAL)
    door = _find(rows, DOOR)
    gate = _find(rows, GATE)

    # Structural barrier cells are protected so the semantic role of the door
    # and periodic gate remains intact in all environments.
    structural = {(r, 7) for r in range(1, n - 1)} | {(7, c) for c in range(8, n - 1)}
    protected = structural | {start, old_key, old_goal, door, gate}
    fraction = 0.18 if kind == "similar" else 0.42

    last_error: Exception | None = None
    for attempt in range(200):
        rng = np.random.default_rng(seed + attempt)
        mutated_rows = _mutate_walls(rows, rng, fraction, protected)
        grid = np.array([list(row) for row in mutated_rows], dtype="U1")

        # Restore unique special symbols before optional relocation.
        for r in range(n):
            for c in range(n):
                if grid[r, c] in {START, KEY, DOOR, GOAL, GATE}:
                    grid[r, c] = NORMAL
        grid[start] = START
        grid[door] = DOOR
        grid[gate] = GATE

        key, goal = old_key, old_goal
        if kind == "different":
            relocation_candidates = [
                (r, c)
                for r in range(1, n - 1)
                for c in range(1, n - 1)
                if grid[r, c] == NORMAL and c != 7 and r != 7
            ]
            rng.shuffle(relocation_candidates)
            # Keep the key on the start side of the locked door and the goal
            # beyond both structural barriers.
            key_candidates = [p for p in relocation_candidates if p[1] < 7 and p != start]
            goal_candidates = [p for p in relocation_candidates if p[1] > 7 and p[0] < 7]
            if not key_candidates or not goal_candidates:
                continue
            key = key_candidates[0]
            goal = goal_candidates[0]

        grid[key] = KEY
        grid[goal] = GOAL

        # Add new penalty cells in the different environment.
        if kind == "different":
            free = [
                (r, c)
                for r in range(1, n - 1)
                for c in range(1, n - 1)
                if grid[r, c] == NORMAL
            ]
            rng.shuffle(free)
            for pos in free[:4]:
                grid[pos] = PENALTY

        out_rows = ["".join(row.tolist()) for row in grid]
        try:
            validation = validate_map(out_rows)
        except Exception as exc:  # deterministic retry
            last_error = exc
            continue

        return {
            "metadata": {
                **source["metadata"],
                "environment": f"target_{kind}",
                "generation_seed": seed + attempt,
                "requested_obstacle_change_fraction": fraction,
                "key_relocated": key != old_key,
                "goal_relocated": goal != old_goal,
            },
            "grid": out_rows,
            "validation": validation,
            "legend": source["legend"],
        }
    raise RuntimeError(f"could not generate a valid {kind} target map: {last_error}")


def save_map(data: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_map(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generate_and_save_all(student_id: str, maps_dir: str | Path) -> dict[str, Path]:
    maps_dir = Path(maps_dir)
    source = generate_source_map(student_id)
    similar = generate_target_map(source, "similar")
    different = generate_target_map(source, "different")
    paths = {
        "source": maps_dir / "source_map.json",
        "similar": maps_dir / "target_similar_map.json",
        "different": maps_dir / "target_different_map.json",
    }
    save_map(source, paths["source"])
    save_map(similar, paths["similar"])
    save_map(different, paths["different"])
    return paths


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    generated = generate_and_save_all("40400356", root / "maps")
    for name, path in generated.items():
        print(f"{name}: {path}")
