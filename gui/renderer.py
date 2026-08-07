"""Canvas renderer with visually distinct maze semantics."""

from __future__ import annotations

import tkinter as tk

import numpy as np

from environments.generator import DOOR, FLOOR, GOAL, KEY, PENALTY, PORTAL_A, PORTAL_B, START, WALL, MazeMap


COLORS = {
    FLOOR: "#f7f7f7",
    WALL: "#252525",
    PENALTY: "#e76f51",
    START: "#8ecae6",
    KEY: "#f4d35e",
    DOOR: "#9c6644",
    GOAL: "#2a9d8f",
    PORTAL_A: "#9b5de5",
    PORTAL_B: "#6a4c93",
}
ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


class MazeRenderer:
    def __init__(self, canvas: tk.Canvas, *, cell_size: int = 34) -> None:
        self.canvas = canvas
        self.cell_size = cell_size

    def configure_for(self, maze: MazeMap) -> None:
        self.canvas.configure(width=maze.width * self.cell_size, height=maze.height * self.cell_size)

    def render(
        self,
        maze: MazeMap,
        state: tuple[int, int, int],
        *,
        policy: np.ndarray | None = None,
        show_policy: bool = False,
        last_events: list[str] | None = None,
    ) -> None:
        self.configure_for(maze)
        self.canvas.delete("all")
        has_key = bool(state[2])
        size = self.cell_size
        for row, line in enumerate(maze.grid):
            for col, tile in enumerate(line):
                x0, y0 = col * size, row * size
                x1, y1 = x0 + size, y0 + size
                fill = COLORS[tile]
                if tile == DOOR and has_key:
                    fill = "#90be6d"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#b8b8b8", width=1)
                if tile == PENALTY:
                    self.canvas.create_line(x0 + 8, y0 + 8, x1 - 8, y1 - 8, fill="white", width=3)
                    self.canvas.create_line(x1 - 8, y0 + 8, x0 + 8, y1 - 8, fill="white", width=3)
                elif tile == KEY and not has_key:
                    self.canvas.create_oval(x0 + 7, y0 + 8, x0 + 19, y0 + 20, outline="#5f4b00", width=3)
                    self.canvas.create_line(x0 + 17, y0 + 18, x1 - 6, y1 - 7, fill="#5f4b00", width=3)
                elif tile == DOOR:
                    self.canvas.create_rectangle(x0 + 8, y0 + 4, x1 - 8, y1 - 3, outline="white", width=2)
                    self.canvas.create_oval(x1 - 13, y0 + size / 2, x1 - 9, y0 + size / 2 + 4, fill="white", outline="")
                elif tile == GOAL:
                    self.canvas.create_text(x0 + size / 2, y0 + size / 2, text="★", fill="white", font=("TkDefaultFont", 18, "bold"))
                elif tile in (PORTAL_A, PORTAL_B):
                    self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, outline="white", width=3)
                    self.canvas.create_text(x0 + size / 2, y0 + size / 2, text=tile, fill="white", font=("TkDefaultFont", 10, "bold"))
                elif tile == START:
                    self.canvas.create_text(x0 + size / 2, y0 + size / 2, text="S", fill="#023047", font=("TkDefaultFont", 11, "bold"))

        if show_policy and policy is not None:
            layer = int(state[2])
            for row in range(maze.height):
                for col in range(maze.width):
                    action = int(policy[layer, row, col])
                    if action >= 0:
                        self.canvas.create_text(
                            col * size + size / 2,
                            row * size + size / 2,
                            text=ARROWS[action],
                            fill="#0b132b",
                            font=("TkDefaultFont", max(8, size // 3), "bold"),
                        )

        row, col, _ = state
        margin = 5
        outline = "#ffd60a" if last_events and "key_collected" in last_events else "white"
        self.canvas.create_oval(
            col * size + margin,
            row * size + margin,
            (col + 1) * size - margin,
            (row + 1) * size - margin,
            fill="#d00000",
            outline=outline,
            width=3,
        )
