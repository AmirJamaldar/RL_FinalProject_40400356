"""Tkinter canvas renderer for the dynamic maze."""
from __future__ import annotations

import tkinter as tk
from typing import Mapping

import numpy as np

from environments.maze import ACTION_NAMES, DynamicMazeEnv, State


TILE_COLORS = {
    "#": "#263238",
    ".": "#FAFAFA",
    "S": "#90CAF9",
    "K": "#FFD54F",
    "D": "#8D6E63",
    "T": "#66BB6A",
    "P": "#EF9A9A",
    "G": "#AB47BC",
}
ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


class MazeRenderer:
    def __init__(self, canvas: tk.Canvas, env: DynamicMazeEnv, cell_size: int = 34) -> None:
        self.canvas = canvas
        self.env = env
        self.cell_size = cell_size
        self.canvas.configure(width=env.size * cell_size, height=env.size * cell_size)

    def draw(self, state: State, policy: np.ndarray | None = None, show_policy: bool = False, message: str = "") -> None:
        self.canvas.delete("all")
        r_agent, c_agent, has_key, phase = state
        for r in range(self.env.size):
            for c in range(self.env.size):
                tile = str(self.env.grid[r, c])
                if tile == "K" and has_key:
                    tile = "."
                color = TILE_COLORS.get(tile, "white")
                if tile == "G":
                    color = "#CE93D8" if self.env.is_gate_open(phase) else "#6A1B9A"
                x0, y0 = c * self.cell_size, r * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#B0BEC5")
                label = {"S": "S", "K": "K", "D": "D", "T": "T", "P": "!", "G": "G"}.get(tile, "")
                if label:
                    self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=label, font=("Arial", 11, "bold"))
                if show_policy and policy is not None and tile not in {"#", "T"}:
                    action = int(policy[r, c, has_key, phase])
                    if action in ARROWS:
                        self.canvas.create_text(x0 + self.cell_size * 0.78, y0 + self.cell_size * 0.25, text=ARROWS[action], font=("Arial", 8))
        x0, y0 = c_agent * self.cell_size, r_agent * self.cell_size
        x1, y1 = x0 + self.cell_size, y0 + self.cell_size
        self.canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="#1565C0", outline="white", width=2)
        self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text="A", fill="white", font=("Arial", 10, "bold"))
        if message:
            self.canvas.create_rectangle(0, 0, self.env.size * self.cell_size, 28, fill="#FFFFFF", stipple="gray50", outline="")
            self.canvas.create_text(8, 14, anchor="w", text=message, font=("Arial", 10, "bold"))
