"""I/O helpers for solutions."""

import csv
from pathlib import Path

import numpy as np


def save_solution(filepath, s, T, metadata=None):
    """Save a solution as a compressed ``.npz`` file."""
    data = {"s": s, "T": T}
    if metadata:
        data.update(metadata)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath, **data)


def save_table_csv(filepath, rows, fieldnames):
    """Save a list of flat dictionaries to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

