"""Bootstrap helpers for standalone experiment scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path(current_file: str) -> Path:
    """Ensure the repository root is importable when a script is run directly."""
    repo_root = Path(current_file).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root
