"""Pytest configuration for the sol-pinn test suite."""

from pathlib import Path
import sys


# Force tests to import the package from the current workspace copy first,
# even if another editable/installable clone exists elsewhere on the machine.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests that run training-heavy or long numerical workflows",
    )
