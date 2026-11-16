"""Pytest root configuration.

Ensures the project root is on ``sys.path`` so tests can import ``src``
and ``core`` packages regardless of the working directory pytest is invoked
from.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
