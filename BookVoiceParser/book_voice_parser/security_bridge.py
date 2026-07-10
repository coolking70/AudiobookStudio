"""Import the repository security policy from package and CLI entry points."""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from security import validate_remote_url
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from security import validate_remote_url

__all__ = ["validate_remote_url"]
