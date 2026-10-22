"""Copy this file over policy.py in the organizers' client folder (PhysicalAI-Hackathon-2026/).

It loads TwinCheck from this repo: set TWINCHECK_REPO to the repo path if the client folder is not
next to it.
"""
import os
import sys
from pathlib import Path

_repo = Path(os.environ.get("TWINCHECK_REPO", Path(__file__).resolve().parent.parent / "twincheck"))
sys.path.insert(0, str(_repo / "cell_policy"))

from twincheck_policy import Policy  # noqa: E402

__all__ = ["Policy"]
