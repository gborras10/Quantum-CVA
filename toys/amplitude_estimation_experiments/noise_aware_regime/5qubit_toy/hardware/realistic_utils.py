from __future__ import annotations

from pathlib import Path
import sys

TOY_DIR = Path(__file__).resolve().parents[1]
if str(TOY_DIR) not in sys.path:
    sys.path.insert(0, str(TOY_DIR))

from ae_pipeline_utils import *  # noqa: F401,F403
