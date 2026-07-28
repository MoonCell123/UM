"""Command-line entry point for GME middle-fusion training.

The implementation lives in :mod:`train.gme` so this file remains a stable
entry point for existing scripts and configuration files.
"""

from __future__ import annotations

import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train.gme.experiment import main


if __name__ == "__main__":
    main()
