"""Project-root CLI launcher.

Allows `python main.py` from the repository root while keeping the real
implementation in `src/main.py`.
"""

from pathlib import Path
import runpy
import sys


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

runpy.run_path(str(SRC_DIR / "main.py"), run_name="__main__")