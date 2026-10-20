"""Root-level package shim so `astropilote.*` imports work from repo root even without `PYTHONPATH=src` or `pip install -e .`."""

from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
_REAL_PKG = _SRC_DIR / "astropilote"

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

__path__ = [str(_REAL_PKG)]
__version__ = "1.0.0"
