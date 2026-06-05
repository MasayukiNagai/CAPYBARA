import sys
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from shared.metrics import *  # noqa: F401, F403
