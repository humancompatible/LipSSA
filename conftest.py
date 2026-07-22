"""pytest bootstrap: put the repository root on sys.path so `import benchmarks`
works without installing the package."""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
