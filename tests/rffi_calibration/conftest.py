import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / 'src', ROOT / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_by_path(name, relpath):
    """Load a historical script by file path (it is not a package member and its own imports
    resolve machine paths at import time; the objects we need from it are self-contained)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# historical P2E reference implementation of the 'proto' rule (EigenRX + m/(m+2))
p2e_eigenrx = load_by_path('p2e_eigenrx_direct', 'scripts/receiver_p2/p2e_eigenrx.py')
