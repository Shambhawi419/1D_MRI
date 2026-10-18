"""
Makes ../odls_dynamic importable from this folder.

odls_eval_plus only REUSES odls_dynamic (data loading, masks, model loading,
the original metrics); it never modifies anything there. Every script in this
folder starts with `import paths` so that lines like `from data import ...`
find odls_dynamic's modules.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DYN_DIR = os.path.join(os.path.dirname(HERE), "odls_dynamic")

if not os.path.isdir(DYN_DIR):
    raise ImportError(f"expected the odls_dynamic folder next to odls_eval_plus, at {DYN_DIR}")
if DYN_DIR not in sys.path:
    sys.path.insert(0, DYN_DIR)
