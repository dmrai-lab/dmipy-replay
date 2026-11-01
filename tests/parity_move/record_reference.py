"""Record the replay move's parity reference (dmrai-lab/dmipy-sim#834) with the import paths before the move:

    python tests/parity_move/record_reference.py OUT_DIR [GROUP ...]

run on the commits of dmipy-sim and dmipy-replay before the move (``reference/manifest.json`` names them). Each group
of :mod:`calls` runs in a fresh interpreter in the environment :data:`parity.ENV` fixes."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import parity  # noqa: E402

if __name__ == "__main__":
    parity.record(sys.argv[1], "before", sys.argv[2:] or None)
