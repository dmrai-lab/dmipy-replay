"""Check the installed packages against the replay move's parity reference (dmrai-lab/dmipy-sim#834), with the import
paths after the move:

    python tests/parity_move/check_reference.py REF_DIR [GROUP ...] [--require-all] [--keep DIR]
    python tests/parity_move/check_reference.py REF_DIR --against RECORD_DIR

Every group of :mod:`calls` runs in a fresh interpreter and is compared with the record value by value; a group whose
names are not yet where the move puts them is reported ``pending``. Exits 1 on any difference (and, with
``--require-all``, on a pending group). ``--against`` compares two records already taken, running nothing."""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import parity  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--against" in args:
        sys.exit(parity.compare_records(args[0], args[args.index("--against") + 1]))
    require_all = "--require-all" in args
    keep = None
    if "--keep" in args:
        i = args.index("--keep")
        keep = args[i + 1]
        del args[i:i + 2]
    args = [a for a in args if a != "--require-all"]
    sys.exit(parity.check(args[0], "after", args[1:] or None, require_all=require_all, keep=keep))
