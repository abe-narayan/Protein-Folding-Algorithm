"""REC 2 verification: route the golden bitstrings through the MinimizationPool (workers
each own their own OBC2 Hamiltonian) and confirm 0 mismatches / max abs error 0.0 vs the
persisted seed-0 CSVs. This proves the process-pool minimization path is byte-identical.
"""
import multiprocessing as mp
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
for _p in (REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import golden_check as gc  # noqa: E402


def main():
    from mps.parallel import MinimizationPool, OBC2Builder
    all_ok = True
    for name in ("chignolin", "trpzip"):
        cfg = gc.PROTEINS[name]
        pool = MinimizationPool(OBC2Builder(cfg["seq"], cfg["n_states"]), n_workers=4)
        try:
            res = gc.check_protein(name, energy_fn=lambda H, bslist: pool.compute(bslist))
        finally:
            pool.close()
        all_ok &= res["ok"]
    print("-" * 72)
    print(f"  REC 2 POOL PATH: {'0 mismatches, max abs error 0.0' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
