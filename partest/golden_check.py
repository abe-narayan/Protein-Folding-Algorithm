"""Golden / determinism guardrail for the safe-speedup work (REC 1/2/4).

The invariant these recommendations must preserve is that the ENERGY FUNCTION is
byte-identical: for a known bitstring the OBC2-native ``AmberHamiltonian.energy``
must return exactly the value persisted in the seed-0 evaluation-set CSVs. REC 1/2/4
only change *who* calls ``energy`` and *when* (orchestration / thread pinning); they
must never move a number.

This module:
  * loads >=30 bitstring->energy pairs from each of the chignolin + trpzip golden CSVs,
  * rebuilds each protein's OBC2-native Hamiltonian,
  * recomputes the energies (optionally through an injected ``energy_fn`` so the REC 2
    process-pool path can be validated with the same reference), and
  * asserts max abs error == 0.0 on the finite pairs and matching +inf (collapse-floor)
    status on the floored pairs,
  * plus a bit-exact determinism check (recompute with a cleared cache).

Run directly:  python partest/golden_check.py
Exit code 0 == all guardrails hold (max abs error 0.0).
"""
import csv
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Golden CSVs: durable copies first, original prior-session scratchpads as fallback.
_GOLDEN_DIR = r"C:\Users\abena\PFA_backups\golden"
_CHIG_FALLBACK = (r"C:\Users\abena\AppData\Local\Temp\claude\C--Abhyuth-claude-protein-build"
                  r"\04a0f21c-acaa-469d-99b6-8775f3298c2d\scratchpad\seed0_full_eval_set.csv")
_TRP_FALLBACK = (r"C:\Users\abena\AppData\Local\Temp\claude\C--Abhyuth-claude-protein-build"
                 r"\84b6b99e-497b-494f-8516-ca4c535644c6\scratchpad\trpzip_seed0_full_eval_set.csv")


def _resolve(durable_name, fallback):
    p = os.path.join(_GOLDEN_DIR, durable_name)
    return p if os.path.exists(p) else fallback


PROTEINS = {
    "chignolin": {"seq": "GYDPETGTWG", "n_states": 8,
                  "csv": _resolve("seed0_full_eval_set.csv", _CHIG_FALLBACK)},
    "trpzip":    {"seq": "SWTWEGNKWTWK", "n_states": 8,
                  "csv": _resolve("trpzip_seed0_full_eval_set.csv", _TRP_FALLBACK)},
}


def load_golden(csv_path, n_finite=40, n_floored=8):
    """Deterministic sample of the CSV: first ``n_finite`` finite rows + first
    ``n_floored`` floored (+inf) rows. Returns list of (bitstring, energy_float)."""
    finite, floored = [], []
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            bs, raw = row[0], row[1]
            try:
                e = float(raw)
            except ValueError:
                e = float("inf")
            if math.isfinite(e):
                if len(finite) < n_finite:
                    finite.append((bs, e))
            else:
                if len(floored) < n_floored:
                    floored.append((bs, e))
            if len(finite) >= n_finite and len(floored) >= n_floored:
                break
    return finite + floored


def build_hamiltonian(seq, n_states):
    import representations as reps
    from amber_obc2 import build_obc2_native
    rep = reps.TorsionStateRepresentation(len(seq), n_states=n_states)
    return build_obc2_native(seq, rep)


def check_protein(name, energy_fn=None, verbose=True):
    """Recompute golden energies for one protein and compare bit-exactly.

    ``energy_fn`` maps a list of bitstrings -> dict{bitstring: energy}. Defaults to
    the serial ``H.energy`` path; REC 2 passes a pool-backed function here.
    Returns dict with max_abs_err (finite pairs), inf_mismatches, n_pairs.
    """
    cfg = PROTEINS[name]
    pairs = load_golden(cfg["csv"])
    H = build_hamiltonian(cfg["seq"], cfg["n_states"])
    bitstrings = [bs for bs, _ in pairs]
    if energy_fn is None:
        got = {bs: H.energy(bs) for bs in bitstrings}
    else:
        got = energy_fn(H, bitstrings)

    max_abs_err = 0.0
    worst_bs = None
    inf_mismatch = 0
    for bs, gold in pairs:
        e = got[bs]
        if math.isfinite(gold):
            if not math.isfinite(e):
                inf_mismatch += 1
                continue
            d = abs(e - gold)
            if d > max_abs_err:
                max_abs_err, worst_bs = d, bs
        else:  # floored -> must still be +inf
            if math.isfinite(e):
                inf_mismatch += 1
    ok = (max_abs_err == 0.0) and (inf_mismatch == 0)
    if verbose:
        print(f"  [{name}] pairs={len(pairs)} "
              f"(finite {sum(1 for _, g in pairs if math.isfinite(g))}, "
              f"floored {sum(1 for _, g in pairs if not math.isfinite(g))}) | "
              f"max_abs_err={max_abs_err!r} | inf_mismatch={inf_mismatch} | "
              f"{'PASS' if ok else 'FAIL'}"
              + (f"  worst@{worst_bs}" if worst_bs else ""))
    return {"protein": name, "n_pairs": len(pairs), "max_abs_err": max_abs_err,
            "inf_mismatch": inf_mismatch, "ok": ok, "H": H, "pairs": pairs}


def check_determinism(H, pairs, n=6):
    """Bit-exact recompute with a cleared cache: max spread must be exactly 0.0."""
    worst = 0.0
    for bs, _ in pairs[:n]:
        H._cache.pop(bs, None)
        e1 = H.energy(bs)
        H._cache.pop(bs, None)
        e2 = H.energy(bs)
        worst = max(worst, abs(e1 - e2) if math.isfinite(e1) and math.isfinite(e2)
                    else (0.0 if (math.isinf(e1) and math.isinf(e2)) else float("inf")))
    return worst


def main():
    print("=" * 72)
    print("GOLDEN + DETERMINISM GUARDRAIL (REC 1/2/4 invariant)")
    print("=" * 72)
    all_ok = True
    for name in ("chignolin", "trpzip"):
        res = check_protein(name)
        det = check_determinism(res["H"], res["pairs"])
        det_ok = (det == 0.0)
        print(f"  [{name}] determinism max spread = {det!r} | "
              f"{'PASS' if det_ok else 'FAIL'}")
        all_ok &= res["ok"] and det_ok
    print("-" * 72)
    print(f"  GUARDRAIL: {'ALL PASS (max abs error 0.0)' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
