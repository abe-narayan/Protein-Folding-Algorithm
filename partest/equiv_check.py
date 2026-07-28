"""REC 1/2 byte-identical equivalence: serial vs parallel driver on a small real OBC2
system. The parallelization touches no physics, so the selection must match bit-for-bit.

Uses a 6-residue peptide at n_states=4 (12 qubits) so the MPS sampler is fast and COBYLA
maxiter is small, while exercising the *identical* orchestration used at 30/36 qubits:
restart processes, shared-memory cache, best_run reduction, best-seen merge, final sample.

Compares vqe/modal/best-seen bitstrings + energies, final_objective, history, and the
distribution stats with exact equality. Any drift is a real difference.
"""
import multiprocessing as mp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SEQ = "AEKAAG"          # all supported residues; 6 res
N_STATES = 4            # -> 12 qubits
LAYERS = 1              # -> 12 params; COBYLA maxiter must exceed 14
SHOTS = 128
MAXITER = 18
RESTARTS = 3
FINAL_SHOTS = 256
SEED = 0
INIT_SCALE = 0.6


def _build_H():
    import representations as reps
    from amber_obc2 import build_obc2_native
    rep = reps.TorsionStateRepresentation(len(SEQ), n_states=N_STATES)
    return build_obc2_native(SEQ, rep)


def _run(restart_procs, minim_workers, use_shared_cache=True):
    from mps import MPSSampler, run_global_cvar_vqe
    from mps.parallel import OBC2Builder
    H = _build_H()
    sampler = MPSSampler(H.n_qubits, layers=LAYERS)
    builder = OBC2Builder(SEQ, N_STATES) if restart_procs > 1 else None
    return run_global_cvar_vqe(
        H, sampler=sampler, layers=LAYERS, alpha=0.15, shots=SHOTS,
        maxiter=MAXITER, restarts=RESTARTS, seed=SEED, final_shots=FINAL_SHOTS,
        init_scale=INIT_SCALE, verbose=False,
        hamiltonian_builder=builder, restart_procs=restart_procs,
        minim_workers=minim_workers, use_shared_cache=use_shared_cache)


def _cmp(name, ref, got):
    fields_str = ["vqe_bitstring", "vqe_modal_bitstring", "best_seen_bitstring"]
    fields_num = ["vqe_energy", "vqe_modal_energy", "best_seen_energy",
                  "final_objective", "distribution_top1_prob",
                  "distribution_top16_mass", "distribution_entropy_bits"]
    ok = True
    for f in fields_str:
        if ref[f] != got[f]:
            ok = False
            print(f"    [{name}] MISMATCH {f}: ref={ref[f]} got={got[f]}")
    worst = 0.0
    for f in fields_num:
        d = abs(float(ref[f]) - float(got[f]))
        worst = max(worst, d)
        if d != 0.0:
            ok = False
            print(f"    [{name}] MISMATCH {f}: ref={ref[f]!r} got={got[f]!r} (|d|={d!r})")
    rh, gh = ref["history"], got["history"]
    if len(rh) != len(gh):
        ok = False
        print(f"    [{name}] MISMATCH history length: {len(rh)} vs {len(gh)}")
    else:
        hw = max((abs(a - b) for a, b in zip(rh, gh)), default=0.0)
        worst = max(worst, hw)
        if hw != 0.0:
            ok = False
            print(f"    [{name}] MISMATCH history max |d|={hw!r}")
    print(f"    [{name}] {'PASS (bit-identical)' if ok else 'FAIL'} | "
          f"worst numeric |d|={worst!r} | vqe_bits={got['vqe_bitstring']} "
          f"vqe_E={got['vqe_energy']:.6f} best_seen_E={got['best_seen_energy']:.6f}")
    return ok


def main():
    print("=" * 72)
    print("REC 1/2 EQUIVALENCE — serial vs parallel driver (small real OBC2 system)")
    print("=" * 72)
    print("  running SERIAL reference ...", flush=True)
    ref = _run(restart_procs=1, minim_workers=0)
    print(f"    ref vqe_bits={ref['vqe_bitstring']} vqe_E={ref['vqe_energy']:.6f} "
          f"best_seen_E={ref['best_seen_energy']:.6f} evals={ref['n_objective_evals_total']}",
          flush=True)

    all_ok = True
    print("  running PARALLEL restarts (restart_procs=3, minim_workers=0) [REC 1] ...",
          flush=True)
    all_ok &= _cmp("REC1", ref, _run(restart_procs=3, minim_workers=0))

    print("  running PARALLEL restarts + minim pool (restart_procs=3, minim_workers=2) "
          "[REC 1+2] ...", flush=True)
    all_ok &= _cmp("REC1+2", ref, _run(restart_procs=3, minim_workers=2))

    print("  running PARALLEL, no shared cache (restart_procs=3, minim_workers=0, "
          "use_shared_cache=False) ...", flush=True)
    all_ok &= _cmp("REC1-nocache", ref,
                   _run(restart_procs=3, minim_workers=0, use_shared_cache=False))

    print("-" * 72)
    print(f"  EQUIVALENCE: {'ALL PASS (byte-identical selection)' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
