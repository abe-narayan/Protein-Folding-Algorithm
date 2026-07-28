"""Combined end-to-end measurement + byte-identical cross-check (REC 1/2/4).

Runs the seed-0 config-A CVaR-VQE for a protein with the opt-in parallel path and times
the real wall clock, then confirms the selected/best-seen result is byte-identical to the
persisted serial (golden) run. Reports the measured speedup vs the persisted baseline.

Usage:  python partest/measure.py <chignolin|trpzip> [restart_procs] [minim_workers]
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# config A (the exact seed-0 construction)
CFGA = dict(layers=4, alpha=0.15, shots=1024, maxiter=200, restarts=3, seed=0,
            final_shots=8192, init_scale=0.25)

PROT = {
    "chignolin": {
        "seq": "GYDPETGTWG", "n_states": 8, "pdb": "1UAO",
        "baseline_s": 2.3621021554867427 * 3600.0,   # seed0_logged_meta.json
        "sel_bits": "001000010000000000000000001000",
        "sel_E": -368.0638953527846,
        "bs_bits": "100000011011111111100000000101",
        "bs_E": -396.4426263461881,
    },
    "trpzip": {
        "seq": "SWTWEGNKWTWK", "n_states": 8, "pdb": "1LE0",
        "baseline_s": 15231.18548321724,               # trpzip_seed0_summary.json
        "sel_bits": "001010000000000000000000001000000000",
        "sel_E": -395.792567917652,
        "bs_bits": "011011100000000000000000001000000000",
        "bs_E": -398.80968070881033,
    },
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "chignolin"
    rprocs = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("PFA_RESTART_PROCS", "3"))
    mworkers = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("PFA_MINIM_WORKERS", "2"))
    p = PROT[which]

    import amber_hamiltonian as amb
    import representations as reps
    import protein_geometry as geo
    from amber_obc2 import build_obc2_native
    from mps import MPSSampler, run_global_cvar_vqe
    from mps.parallel import OBC2Builder

    rep = reps.TorsionStateRepresentation(len(p["seq"]), n_states=p["n_states"])
    H = build_obc2_native(p["seq"], rep)
    _, nc, nphi, npsi = geo.native_coords_from_pdb(
        os.path.join(os.path.dirname(amb.__file__), "pdbs", f"{p['pdb']}.pdb"))
    nat_ca = np.asarray(nc["CA"])
    rmsd = lambda b: geo.ca_rmsd(rep.build_coords(b)["CA"], nat_ca)

    sampler = MPSSampler(H.n_qubits, layers=CFGA["layers"])
    builder = OBC2Builder(p["seq"], p["n_states"]) if rprocs > 1 else None
    print(f"[{which}] n_qubits={H.n_qubits} config A | restart_procs={rprocs} "
          f"minim_workers={mworkers} | baseline {p['baseline_s']:.1f}s "
          f"({p['baseline_s']/3600:.3f}h)", flush=True)

    t0 = time.time()
    res = run_global_cvar_vqe(H, sampler=sampler, verbose=True,
                              hamiltonian_builder=builder, restart_procs=rprocs,
                              minim_workers=mworkers, **CFGA)
    wall = time.time() - t0

    sel_bits, sel_E = res["vqe_bitstring"], res["vqe_energy"]
    bs_bits, bs_E = res["best_seen_bitstring"], res["best_seen_energy"]
    sel_rmsd = rmsd(sel_bits)
    # byte-identical cross-check vs persisted serial (golden) run
    checks = {
        "sel_bits_match": sel_bits == p["sel_bits"],
        "sel_E_abs_err": abs(sel_E - p["sel_E"]),
        "bs_bits_match": bs_bits == p["bs_bits"],
        "bs_E_abs_err": abs(bs_E - p["bs_E"]),
    }
    byte_identical = (checks["sel_bits_match"] and checks["bs_bits_match"]
                      and checks["sel_E_abs_err"] == 0.0 and checks["bs_E_abs_err"] == 0.0)
    speedup = p["baseline_s"] / wall

    summary = {
        "protein": which, "n_qubits": H.n_qubits, "restart_procs": rprocs,
        "minim_workers": mworkers, "wall_s": wall, "wall_h": wall / 3600.0,
        "baseline_s": p["baseline_s"], "baseline_h": p["baseline_s"] / 3600.0,
        "speedup": speedup,
        "selected_bitstring": sel_bits, "selected_energy": sel_E,
        "selected_ca_rmsd": sel_rmsd,
        "best_seen_bitstring": bs_bits, "best_seen_energy": bs_E,
        "n_objective_evals_total": res["n_objective_evals_total"],
        "n_energy_evaluations": res["n_energy_evaluations"],
        "n_unique_structures_cached": res["n_unique_structures_cached"],
        "checks": checks, "byte_identical_selection": byte_identical,
        "audit_clean": bool((not res["audit"]["final_selected_is_collapse"])
                            and res["audit"]["n_evals_lowtail_collapse_best_run"] == 0),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"measure_{which}_r{rprocs}_m{mworkers}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(f"[{which}] WALL {wall:.1f}s ({wall/3600:.3f}h)  vs baseline "
          f"{p['baseline_s']:.1f}s ({p['baseline_s']/3600:.3f}h)  -> SPEEDUP {speedup:.2f}x")
    print(f"  selected : {sel_bits} E={sel_E:.6f} RMSD={sel_rmsd:.4f}  "
          f"(golden {p['sel_bits']} E={p['sel_E']:.6f})")
    print(f"  best_seen: {bs_bits} E={bs_E:.6f}  (golden {p['bs_bits']} E={p['bs_E']:.6f})")
    print(f"  BYTE-IDENTICAL SELECTION: {byte_identical}  "
          f"(sel|dE|={checks['sel_E_abs_err']!r}, bs|dE|={checks['bs_E_abs_err']!r})")
    print(f"  wrote {out}")
    return 0 if byte_identical else 2


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
