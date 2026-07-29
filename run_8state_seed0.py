"""8-state OBC2-native chignolin seed-0 run.

PINNED. This script reproduces a persisted reference dataset (`IMPLEMENTATION_LOG.md`,
golden CSVs, verified by `partest/golden_check.py` to max absolute error 0.0), so two
things here are deliberately *not* modernised:

* ``legacy_library=True`` keeps the pre-2026-07-28 flat 8-state torsion library. The new
  per-residue-class libraries in `representations` change the (phi, psi) values a
  bitstring decodes to -- chignolin is 3 Gly and 1 Pro out of 10 residues, all of which
  get different states now -- and therefore every energy in the golden set.
* ``optimizer_guard="warn"`` keeps ``maxiter=200`` at 120 parameters, which
  `budget.check_optimizer_budget` correctly rejects for new work (it is below 3x
  n_params, so most of the allowance goes into constructing a COBYLA simplex). Raising it
  would change the trajectory and invalidate the reference.

Neither exemption should be copied into a new experiment. To re-run this configuration
*properly*, drop both flags, accept that the golden data no longer applies, and
regenerate it.
"""
import os


for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import multiprocessing as mp

import amber_hamiltonian as amb
import representations as reps
import protein_geometry as geo
from amber_obc2 import build_obc2_native
from mps import MPSSampler, run_global_cvar_vqe
from mps.parallel import OBC2Builder

SEQ = "GYDPETGTWG"

# Opt-in REC 1/2 parallelism (default OFF -> serial, byte-identical to the persisted
# seed-0 run). Set PFA_RESTART_PROCS>1 to run the 3 restarts as processes and
# PFA_MINIM_WORKERS>1 to add a per-restart minimization pool. The result is unchanged.
RESTART_PROCS = int(os.environ.get("PFA_RESTART_PROCS", "1"))
MINIM_WORKERS = int(os.environ.get("PFA_MINIM_WORKERS", "0"))


def main() -> None:
    # legacy_library=True: see the module docstring. Do not remove without regenerating
    # the golden set.
    rep = reps.TorsionStateRepresentation(len(SEQ), n_states=8, sequence=SEQ,
                                          legacy_library=True)
    H = build_obc2_native(SEQ, rep)

    _, nc, nphi, npsi = geo.native_coords_from_pdb(
        os.path.join(os.path.dirname(amb.__file__), "pdbs", "1UAO.pdb"))
    nat_ca = np.asarray(nc["CA"])
    rmsd = lambda b: geo.ca_rmsd(rep.build_coords(b)["CA"], nat_ca)

    sampler = MPSSampler(H.n_qubits, layers=4)
    # legacy_library must match `rep` above, or workers would decode bitstrings to
    # different torsions than the parent and silently return different energies.
    builder = (OBC2Builder(SEQ, 8, legacy_library=True)
               if RESTART_PROCS > 1 else None)
    print(f"n_qubits={H.n_qubits}  LEGACY 8-state library  MPS backend  config A"
          f"  | restart_procs={RESTART_PROCS} minim_workers={MINIM_WORKERS}", flush=True)
    res = run_global_cvar_vqe(H, sampler=sampler, layers=4, alpha=0.15, shots=1024,
                              maxiter=200, restarts=3, seed=0, final_shots=8192,
                              init_scale=0.25, verbose=True,
                              hamiltonian_builder=builder,
                              restart_procs=RESTART_PROCS, minim_workers=MINIM_WORKERS,
                              optimizer_guard="warn")
    a = res["audit"]

    print("\n============ 8-STATE CHIGNOLIN, SEED 0 (config A, LEGACY lib) ============")
    print(f"  evals {res['n_objective_evals_total']} | "
          f"unique {res['n_unique_structures_cached']}")
    n_floored = sum(1 for v in H._cache.values() if not np.isfinite(v))
    print(f"\n  --- COLLAPSE AUDIT (source floor -600; collapses -> +inf, excluded) ---")
    print(f"  floor caught {n_floored} collapse structures of {H.cache_size()} "
          f"unique (floored to +inf)")
    print(f"  (i)   selected vqe_bitstring energy : {a['final_selected_energy']:.2f} kcal  "
          f"-> {'PHYSICAL (-350..-390)' if -390 <= a['final_selected_energy'] <= -350 else 'OUT OF RANGE'}"
          f"{'  *** FLOORED ***' if a['final_selected_is_collapse'] else ''}")
    print(f"  (ii)  evals whose CVaR low-tail contained a floored structure: "
          f"{a['n_evals_lowtail_collapse_best_run']}/{a['n_evals_best_run']} (best run) | "
          f"{a['n_evals_lowtail_collapse_all']} (all runs)  [must be 0]")
    print(f"  (iii) final selected tail : lowest-5 energies "
          f"{[round(x, 1) for x in a['final_sample_lowest5']]}  "
          f"({a['final_sample_n_floored']} floored excluded from "
          f"{a['final_sample_n_unique']}-unique final sample)")

    clean = ((not a['final_selected_is_collapse'])
             and (a['n_evals_lowtail_collapse_best_run'] == 0)
             and (-390 <= a['final_selected_energy'] <= -350))
    print(f"\n  AUDIT VERDICT: "
          f"{'CLEAN (floor working: collapses excluded, answer physical)' if clean else 'CONTAMINATED -> STOP'}")

    if clean:
        r = rmsd(res['vqe_bitstring'])
        print(f"\n  --- RESULT ---")
        print(f"  CA-RMSD (vqe_bitstring) : {r:.2f} A")
        e_nat_real = H.energy_from_coords(nc, nphi, npsi)
        e_nat_snap8 = H.energy(rep.angles_to_bits(nphi, npsi))
        print(f"\n  --- native-vs-prediction energy gap at 8 states ---")
        print(f"  native (real backbone)   {e_nat_real:.2f}")
        print(f"  native (snapped 8-state) {e_nat_snap8:.2f}")
        print(f"  prediction (vqe)         {res['vqe_energy']:.2f}")
        print(f"  gap real-native - pred     : {e_nat_real - res['vqe_energy']:+.2f} kcal")
        print(f"  gap snap8-native - pred    : {e_nat_snap8 - res['vqe_energy']:+.2f} kcal")
    print(f"\n  best_seen (blowup-unreliable): {res['best_seen_energy']:.1f}  "
          f"RMSD {rmsd(res['best_seen_bitstring']):.2f} A")
    print("  vqe_bitstring:", res['vqe_bitstring'], flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
