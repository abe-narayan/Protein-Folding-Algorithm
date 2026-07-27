"""8-state chignolin CVaR-VQE, seed 0, config A (layers 4 / maxiter 200 /
restarts 3 / shots 1024), default STATES_8 torsion library, MPS backend, OBC2-native
solvation tier.

The collapse audit is pure observation (the source floor sends collapses to +inf, so
they cannot enter the CVaR low tail or be selected). Reports the selected energy, the
audit verdict, CA-RMSD vs the 4-state 4.80 A and the 8-state ceiling, and the
native-vs-prediction energy gap.

This is the driver that produced the persisted seed-0 evaluation set. Run with the
interpreter that owns openmm + quimb, e.g.:

    PYTHONPATH=<repo> python run_8state_seed0.py
"""
import os

import numpy as np

import amber_hamiltonian as amb
import representations as reps
import protein_geometry as geo
from amber_obc2 import build_obc2_native
from mps import MPSSampler, run_global_cvar_vqe

SEQ = "GYDPETGTWG"


def main() -> None:
    rep = reps.TorsionStateRepresentation(len(SEQ), n_states=8)   # default STATES_8
    H = build_obc2_native(SEQ, rep)

    _, nc, nphi, npsi = geo.native_coords_from_pdb(
        os.path.join(os.path.dirname(amb.__file__), "pdbs", "1UAO.pdb"))
    nat_ca = np.asarray(nc["CA"])
    rmsd = lambda b: geo.ca_rmsd(rep.build_coords(b)["CA"], nat_ca)

    sampler = MPSSampler(H.n_qubits, layers=4)
    print(f"n_qubits={H.n_qubits}  default STATES_8  MPS backend  config A", flush=True)
    res = run_global_cvar_vqe(H, sampler=sampler, layers=4, alpha=0.15, shots=1024,
                              maxiter=200, restarts=3, seed=0, final_shots=8192,
                              init_scale=0.25, verbose=True)
    a = res["audit"]

    print("\n==================== 8-STATE CHIGNOLIN, SEED 0 (config A) ====================")
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
        print(f"     vs 4-state 4.80 A    : {'improved' if r < 4.80 else 'worse'} by {4.80 - r:+.2f} A")
        print(f"     vs 8-state ceiling 0.919 A : +{r - 0.919:.2f} A above ceiling")
        e_nat_real = H.energy_from_coords(nc, nphi, npsi)
        e_nat_snap8 = H.energy(rep.angles_to_bits(nphi, npsi))
        print(f"\n  --- native-vs-prediction energy gap at 8 states ---")
        print(f"  native (real backbone)   {e_nat_real:.2f}")
        print(f"  native (snapped 8-state) {e_nat_snap8:.2f}")
        print(f"  prediction (vqe)         {res['vqe_energy']:.2f}")
        print(f"  gap real-native - pred     : {e_nat_real - res['vqe_energy']:+.2f} kcal")
        print(f"  gap snap8-native - pred    : {e_nat_snap8 - res['vqe_energy']:+.2f} kcal  (4-state snap was ~+33)")
    print(f"\n  best_seen (blowup-unreliable): {res['best_seen_energy']:.1f}  "
          f"RMSD {rmsd(res['best_seen_bitstring']):.2f} A")
    print("  vqe_bitstring:", res['vqe_bitstring'], flush=True)


if __name__ == "__main__":
    main()
