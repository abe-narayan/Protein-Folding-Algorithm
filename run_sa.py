"""Run the classical arm (simulated annealing) on one target and report RMSD.

  python run_sa.py 1UAO                 # one PDB id, seeds 0,1,2
  python run_sa.py 1UAO --steps 200000  # longer anneal
  python run_sa.py ALL                  # every peptide in the dataset

SA is the arm that reaches the global minimum: it found chignolin's 1.96 A optimum
on all 3 seeds at the standard 20000-step budget, where the VQE needed ~7.6x the
evaluations and random search reached only 2.23 A.
"""
import argparse, time
import numpy as np
import classical_baselines as cb, dataset as ds, evaluation as ev
import hamiltonian as ham, protein_geometry as geo, representations as reps

ap = argparse.ArgumentParser()
ap.add_argument("target", help="PDB id (e.g. 1UAO) or ALL")
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--states", type=int, default=4, choices=[4, 8])
a = ap.parse_args()

ids = None if a.target.upper() == "ALL" else [a.target.upper()]
seeds = [int(s) for s in a.seeds.split(",")]
entries = ds.build_dataset(pdb_ids=ids, verbose=False)
if not entries:
    raise SystemExit(f"{a.target} is not in the dataset (or failed the rebuild gate)")

print(f"simulated annealing, {a.steps} steps, seeds {seeds}, {a.states} states\n")
print(f"{'pdb':<7}{'seq':<20}{'RMSD':>7}{'+/-':>7}{'ens-min':>9}{'ceiling':>9}{'energy':>10}{'sec':>7}")
for e in entries:
    seq, nc, nphi, npsi = ds.load_native(e)
    rep = reps.TorsionStateRepresentation(len(seq), n_states=a.states, sequence=seq)
    nat = np.asarray(nc["CA"], float)
    ceil = ev.representation_ceiling(rep, nat, nphi, npsi, seed=0)["ceiling_ca_rmsd"]
    rmsds, ens, es, t0 = [], [], [], time.time()
    for s in seeds:
        H = ham.FoldingHamiltonian(seq, rep, cache_limit=0)
        out = cb.simulated_annealing(H, n_steps=a.steps, seed=s)
        ca = rep.build_coords(out["best_bitstring"])["CA"]
        rmsds.append(geo.ca_rmsd(ca, nat))
        models = [np.asarray(c["CA"], float)
                  for _, c, _, _ in geo.native_ensemble_from_pdb(e.pdb_path)]
        ens.append(geo.ca_rmsd_to_ensemble(ca, models)["min"])
        es.append(out["best_energy"])
    print(f"{e.pdb_id:<7}{seq:<20}{np.mean(rmsds):>7.2f}{np.std(rmsds):>7.2f}"
          f"{np.min(ens):>9.2f}{ceil:>9.2f}{np.mean(es):>10.3f}{time.time()-t0:>7.0f}")
