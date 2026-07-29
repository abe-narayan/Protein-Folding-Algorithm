"""Is 6.6 A an objective failure or a search failure?

    python why_rmsd.py 1LE1 0000000000000000000000001111
"""
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import protein_geometry as geo
import representations as reps
import hamiltonian as ham
import energy_terms as et
import evaluation as ev
import dataset as ds

pdb_id, bits = sys.argv[1].upper(), sys.argv[2]

ens = geo.native_ensemble_from_pdb(ds.download_pdb(pdb_id))
nseq, ncoords, nphi, npsi = ens[0]
models = [np.asarray(c["CA"]) for _, c, _, _ in ens]

rep = reps.make_representation("torsion", len(nseq), n_states=4, sequence=nseq)
H = ham.FoldingHamiltonian(nseq, rep, eval_budget=None)   # no budget: this is analysis

pred = rep.build_coords(bits)
ceil = ev.representation_ceiling(rep, models[0], nphi, npsi)
best = rep.build_coords(ceil["ceiling_bitstring"])

rows = [
    ("prediction", H.components(bits), H.energy(bits),
     geo.ca_rmsd(pred["CA"], models[0], allow_scale=False),
     geo.assign_secondary_structure(pred)),
    ("best-RMSD in rep", H.components(ceil["ceiling_bitstring"]),
     H.energy(ceil["ceiling_bitstring"]),
     ceil["ceiling_ca_rmsd"], geo.assign_secondary_structure(best)),
    ("native", et.energy_components(nseq, ncoords, nphi, npsi),
     H.energy_from_coords(ncoords, nphi, npsi),
     0.0, geo.assign_secondary_structure(ncoords)),
]

print(f"sequence : {nseq}")
print(f"representation ceiling : {ceil['ceiling_ca_rmsd']:.2f} A "
      f"(native projected onto the state library: "
      f"{ceil['projection_ca_rmsd']:.2f} A)\n")

print(f"  {'term':<16} " + "".join(f"{r[0]:>18}" for r in rows))
for term in et.TERM_NAMES:
    w = H.weights.get(term, 0.0)
    print(f"  {term:<16} " + "".join(f"{w * r[1][term]:>18.3f}" for r in rows))
print(f"  {'TOTAL':<16} " + "".join(f"{r[2]:>18.3f}" for r in rows))
print(f"  {'CA-RMSD':<16} " + "".join(f"{r[3]:>18.2f}" for r in rows))
print(f"  {'topology':<16} " + "".join(f"{r[4]:>18}" for r in rows))

e_pred, e_best = rows[0][2], rows[1][2]
print()
if e_best > e_pred:
    print(f"OBJECTIVE FAILURE: the best structure the representation can express "
          f"({ceil['ceiling_ca_rmsd']:.2f} A) scores {e_best:.2f}, WORSE than the "
          f"prediction's {e_pred:.2f}. No search could have found it. "
          f"Calibrate weights (--calibrate-weights) before touching the search.")
else:
    print(f"SEARCH FAILURE: the {ceil['ceiling_ca_rmsd']:.2f} A structure scores "
          f"{e_best:.2f}, BETTER than the prediction's {e_pred:.2f}. The objective "
          f"points the right way and the VQE did not get there -- raise --eval-budget "
          f"or lower --shots to buy optimizer steps.")