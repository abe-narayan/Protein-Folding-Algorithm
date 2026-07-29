"""Predict via the MPS sampler, for registers too big for the statevector path.

    python predict_mps.py 1LE0
"""
import os, sys
# Must precede the numpy import: pinning is a determinism requirement, not a speedup.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import protein_geometry as geo
import representations as reps
import hamiltonian as ham
import dataset as ds
import experiments as exp
from mps.sampler import MPSSampler
from mps import driver

pdb_id = sys.argv[1] if len(sys.argv) > 1 else "1LE0"
seq = geo.parse_pdb(ds.download_pdb(pdb_id))[0]   # before the search, so this is legal
rep = reps.make_representation("torsion", len(seq), n_states=4, sequence=seq)
H = ham.FoldingHamiltonian(seq, rep, eval_budget=exp.DEFAULT_EVAL_BUDGET)
cfg = exp.default_vqe_config()
print(f"{pdb_id}: {seq}  (N={len(seq)}, qubits={rep.n_qubits})")

geo.reset_pdb_log()                               # the leakage audit starts here
res = driver.run_global_cvar_vqe(
    H, sampler=MPSSampler(rep.n_qubits, cfg["layers"]),
    seed=0, verbose=True, **cfg)
assert len(geo.get_pdb_log()) == 0, "LEAKAGE: PDB read during the search"
H.end_search()

out = os.path.join(exp._ensure_results_dir(), f"{seq}_prediction.pdb")
geo.write_pdb(out, seq, rep.build_coords(res["vqe_bitstring"]),
              remark=f"MPS CVaR-VQE sequence-only prediction ({pdb_id})")
print("wrote", out)