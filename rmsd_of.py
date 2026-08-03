"""RMSD of a bitstring against a native.  python rmsd_of.py <bitstring> [PDB_ID]

  python rmsd_of.py 0000110000111001000111          # defaults to 1UAO
  python rmsd_of.py <bits> 5AWL
"""
import sys
import numpy as np
import dataset as ds, energy_terms as et, hamiltonian as ham
import protein_geometry as geo, representations as reps

bits = sys.argv[1].strip()
pdb = (sys.argv[2] if len(sys.argv) > 2 else "1UAO").upper()

entry = ds.build_dataset(pdb_ids=[pdb], verbose=False)
if not entry:
    raise SystemExit(f"{pdb} not in the dataset")
seq, nc, nphi, npsi = ds.load_native(entry[0])
rep = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
if len(bits) != rep.n_bits:
    raise SystemExit(f"{pdb} needs {rep.n_bits} bits, got {len(bits)}")
H = ham.FoldingHamiltonian(seq, rep, cache_limit=0)
nat = np.asarray(nc["CA"], float)
models = [np.asarray(c["CA"], float) for _, c, _, _ in
          geo.native_ensemble_from_pdb(entry[0].pdb_path)]

ph, ps, co, rg = rep.build_all(bits)
w = rep.bits_per_residue
print(f"target    : {pdb}  {seq}")
print(f"bitstring : {bits}")
print(f"states    : {[int(bits[i*w:(i+1)*w],2) for i in range(len(seq))]}")
print(f"energy    : {H.energy(bits):.4f}")
print(f"CA-RMSD   : {geo.ca_rmsd(co['CA'], nat):.3f} A  (model 1)")
print(f"          : {geo.ca_rmsd_to_ensemble(co['CA'], models)['min']:.3f} A  "
      f"(best of {len(models)} models)")
print(f"SS        : {geo.assign_secondary_structure(co)}   native "
      f"{geo.assign_secondary_structure(nc)}")
comp = et.energy_components(seq, co, ph, ps, rings=rg)
print("\nweighted terms:")
for t in ("steric", "hbond_local", "hbond_longrange", "coop_helix",
          "coop_sheet", "aromatic", "solvation"):
    print(f"  {t:<18}{et.DEFAULT_WEIGHTS[t]*comp[t]:>9.3f}")
