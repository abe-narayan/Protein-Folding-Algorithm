# Peptide Structure Prediction with a Global CVaR-VQE

Sequence-only structure prediction for short peptides (10–20 residues), cast as a
global Variational Quantum Eigensolver (VQE) with a CVaR objective. The optimizer
searches a discrete torsion-state configuration space over the **entire** peptide at
once (no block decomposition) and never reads a native structure during the search.

Chignolin (`GYDPETGTWG`, PDB `1UAO`) is the primary target.

## What's here

Two independent energy models behind one Hamiltonian interface:

- **legacy** (`energy_terms.py` + `hamiltonian.py`) — a fast, physics-flavoured 7-term
  score (Miyazawa–Jernigan contacts, DSSP-style H-bonds, steric, solvation,
  electrostatics, Ramachandran torsion, compactness).
- **amber** (`amber_hamiltonian.py`) — OpenMM Amber ff14SB with implicit-solvent GB,
  backbone-restrained capped minimization, and a collapse floor that sends GB
  Coulomb-collapse artifacts to `+inf` so they can never be selected.

Two structure representations (`representations.py`):

- **torsion** — per-residue backbone (φ, ψ) drawn from a 4- or 8-state discrete torsion
  library; full backbone + Cβ built by NeRF, sidechains by `sidechains.py`. This is the
  representation the Amber model requires.
- **lattice** — a tetrahedral CA-only lattice (Cα trace), for scaling comparisons.

Two VQE backends:

- **lightning** (`vqe.py`) — dense statevector via PennyLane; the default, simulable to
  ~30 qubits.
- **MPS** (`mps/`) — an exact-to-cutoff matrix-product-state sampler (quimb) with the
  same ansatz, for qubit counts a dense statevector cannot hold (e.g. 30-qubit STATES_8
  chignolin).

## Quickstart

```bash
pip install -r requirements.txt
python main.py --validate          # 35-test suite (geometry, energy, CVaR, VQE, amber)
python main.py --scaling           # analytic qubit / memory / config-space report
python main.py --predict --sequence GYDPETGTWG
```

`--predict` prints the VQE solution, distribution stats, and (legacy model) a weighted
energy breakdown vs a known native, then writes `results/prediction.pdb`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `protein_geometry.py` | Backbone/NeRF geometry, Kabsch/RMSD, PDB IO, DSSP, torsions |
| `sidechains.py` | Fixed-chi heavy-atom sidechains (G A S T D E N K P Y W) |
| `representations.py` | Torsion-state library + tetrahedral lattice; `make_representation` |
| `energy_terms.py` | Legacy 7-term energy (MJ, Ramachandran, H-bond, …) |
| `hamiltonian.py` | `FoldingHamiltonian` — legacy energy model |
| `amber_hamiltonian.py` | `AmberHamiltonian` — OpenMM ff14SB + GBn2, collapse floor |
| `amber_obc2.py` | `build_obc2_native` — AmberHamiltonian on the native GBSAOBCForce (OBC2) |
| `vqe.py` | CVaR objective, ansatz, and the lightning CVaR-VQE loop |
| `mps/` | `MPSSampler` + MPS `run_global_cvar_vqe` (pluggable-sampler driver + collapse audit) |
| `classical_baselines.py` | Random search, simulated annealing, exhaustive search |
| `evaluation.py` | Representation ceiling + predicted-vs-native metrics |
| `dataset.py` | PDB download/cache, peptide dataset, identity clustering |
| `experiments.py` | Experiment drivers + CSV IO (`run_one`, main comparison, ablations) |
| `main.py` | CLI entry point |
| `validation.py` | 35-test correctness/leakage suite (`run_all`) |
| `plot_structures.py` | 3D structure plots (`python plot_structures.py <SEQ>`) |
| `run_8state_seed0.py` | Driver for the 8-state OBC2-native chignolin seed-0 run |
| `pdbs/`, `results/` | Cached reference PDBs; generated data and figures |

## Notes

- **Determinism.** With fixed seeds every energy and selection is reproducible to the
  last bit (the validation suite asserts Amber energy spread `0.000e+00`); `--validate`
  produces byte-identical output run to run.
- **No native leakage.** The Hamiltonians take no native structure, and the suite audits
  that no PDB is read during optimization.
- **Honest scaling limit.** A genuine global VQE with the 4-state torsion representation
  is simulable to N ≈ 15 residues (≈30 qubits); the MPS backend extends the reachable
  qubit count but the config space still grows as `n_states ** N`.

## Dependencies

`numpy`, `scipy`, `pennylane` + `pennylane-lightning` (lightning backend), `quimb` (MPS
backend), `biopython` (PDB parsing), `matplotlib` (plots), and `openmm` (the Amber
energy model). All pinned in `requirements.txt`.
