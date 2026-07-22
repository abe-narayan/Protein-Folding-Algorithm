# Protein Folding: Quantum vs. Classical Scalability

**Research question:** How do quantum and classical optimization methods compare in
scalability as protein length increases in lattice-based protein folding?

## Overview

Protein folding is a complex optimization problem where the goal is to find the
lowest-energy structure of an amino acid sequence. As protein length increases, the
number of possible folding configurations grows exponentially, creating challenges for
classical optimization methods.

This project explores the scalability of quantum and classical approaches for
lattice-based protein folding. A hybrid quantum-classical algorithm using a
CVaR-optimized Variational Quantum Eigensolver (VQE) is implemented and compared with
classical methods such as brute-force search and simulated annealing. The goal is to
analyze how accuracy, computational cost, and resource requirements change as protein
length increases.

## Objectives

- Implement a lattice-based protein folding model
- Encode folding configurations and construct the cost Hamiltonian
- Implement CVaR-optimized VQE
- Compare quantum results with classical optimization methods
- Analyze scalability through runtime, accuracy, and resource requirements

## Repository structure

| File | Purpose |
| --- | --- |
| `encoding.py` | Lattice model and turn-based encoding of folding configurations |
| `hamiltonian.py` | Construction of the cost Hamiltonian from a sequence |
| `vqe.py` | Ansatz, CVaR objective, and the variational optimization loop |
| `local_hamiltonian.py` | Cost-function wrapper over the energy model |
| `classical.py` | Classical baseline (exhaustive brute-force search) |
| `main.py` | Entry point for running the VQE |
| `results/` | Generated data and figures (planned) |

## Dependencies

The project requires Python 3 and the following packages (all pinned in
`requirements.txt`):

| Package | Used for |
| --- | --- |
| `numpy` | Numerical arrays and linear algebra |
| `scipy` | Classical optimizer (`scipy.optimize.minimize`) |
| `matplotlib` | Plotting protein structures and results |
| `pennylane` | Quantum circuits and the VQE (`pennylane as qml`) |
| `networkx` | Graph utilities for the lattice model |
| `biopython` | Parsing reference PDB structures (`Bio.PDB.PDBParser`) |
| `jupyter` | Running exploratory notebooks |

The standard-library modules `csv`, `os`, and `time` are also used but require no
installation.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Energy model notes

The energy in `hamiltonian.py` is a **relative Miyazawa-Jernigan lattice contact
score** (contact matrix + a radius-of-gyration compactness term + an overlap
penalty). It is dimensionless and is **not** a physical free energy in kcal/mol, so
its absolute value is not comparable to any experimental measurement. Correspondence
to real structures is measured by **Cα RMSD** to a reference PDB (see `main.py`), not
by the energy value.

**Disulfide bonds.** Cys-Cys disulfides are covalent links that often dominate a small
peptide's structure. `find_disulfide_pairs` infers them from the sequence (a peptide
with exactly two cysteines is assumed to form one disulfide) and `path_energy` applies
them by default as a **hard topological constraint** (bonded cysteines must sit within
the lattice bond shell), restricting the search to the native cyclic class. For
oxytocin (`CYIQNCPLG`, ref `7OFG`, header `CYS-CYS DISULFIDE BOND`) the bond is Cys1-Cys6.

## What is (and isn't) validated

Reporting a single RMSD-to-reference is not a result here, because the energy ground
state is highly degenerate (96 symmetry-equivalent folds for oxytocin, RMSD 0.53-0.65),
so a bare RMSD reflects which degenerate fold was sampled, not model skill. `analysis.py`
and `validate_structure.py` replace it with measured statistics (nothing tuned to the
reference). For oxytocin vs `7OFG` (exact enumeration, 42,184 folds):

- **Predictive validity — negative.** Spearman(energy, RMSD) = **−0.26** (p ≈ 0): lower
  MJ energy predicts a *less* native fold. The MJ contact potential rewards hydrophobic
  collapse and is out of its domain for a small solvent-exposed cyclic peptide, so
  energy minimization is not a valid structure predictor here. A single energy-minimum
  fold must **not** be reported as a prediction.
- **Disulfide constraint — significant, positive.** Restricting to disulfide-satisfying
  folds shifts the accessible ensemble toward native: median Cα-RMSD **0.498 → 0.443**
  (Mann-Whitney U, one-sided, **p ≈ 6×10⁻¹⁹⁶**). This is a pure-geometry result,
  independent of the energy objective.

Run `python validate_structure.py` to regenerate these numbers and
`results/structure_validity_CYIQNCPLG.png`.

## Status

Work in progress. The core module files are now implemented: the turn-based lattice
encoding, the cost Hamiltonian / energy model, an exhaustive brute-force search, and a
CVaR-optimized VQE with its ansatz and optimization loop.

Still to do:

- Connect the VQE and brute-force search into a single-sequence validation harness that
  confirms the VQE recovers the true minimum-energy fold on small proteins.
- Add the simulated-annealing classical baseline.
- Implement the scalability study across protein lengths (runtime, accuracy, and
  resource requirements) and generate the comparison plots.
