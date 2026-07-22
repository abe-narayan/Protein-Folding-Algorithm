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
a distance restraint that pulls the bonded cysteines together, reproducing the native
cyclic topology. For oxytocin (`CYIQNCPLG`, ref `7OFG`, whose header notes a
`CYS-CYS DISULFIDE BOND`) this drives the Cys1-Cys6 pair into contact and lowers RMSD
to the reference (~0.62 → ~0.51) — a realistic constraint, not a tuned fit.

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
