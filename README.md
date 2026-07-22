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

## Setup

```bash
pip install -r requirements.txt
python main.py
```

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
