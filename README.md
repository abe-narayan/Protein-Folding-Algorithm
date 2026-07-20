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
| `classical.py` | Classical baselines (brute-force search, simulated annealing) |
| `main.py` | Entry point for running the scalability comparison |
| `results/` | Generated data and figures |

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Status

Work in progress. A working prototype of the 2-qubit toy model (2D lattice,
CVaR-VQE with a brute-force benchmark) exists as a notebook; the module files above
are still being filled in, and the scalability study across protein lengths is not yet
implemented.
