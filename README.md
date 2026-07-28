# Peptide Structure Prediction with a Global CVaR-VQE

Sequence-only structure prediction for short peptides, cast as a global Variational
Quantum Eigensolver with a CVaR objective. One circuit covers the entire peptide (no
block decomposition), and no native structure is read during the search. Chignolin
(`GYDPETGTWG`, PDB `1UAO`) is the primary target.

The comparison the repo is built for: how do quantum and classical optimizers compare in
scalability as peptide length grows? Four arms — lattice-VQE, torsion-VQE, simulated
annealing, random search — search the same space against the same energy model.

## Findings

All evidence is chignolin at one length, three seeds, on the legacy energy model. That
is narrow, and the results below should not be read as general claims about VQE.

- On the lattice arm, annealing reaches the global optimum in 185 unique energy
  evaluations; the VQE needs ~16 358; random search does not find it in 20 000.
- That optimum is a straight extended chain, 9.85 Å CA-RMSD from native against a 1.94 Å
  representation ceiling. Every arm that succeeds returns it. The energy model, not the
  optimizer, is what limits the result.
- On the torsion arm at matched cost, the VQE returns the worst energy of the three
  arms (−1.306 vs −2.282 and −2.153) with about five times the seed-to-seed spread.
  Random search beat it on every seed.
- The VQE has the best CA-RMSD on that arm (4.33 Å vs 5.17 and 5.49). This is not a
  search win. It minimizes the objective least well and lands nearer the native
  structure as a result, because the energy model's minimum is in the wrong place.
- Length scaling is analytic only. Qubit count and memory are derived; nothing was
  measured across lengths.

## What's here

Two energy models behind one Hamiltonian interface:

- **legacy** (`energy_terms.py`, `hamiltonian.py`) — 7-term score: steric (weight 4.0),
  MJ contacts (1.0), DSSP-style H-bond (1.0), torsion (1.0), electrostatic (0.5),
  solvation (0.3), compactness (0.05). Known to be wrong: natives can score worse than
  predictions. Every number here is search efficiency on that landscape.
- **amber** (`amber_hamiltonian.py`) — OpenMM ff14SB with implicit-solvent GB,
  backbone-restrained capped minimization, and a collapse floor sending GB
  Coulomb-collapse artifacts to `+inf`. Not yet run.

Two representations (`representations.py`), 2 bits per slot:

| rep | qubits | config space | expresses |
| --- | --- | --- | --- |
| torsion — per-residue (φ, ψ) from a 4- or 8-state library; backbone and Cβ by NeRF, sidechains by `sidechains.py` | `2N` / `3N` | `n_states ** N` | helix, strand, turns; chiral |
| lattice — tetrahedral CA-only trace | `2(N−1)` | `4 ** (N−1)` | β approximately, turns coarsely; achiral |

Two VQE backends: **lightning** (`vqe.py`, dense statevector, ~30 qubits) and **MPS**
(`mps/`, quimb, for larger registers). Both use the same ansatz — `layers` × (RY on
every wire, CNOT chain, ring closure) — optimized by COBYLA or SPSA against a CVaR-α
objective over sampled bitstrings.

## Results

### Lattice arm, chignolin

18 qubits, so the space can be enumerated. All 262 144 structures scored in 26 s; the
minimum is `000000000000000000` at `E = 9.847041`.

| method | best energy | unique evals to reach it | runtime |
| --- | ---: | ---: | ---: |
| exhaustive (reference) | 9.847041 | 262 144 | 26 s |
| simulated annealing | 9.847041 | 185 | 0.1 s |
| lattice CVaR-VQE | 9.847041 | ~16 358 | 17 s |
| random search | 34.079 | not found in 20 000 | 2 s |

The three successful arms return the same bitstring. CA-RMSD 9.85 Å, ceiling 1.94 Å,
contact F1 0.00.

### Torsion arm, chignolin, cost-matched

Legacy model, 4-state torsion (20 qubits), seeds 0/1/2, 20 000 unique energy evaluations
per arm (`results/main_comparison_matched.csv`):

| arm | evals | best energy (the objective) | CA-RMSD (Å) |
| --- | ---: | ---: | ---: |
| B_torsion_vqe | 19 982 | −1.306 ± 0.583 | 4.33 ± 0.66 |
| C_torsion_sa | 20 000 | −2.282 ± 0.118 | 5.17 ± 0.45 |
| D_torsion_random | 19 820 | −2.153 ± 0.061 | 5.49 ± 0.15 |

Per-seed energy — VQE −2.130 / −0.872 / −0.918; SA −2.203 / −2.194 / −2.450; random
−2.069 / −2.213 / −2.176. One of three VQE seeds was competitive.

Cost is counted in unique energy evaluations (cache misses), which under the amber model
is one OpenMM minimization each and dominates wall-clock for every arm. Step counts are
not comparable: annealing revisits structures often, so 20 000 SA steps cost ~11–14k
evaluations while 20 000 random draws cost ~19.8k.

### Scaling

`experiments.experiment_scaling_report` → `results/scaling.json`. Derived, not measured.

| N | lattice q | torsion-4 q | torsion-8 q | torsion-4 space | statevector |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 18 | 20 | 30 | 1.0e6 | 16 MB |
| 12 | 22 | 24 | 36 | 1.7e7 | 268 MB |
| 15 | 28 | 30 | 45 | 1.1e9 | 17 GB |
| 20 | 38 | 40 | 60 | 1.1e12 | 17 TB |

At a 30-qubit statevector limit a global VQE reaches N ≈ 16 on the lattice, N ≈ 15 on
4-state torsion, N ≈ 10 on 8-state torsion. N = 20 is not simulable globally. MPS raises
the reachable qubit count; the configuration space still grows as `n_states ** N`.

## Quickstart

```bash
pip install -r requirements.txt
python main.py --validate    # 35-test suite (geometry, energy, CVaR, VQE, amber)
python main.py --scaling     # analytic qubit / memory / config-space report
python main.py --predict --sequence GYDPETGTWG
```

`main.py` requires `openmm` for every subcommand, including those that never use the
Amber model: `main.py:12` imports `amber_hamiltonian`, which imports `openmm` at module
scope. Without it, `--scaling` fails on import. The library modules are unaffected —
`representations`, `hamiltonian`, `vqe`, `classical_baselines` and `evaluation` import
cleanly and produced the lattice results above.

`--predict` prints the VQE solution, distribution stats, and (legacy model) a weighted
energy breakdown against a known native, then writes `results/prediction.pdb`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `protein_geometry.py` | Backbone/NeRF geometry, Kabsch/RMSD, PDB IO, DSSP, torsions |
| `sidechains.py` | Fixed-chi heavy-atom sidechains (G A S T D E N K P Y W) |
| `representations.py` | Torsion-state library, tetrahedral lattice, `make_representation` |
| `energy_terms.py`, `hamiltonian.py` | Legacy 7-term energy and `FoldingHamiltonian` |
| `amber_hamiltonian.py`, `amber_obc2.py` | OpenMM ff14SB + GBn2 with collapse floor; OBC2-native variant |
| `vqe.py` | CVaR objective, global ansatz, lightning VQE loop |
| `mps/` | `MPSSampler` and MPS driver |
| `classical_baselines.py` | Random search, simulated annealing, exhaustive search |
| `evaluation.py` | Representation ceiling and predicted-vs-native metrics |
| `dataset.py` | PDB download/cache, peptide dataset, identity clustering |
| `experiments.py` | Experiment drivers and CSV IO |
| `validation.py` | 35-test correctness/leakage suite (`run_all`) |
| `main.py`, `plot_structures.py` | CLI entry point; 3D structure plots |
| `budget.py` | Shared evaluation-budget accounting; not wired in (see Status) |
| `run_8state_seed0.py` | 8-state OBC2-native chignolin seed-0 run |
| `pdbs/`, `results/`, `docs/` | Cached PDBs; generated data; design notes |

The suite pins determinism under fixed seeds, absence of native leakage during
optimization, that the VQE is global rather than blockwise, that it does not enumerate,
and that it matches exhaustive search on an enumerable system.

## Status

Verified here (Python 3.14, numpy 2.5.1, scipy 1.18, PennyLane 0.45.1, biopython 1.87):
the lattice enumeration and per-method costs above, and VQE matching exhaustive search
at 14 qubits on `GYDPETG` (−1.0649, also matched by SA and random search).

Open gaps:

1. `budget.py` is imported by nothing. Both Hamiltonians keep their own uncapped
   counters and `experiment_main_comparison` takes no budget argument. The wiring,
   `config_id` stamping, optimizer guard and five tests that `docs/evaluation-budget.md`
   describes as landed are not in the source, so `main_comparison_matched.csv` cannot be
   regenerated by this code.
2. `--validate` cannot run without `openmm`, so the 35-test suite is unrun here. The
   "35/35 passed" in the docs predates the current tree.
3. The Amber and MPS paths are unexecuted; neither `openmm` nor `quimb` is installed.
   The Amber comparison is the one that matters.
4. `results/main_comparison.csv` is stale: unmatched budgets, mixed `maxiter`/`restarts`
   configs interleaved, and every row seed 0 despite the summary printing a standard
   deviation across seeds. Superseded by `main_comparison_matched.csv`.
5. There is no length sweep. A cost-matched sweep over N is the missing experiment.
6. The VQE has not been retuned for a matched budget. `alpha`, `layers` and
   COBYLA-vs-SPSA were chosen under the earlier per-arm regime.

## Dependencies

`numpy`, `scipy`, `pennylane` + `pennylane-lightning`, `quimb` (MPS), `biopython` (PDB
parsing), `matplotlib`, `openmm` (Amber model, and currently an unconditional import for
`main.py`). Pinned in `requirements.txt`.
