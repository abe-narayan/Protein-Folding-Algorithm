# Peptide Structure Prediction with a Global CVaR-VQE

Sequence-only structure prediction for short peptides, cast as a global Variational
Quantum Eigensolver with a CVaR objective. The optimizer searches a discrete
configuration space over the **entire** peptide at once (no block decomposition) and
never reads a native structure during the search.

Chignolin (`GYDPETGTWG`, PDB `1UAO`) is the primary target.

The comparison this repo exists to make: **how do quantum and classical optimizers
compare in scalability as peptide length grows?** Four arms — lattice-VQE, torsion-VQE,
simulated annealing, random search — search the same space against the same energy
model.

## The short answer

**Classical wins on this problem, and the margin is large. But the binding constraint is
the energy model, not the optimizer.**

1. **On the lattice arm, annealing reaches the global optimum ~90× cheaper than the VQE.**
   Chignolin on the tetrahedral lattice is 18 qubits — small enough to enumerate. All
   262 144 structures were scored: the true minimum is `000000000000000000` at
   `E = 9.847041`. Simulated annealing finds it in **185 unique energy evaluations**
   (0.1 s); the lattice-VQE reaches the same answer in **~16 358** (17 s); random search
   over 20 000 draws never finds it (best `34.08`).
2. **That optimum is a straight line.** All-zeros is the fully extended tetrahedral
   chain, 9.85 Å CA-RMSD from native against a representation ceiling of 1.94 Å. Every
   arm that succeeds on the lattice returns it. This is a *model* failure, not a search
   failure — the legacy energy's global minimum on this lattice is in the wrong place.
3. **On the torsion arm at matched cost, the VQE is worst at the thing it optimizes.**
   Mean best energy −1.306 vs annealing's −2.282 and random search's −2.153, with five
   times the seed-to-seed spread. Plain random search beat the CVaR-VQE on every seed.
4. **The RMSD column is a trap.** Read RMSD alone and the VQE looks best (4.33 Å vs 5.17
   and 5.49). It is not a search win: the VQE is *failing* to minimize a known-broken
   energy and landing nearer the native structure by accident. On a landscape whose
   minimum is in the wrong place, worse optimization produces better structures. Quoting
   the RMSD row as a quantum result would be exactly backwards.
5. **Length scaling here is analytic, not measured.** The matched comparison is one
   peptide at one length. See [Scaling](#scaling) and [Status](#status).

## What's here

Two energy models behind one Hamiltonian interface:

- **legacy** (`energy_terms.py` + `hamiltonian.py`) — fast 7-term score: steric (w 4.0),
  MJ contacts (1.0), DSSP-style H-bond (1.0), torsion (1.0), electrostatic (0.5),
  solvation (0.3), compactness (0.05). **Known broken** — natives can score worse than
  predictions; every number below is search efficiency on a wrong landscape.
- **amber** (`amber_hamiltonian.py`) — OpenMM Amber ff14SB with implicit-solvent GB,
  backbone-restrained capped minimization, and a collapse floor sending GB
  Coulomb-collapse artifacts to `+inf`. This is the comparison that would count; it has
  not been run.

Two representations (`representations.py`), both 2 bits per slot:

| rep | qubits | expresses | config space |
| --- | --- | --- | --- |
| **torsion** — per-residue (φ, ψ) from a 4- or 8-state library, backbone + Cβ by NeRF, sidechains by `sidechains.py` | `2N` (4-state) / `3N` (8-state) | helix, strand, turns; chiral | `n_states ** N` |
| **lattice** — tetrahedral CA-only trace, for scaling comparisons | `2(N−1)` | no helix, β approximately, turns coarsely; achiral | `4 ** (N−1)` |

Two VQE backends: **lightning** (`vqe.py`, dense statevector, ~30 qubits) and **MPS**
(`mps/`, quimb, for registers a statevector cannot hold). Both drive the same ansatz —
`layers` × (RY on every wire, then a CNOT chain plus ring closure) — optimized by COBYLA
or SPSA against a CVaR-α objective over sampled bitstrings.

## Results

### Lattice arm, chignolin — verified by enumeration

18 qubits, all 262 144 structures scored in 26 s (`classical_baselines.exhaustive_search`):

| method | best energy | unique evals to reach it | runtime |
| --- | ---: | ---: | ---: |
| exhaustive (reference) | 9.847041 | 262 144 | 26 s |
| simulated annealing | 9.847041 | **185** | 0.1 s |
| lattice CVaR-VQE | 9.847041 | ~16 358 | 17 s |
| random search | 34.079 | 20 000 (never found) | 2 s |

All three successful arms return the same bitstring, `000000000000000000`. CA-RMSD
9.85 Å; ceiling 1.94 Å; contact F1 0.00.

### Torsion arm, chignolin — cost-matched

Legacy model, 4-state torsion (20 qubits), seeds 0/1/2, 20 000 unique energy evaluations
per arm (`results/main_comparison_matched.csv`):

| arm | evals | best energy — *the objective* | CA-RMSD (Å) |
| --- | ---: | ---: | ---: |
| B_torsion_vqe | 19 982 | −1.306 ± 0.583 | **4.33 ± 0.66** |
| C_torsion_sa | 20 000 | **−2.282 ± 0.118** | 5.17 ± 0.45 |
| D_torsion_random | 19 820 | −2.153 ± 0.061 | 5.49 ± 0.15 |

Per-seed best energy — VQE −2.130 / −0.872 / −0.918; SA −2.203 / −2.194 / −2.450;
random −2.069 / −2.213 / −2.176. Only one of three VQE seeds was competitive.

Cost is counted in **unique** energy evaluations (cache misses), the one
hardware-agnostic currency: under the amber model that is one OpenMM minimization each,
and it dominates wall-clock for every arm. This matters — annealing revisits structures
heavily, so 20 000 SA *steps* cost only ~11–14k evaluations while 20 000 random draws
cost ~19.8k. Same nominal budget, ~1.7× the real cost.

### Scaling

`experiments.experiment_scaling_report` → `results/scaling.json`. Analytic only; no
optimization is run.

| N | lattice q | torsion-4 q | torsion-8 q | torsion-4 space | statevector |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 18 | 20 | 30 | 1.0e6 | 16 MB |
| 12 | 22 | 24 | 36 | 1.7e7 | 268 MB |
| 15 | 28 | 30 | 45 | 1.1e9 | 17 GB |
| 20 | 38 | 40 | 60 | 1.1e12 | 17 TB |

At a 30-qubit statevector limit a genuine global VQE reaches **N ≈ 16 (lattice)**,
**N ≈ 15 (4-state torsion)**, **N ≈ 10 (8-state torsion)**. N = 20 is not simulable
globally. The MPS backend raises the reachable qubit count, but the configuration space
still grows as `n_states ** N`, so nothing here changes the exponent.

## Quickstart

```bash
pip install -r requirements.txt
python main.py --validate    # 35-test suite (geometry, energy, CVaR, VQE, amber)
python main.py --scaling     # analytic qubit / memory / config-space report
python main.py --predict --sequence GYDPETGTWG
```

> **`main.py` requires `openmm` for every subcommand**, including the ones that never
> touch the Amber model. `main.py:12` imports `amber_hamiltonian`, which imports
> `openmm` at module scope, so without it even `--scaling` dies on import. The library
> modules themselves are fine — `representations`, `hamiltonian`, `vqe`,
> `classical_baselines` and `evaluation` import cleanly and were used directly for the
> lattice results above.

`--predict` prints the VQE solution, distribution stats, and (legacy model) a weighted
energy breakdown vs a known native, then writes `results/prediction.pdb`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `protein_geometry.py` | Backbone/NeRF geometry, Kabsch/RMSD, PDB IO, DSSP, torsions |
| `sidechains.py` | Fixed-chi heavy-atom sidechains (G A S T D E N K P Y W) |
| `representations.py` | Torsion-state library + tetrahedral lattice; `make_representation` |
| `energy_terms.py`, `hamiltonian.py` | Legacy 7-term energy and `FoldingHamiltonian` |
| `amber_hamiltonian.py`, `amber_obc2.py` | OpenMM ff14SB + GBn2 with collapse floor; OBC2-native variant |
| `vqe.py` | CVaR objective, global ansatz, lightning CVaR-VQE loop |
| `mps/` | `MPSSampler` + MPS driver (pluggable sampler, collapse audit) |
| `classical_baselines.py` | Random search, simulated annealing, exhaustive search |
| `evaluation.py` | Representation ceiling + predicted-vs-native metrics |
| `dataset.py` | PDB download/cache, peptide dataset, identity clustering |
| `experiments.py` | Experiment drivers + CSV IO (`run_one`, comparison, ablations, scaling) |
| `validation.py` | 35-test correctness/leakage suite (`run_all`) |
| `main.py`, `plot_structures.py` | CLI entry point; 3D structure plots |
| `budget.py` | Shared evaluation-budget accounting — **written but not wired in** (see Status) |
| `run_8state_seed0.py` | Driver for the 8-state OBC2-native chignolin seed-0 run |
| `pdbs/`, `results/`, `docs/` | Cached PDBs; generated data and figures; design notes |

## Properties the suite pins

- **Determinism.** With fixed seeds every energy and selection reproduces bit-for-bit
  (the suite asserts Amber energy spread `0.000e+00`).
- **No native leakage.** The Hamiltonians take no native structure, and `run_one` asserts
  no PDB was read during optimization.
- **Global, not blockwise.** `test_vqe_is_full_system` pins one circuit over all qubits;
  `test_vqe_does_not_enumerate` pins that the search never enumerates the space.
- **VQE optimality on an enumerable system.** `test_vqe_matches_exhaustive_on_tiny_system`
  — exact −2.9608, VQE −2.9608, gap 0.0000. Independently reproduced here at 14 qubits:
  exhaustive, SA, random and VQE all reach −1.0649 on `GYDPETG`.
- **The ceiling is a real ceiling.** `representation_ceiling` reports the best RMSD the
  representation could express, so a result is never credited past what the encoding
  allows.

## Status

Verified in this environment (Python 3.14, numpy 2.5.1, scipy 1.18, PennyLane 0.45.1,
biopython 1.87):

- The lattice enumeration and per-method costs in [Results](#results).
- Core library imports and runs; VQE matches exhaustive search at 14 qubits.

Known gaps, in the order they should be closed:

1. **`budget.py` is not wired to anything.** No module imports it — `FoldingHamiltonian`
   and `AmberHamiltonian` still carry their own uncapped counters, and
   `experiment_main_comparison` takes no `eval_budget`. `docs/evaluation-budget.md`
   describes the wiring, the `config_id` stamping, the `check_optimizer_budget` guard and
   five new validation tests as landed; **none of that is in the committed source.**
   `results/main_comparison_matched.csv` therefore cannot be regenerated by this code as
   it stands.
2. **`--validate` cannot run without `openmm`** (module-scope import chain, above), so
   the 35-test suite is unrun here. The doc's "35/35 passed" predates that.
3. **The Amber and MPS paths are unexecuted** — neither `openmm` nor `quimb` is
   installed. The Amber comparison is the one that counts.
4. **`results/main_comparison.csv` is stale** — unmatched budgets, mixed `maxiter`/
   `restarts` configs interleaved as one experiment, and every row seed 0 despite the
   summary printing "± std across seeds". Superseded by `main_comparison_matched.csv`;
   should be regenerated or deleted.
5. **No length sweep.** The matched comparison is a single peptide at N = 10. The
   headline scalability question is answered analytically for qubits and memory but
   empirically at one point; a sweep over N with cost-matched arms is the missing
   experiment.
6. **The VQE has not been retuned** for a fair budget. `alpha`, `layers` and
   COBYLA-vs-SPSA were all chosen under the old per-arm regime; these numbers describe
   this configuration, not the method.

## Dependencies

`numpy`, `scipy`, `pennylane` + `pennylane-lightning` (lightning backend), `quimb` (MPS
backend), `biopython` (PDB parsing), `matplotlib` (plots), `openmm` (Amber energy model,
and currently an unconditional import for `main.py`). All pinned in `requirements.txt`.
