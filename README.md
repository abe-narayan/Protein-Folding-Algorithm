# Peptide Structure Prediction with a Global CVaR-VQE

Sequence-only structure prediction for short peptides, cast as a global Variational
Quantum Eigensolver with a CVaR objective. One circuit covers the entire peptide (no
block decomposition), and no native structure is read during the search.

The comparison the repo is built for: how do quantum and classical optimizers compare in
scalability as peptide length grows? Four arms — lattice-VQE, torsion-VQE, simulated
annealing, random search — search the same space against the same energy model, at equal
cost measured in unique energy evaluations.

## Read this first

The energy model was rewritten on 2026-07-28 and **the results below are from before
that**. They are kept because they are what was measured, but they describe a landscape
that `docs/energy-model-diagnosis.md` proves is anti-correlated with correctness:
Spearman(energy, RMSD) = −0.351, so optimising it did *worse than random sampling*.

`docs/rmsd-accuracy-fixes.md` describes the rewrite. **Nothing in it has been executed** —
it was written on a machine with no Python interpreter. The first thing to run is:

```bash
python main.py --validate --energy-quality
```

`--energy-quality` reports whether the objective correlates with distance-to-native at
all. If it is negative, no RMSD from this repo means anything in either direction, and
that must be fixed before anything else is worth running.

## Findings (pre-rewrite, legacy energy model)

All evidence is chignolin at one length, three seeds, on the legacy energy model. That is
narrow, and none of it should be read as a general claim about VQE.

- On the lattice arm, annealing reached the global optimum in 185 unique energy
  evaluations; the VQE needed ~16 358; random search did not find it in 20 000.
- That optimum was a straight extended chain, 9.85 Å CA-RMSD from native. It has since
  been traced to a **unit error**: the lattice returned lattice units into an
  Ångström-parameterised energy model, which made the steric term charge 46× more for a
  turn than for going straight. Term-by-term accounting reproduces 9.847041 from two
  artifacts, with the only two fold-aware terms identically zero. See
  `docs/rmsd-accuracy-fixes.md` §1.
- On the torsion arm at matched cost, the VQE returned the worst energy of the three arms
  (−1.306 vs −2.282 and −2.153) with about five times the seed-to-seed spread. Random
  search beat it on every seed.
- The VQE had the best CA-RMSD on that arm (4.33 Å vs 5.17 and 5.49). **This was not a
  search win.** On an anti-correlated landscape, worse optimisation produces better
  structures — the RMSD column rewarded whichever arm failed hardest.
- The annealing arm never actually annealed: its temperature schedule was a function of
  the step index while the shared budget terminated the run, so it ended at 92 % of its
  starting temperature. See `docs/rmsd-accuracy-fixes.md` §9.
- Length scaling is analytic only. Qubit count and memory are derived; nothing was
  measured across lengths.

## What's here

Two energy models behind one Hamiltonian interface, both mixing in
`budget.BudgetedEnergyModel` so they account cost identically:

- **legacy** (`energy_terms.py`, `hamiltonian.py`) — 11-term knowledge-based score:
  steric (4.0), hbond_longrange (3.0), coop_helix (2.0), coop_sheet (2.0), contact (1.0),
  hbond_local (1.0), electrostatic (1.0), aromatic (0.8), solvation (0.5),
  compactness (0.4), torsion (0.15). Burial uses the Fauchère–Pliška octanol/water scale
  (Kyte–Doolittle calls the aromatics hydrophilic, which makes an aromatic-core peptide
  unfoldable by construction); H-bonds are separate local and long-range terms, neither
  normalised by chain length; `coop_helix` / `coop_sheet` reward *runs* of H-bonds —
  consecutive n-turns and consecutive antiparallel ladder rungs — which is the one thing an
  additive pairwise potential cannot express and the reason a correct register used to score
  the same as a scattered tangle; steric is a soft-sphere over all backbone heavy atoms, Cβ,
  and aromatic ring atoms; the aromatic term is an orientation-dependent π-stacking
  potential on real ring centroids and normals. The seven weights in
  `energy_terms.FREE_WEIGHTS` are meant to be set by `energy_quality.calibrate_weights`
  over a *set* of training sequences, never by hand.
- **amber** (`amber_hamiltonian.py`) — OpenMM ff14SB with implicit-solvent GB,
  backbone-restrained capped minimization, and a collapse floor sending GB
  Coulomb-collapse artifacts to `+inf`. Honours the encoded χ₁. Not yet run.

Two representations (`representations.py`):

| rep | qubits | config space | expresses |
| --- | --- | --- | --- |
| torsion — per-residue (φ, ψ) from a **residue-class-specific** 4- or 8-state library (Gly / Pro / pre-Pro / general), plus **one χ₁ bit per aromatic** (F/Y/W); backbone and Cβ by NeRF, sidechains by `sidechains.py` | `2N + A` / `3N + A` | `n_states ** N · 2 ** A` | helix, strand, turns, aromatic stacking geometry; chiral |
| lattice — tetrahedral CA-only trace, in **Ångströms** (CA–CA = 3.80) | `2(N−1)` | `4 ** (N−1)` | β approximately, turns coarsely; achiral |

`A` is the number of F/Y/W residues. Chignolin is 20 → **22 qubits**; trpzip 24 → 28. For
comparison, moving a 10-mer from 4 to 8 backbone states costs **+10** qubits and buys
0.04 Å of ceiling. χ₁ moves no CA atom, so the representation ceiling and every backbone
metric are unaffected — the bits cost qubits and nothing else.

Index 0 is the helical state and index 1 the extended state in every library, and χ₁
rotamer 0 is the value in `sidechains.CHI_ANGLES`, so `bitstring_from_states([0]*n)` /
`[1]*n` remain meaningful for any composition and reproduce what the repo always built.

Two VQE backends: **lightning** (`vqe.py`, dense statevector, ~30 qubits) and **MPS**
(`mps/`, quimb, for larger registers). Both use the same ansatz — `layers` × (RY on every
wire, CNOT chain, ring closure) — optimized against a CVaR-α objective over sampled
bitstrings.

The optimizer defaults to **SPSA**, not COBYLA. The objective samples `shots` bitstrings per
call, so the same parameters return different values; COBYLA is a trust-region method that
requires a reproducible objective and shrinks its region whenever noise makes predicted and
actual improvement disagree. SPSA's convergence theory is for noisy evaluations and it costs
two evaluations per iteration regardless of dimension, against COBYLA's n+1 just to build a
simplex (89 at 88 parameters). COBYLA stays selectable via `--optimizer COBYLA`, and
`--hparams` runs both under one shared budget. See `docs/rmsd-accuracy-fixes.md` §15 — the
SPSA implementation had three defects of its own that had to be fixed first, including a
gain schedule calibrated to a horizon the budget never reaches (the same defect that made
annealing never cool).

## Quickstart

```bash
pip install -r requirements.txt
python main.py --validate           # 78-test suite (geometry, energy, cooperativity,
                                    # objective quality, chi1/stacking, CVaR, VQE,
                                    # optimizer, budget, leakage, amber);
                                    # 73 run without openmm
python main.py --energy-quality     # is the objective worth minimising at all?
python main.py --calibrate-weights --proteins 1UAO,5AWL,1LE0,1LE1,1L2Y,2JOF
python main.py --scaling            # analytic qubit / memory / config-space report
python main.py --predict --sequence GYDPETGTWG
python plot_structures.py results/GYDPETGTWG_prediction.pdb
```

Every legacy-model command runs without OpenMM: `amber_hamiltonian` is imported lazily
inside the amber branch, and `validation.run_all()` skips the five amber tests with a
notice.

`--predict` prints the VQE solution, distribution stats, a weighted energy breakdown
against helix/extended references, and — if a cached PDB matches the sequence — the native
breakdown, the NMR ensemble spread, and the per-residue torsion table.

## Repository layout

| Path | Purpose |
| --- | --- |
| `protein_geometry.py` | Backbone/NeRF geometry, Kabsch/RMSD (+ ensemble RMSD), PDB IO, DSSP, torsions |
| `sidechains.py` | Heavy-atom sidechains (G A S T D E N K P **F** Y W), χ₁-overridable, ring atom sets |
| `representations.py` | Residue-class torsion libraries, χ₁ encoding, tetrahedral lattice, `make_representation` |
| `energy_terms.py`, `hamiltonian.py` | Knowledge-based energy and `FoldingHamiltonian` |
| `energy_quality.py` | **Is the objective worth minimising?** Spearman / enrichment / native percentile, plus variance-balanced weight calibration |
| `amber_hamiltonian.py`, `amber_obc2.py` | OpenMM ff14SB + GBn2 with collapse floor; OBC2-native variant |
| `budget.py` | Shared evaluation-budget accounting, wired into both Hamiltonians |
| `vqe.py` | CVaR objective, global ansatz, lightning VQE loop |
| `mps/` | `MPSSampler` and MPS driver (now budget-aware) |
| `classical_baselines.py` | Random search, simulated annealing, exhaustive search |
| `evaluation.py` | Representation ceiling, predicted-vs-native metrics, ensemble/core RMSD |
| `dataset.py` | PDB download/cache, NMR ensembles, alignment-identity clustering |
| `experiments.py` | Experiment drivers and CSV IO |
| `validation.py` | 60-test correctness / leakage / budget / objective-quality suite |
| `main.py` | CLI entry point |
| `plot_structures.py` | Plots a prediction PDB against its native ensemble |
| `diagnose_energy_model.py` | Exhaustive landscape diagnosis with per-term attribution |
| `run_8state_seed0.py` | **Pinned** 8-state OBC2 chignolin seed-0 run (legacy library) |
| `run_sa.py` | Classical arm (simulated annealing) on one target or `ALL`, with CA-RMSD |
| `rmsd_of.py` | CA-RMSD, energy and secondary structure of a given bitstring |
| `pdbs/`, `results/`, `docs/` | Cached PDBs; generated data; design notes |

The suite pins determinism under fixed seeds, absence of native leakage during
optimization, that the VQE is global rather than blockwise, that it does not enumerate,
that it matches exhaustive search on an enumerable system, that the shared budget is a
hard ceiling, that χ₁ moves rings but never a CA atom, that stacking beats
interpenetration, that a contiguous H-bond ladder beats the same bonds scattered, that the
native is scored with the same aromatic geometry as a prediction, and — the group that was
missing entirely — that the objective correlates positively with distance to native.

## Status

Open gaps:

1. **Nothing in the 2026-07-28 accuracy pass has been run.** It was written without a
   Python interpreter available. `--validate` is the gate.
2. The Amber and MPS paths remain unexecuted; neither `openmm` nor `quimb` was installed
   in the environment where this was last touched. The Amber comparison is the one that
   matters.
3. `results/main_comparison.csv` and `results/main_comparison_matched.csv` predate the
   energy-model rewrite and are historical. Regenerate or delete.
4. There is no length sweep. A cost-matched sweep over N is the missing experiment.
5. The VQE has not been retuned for a matched budget. `alpha`, `layers`, `shots` and
   Spall's `a`/`c` gains are all untouched; `--hparams` now carries the arms that would
   settle them, including the optimizer comparison and `shots ∈ {256, 512, 2048}`.
   `shots` is the one most likely mis-set: at 2048 it trades away the iteration count SPSA
   depends on.
9. Sidechains cover **12 of 20** residues (missing R, C, Q, H, I, L, M, V). The legacy
   energy model is unaffected — it uses backbone, Cβ and aromatic rings only — but
   `AmberHamiltonian` raises `NotImplementedResidueError` for any peptide containing them,
   which excludes the trp-cages (1L2Y, 2JOF) from the Amber path.
6. Excluded volume for **non-aromatic** sidechains is still missing (the aromatics now have
   real ring atoms). Histidine has no ring template, so it falls back to a CB proxy. χ₂ is
   not encoded, which matters for Trp. See `docs/rmsd-accuracy-fixes.md` §12 and
   "Deliberately not done".
7. `mps/driver.py` is still a near-copy of `vqe.run_global_cvar_vqe`. It now has budget
   handling and the read-out fix, but the merge into a single `sampler=`-parameterised
   function is outstanding.
8. The long-range H-bond weight defaults to 3.0, which favours hairpins. Run
   `--calibrate-weights` with a helix (`1DU1`) and the trp-cages in the held-out clusters
   before trusting it; the `no_hbond_longrange` ablation arm is there to expose it.

## Dependencies

`numpy`, `scipy`, `pennylane` + `pennylane-lightning`, `quimb` (MPS), `biopython` (PDB
parsing), `matplotlib`, `openmm` (Amber model only — optional). Pinned in
`requirements.txt`.
