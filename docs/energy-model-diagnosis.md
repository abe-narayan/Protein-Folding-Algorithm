# The legacy energy model is anti-correlated with correctness

**Date:** 2026-07-28
**Method:** exhaustive enumeration of all 4^10 = 1,048,576 representable structures for
chignolin (1UAO, `GYDPETGTWG`) at 4 torsion states, plus a term-variance decomposition
over 4,000 random structures. Reproduce with `python diagnose_energy_model.py`.
**Raw output:** `results/energy_model_diagnosis.json`

---

## The finding

`docs/evaluation-budget.md` records that the legacy model scores native worse than
predictions and concludes the arms are "finding the true minimum of a wrong Hamiltonian".
That is a claim about the landscape, not about any one search, so it can be settled
exactly rather than inferred. At N=10 the configuration space is small enough to
enumerate completely.

|  | energy | CA-RMSD |
| --- | ---: | ---: |
| native (real backbone) | 5.5135 | — |
| native (snapped to the encoding) | 1.1827 | 2.32 Å |
| **global energy minimum** | **−2.4496** | **5.54 Å** |
| closest representable structure to native | 0.4106 | 0.96 Å |

**126,348 of 1,048,576 structures — 12.05% — score better than native.**

The global minimum of this energy function sits **5.54 Å** from the native fold. The
structure that actually is closest to native (0.96 Å) scores +0.41, worse than 12% of the
space. Optimizing this function to convergence lands you nowhere near the answer.

## It is worse than uninformative — it is inverted

```
Spearman(energy, RMSD)        : -0.3505
mean RMSD, all structures     : 4.91 A
mean RMSD, 1000 lowest-energy : 5.18 A
mean RMSD, 100 lowest-energy  : 5.16 A
   enrichment                 : -0.25 A
```

The rank correlation between energy and RMSD-to-native is **negative**. Lower energy is
associated with being *further* from the native fold. The 100 lowest-energy structures
average 5.16 Å against a population mean of 4.91 Å, so **a search that optimizes this
function does worse than drawing structures at random.**

This retrospectively explains the result in `docs/evaluation-budget.md` that looked
paradoxical — the VQE getting the *best* CA-RMSD (4.33 Å) while being clearly the *worst*
optimizer (mean energy −1.306 vs SA's −2.282). That was not a quantum result. On an
anti-correlated landscape, worse optimization produces better structures, and the RMSD
column rewards whichever arm failed hardest. Ranking arms by RMSD here is ranking them by
how badly they optimized.

## Why: the objective is dominated by a term with no tertiary-structure content

Weighted contribution of each term across 4,000 random structures:

| term | weight | mean | std | share of variance |
| --- | ---: | ---: | ---: | ---: |
| steric | 4.00 | 1.469 | 6.209 | **87.3 %** |
| torsion | 1.00 | −0.104 | 2.272 | **11.7 %** |
| solvation | 0.30 | 3.727 | 0.566 | 0.7 % |
| contact | 1.00 | 0.138 | 0.293 | 0.2 % |
| compactness | 0.05 | 0.122 | 0.146 | 0.0 % |
| hbond | 1.00 | −0.026 | 0.097 | 0.0 % |
| electrostatic | 0.50 | 0.030 | 0.011 | 0.0 % |

Two things follow.

**1. Steric is a filter, not a discriminator.** It carries 87 % of the variance across
random structures, but it is *zero* for both native and the global minimum — once a
structure has no clashes it contributes nothing. So among the feasible structures a search
actually explores, steric is flat.

**2. Among feasible structures, torsion is the entire objective — and it is separable.**
`torsion_term` is `sum(rama_penalty(seq[i], phi[i], psi[i]) for i in range(n))`: a sum of
independent per-residue terms with **no coupling between residues whatsoever**. Minimizing
it means independently choosing each residue's most favourable Ramachandran state. The
result is a structure that is locally ideal at every residue and globally meaningless —
which is exactly what the global minimum is: torsion −5.968 against native's +0.361, at
5.54 Å.

The per-term attribution at the global minimum confirms it: of native's +7.963 total
penalty relative to the minimum, **+6.329 is torsion** and +1.908 solvation. Contact
contributes +0.101 and hbond −0.106.

**The only two terms that know anything about the three-dimensional fold — contact and
hbond — have standard deviations of 0.293 and 0.097 against torsion's 2.272.** They are
outvoted roughly 8:1 and 23:1. The energy function has essentially no tertiary-structure
signal in it.

### A contributing bug

`energy_components` returns `"hbond": (hb_local + hb_lr) / max(1, n)`. The hydrogen-bond
term — the main long-range structural signal, and the thing that makes a beta hairpin a
beta hairpin — is **divided by the number of residues**. At N=10 that is a 10× suppression
of the term that most needs to be heard. Nothing else is normalized this way.

## What this means for the results in the repo

- Every legacy-model comparison measures **search efficiency on an anti-correlated
  landscape**. That is a real and now cost-matched measurement, but it says nothing about
  structure prediction.
- **No RMSD number from the legacy model should be quoted as a result.** Not in either
  direction: a good RMSD indicates a failed search, and a bad RMSD indicates a successful
  one.
- The energy column remains meaningful, because it is what every arm is actually
  minimizing. The cost-matched comparison of *optimizers* stands.

## What to do about it

In rough order of defensibility.

1. **Run the comparison on the amber model.** It is physics-based rather than a weighted
   sum of hand-tuned heuristics, and `docs/evaluation-budget.md` already identifies it as
   "the one that counts". Blocked only on `openmm` not being installed here.
2. **Fix the `/n` on the hbond term.** This is a defensible bug fix independent of any
   tuning: no other term is normalized by length, and it suppresses the main structural
   signal by 10×.
3. **Give the coupled terms authority over the separable one.** Any weighting where
   `torsion` has ~8× the spread of `contact` cannot produce a fold. Either weight torsion
   down sharply or weight contact/hbond up by roughly an order of magnitude.

⚠️ **The trap to avoid.** Do not tune weights until native wins on chignolin. With seven
weights and one target structure you can always succeed, and you will have fit the answer
rather than a model. `dataset.py` already provides `split_by_cluster` and
`check_no_cluster_leak` for exactly this reason — any re-weighting must be fit on a train
split and reported on held-out clusters, or it is not a result.

## Reproducing

```
python diagnose_energy_model.py --protein 1UAO --workers 8     # ~166 s on 8 cores
```

Enumeration is exact, so this is not a sampling estimate. The same script runs for any
sequence whose configuration space is enumerable (N ≤ 11 at 4 states on this machine);
beyond that it would need sampling and the numbers become estimates.
