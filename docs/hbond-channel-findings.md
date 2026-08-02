# The helix bias is a weight problem after all: `coop_helix` alone inverts the objective

Measurements taken 2026-08-02 on `claude/pfa-helix-bias-root-cause-g0kxto`, after the
N...O interpenetration patch (commit `73bb942`). Every number was produced by running the
code in this repository; `diagnose_hbond_channel.py` reproduces the mechanism tables.

Target is chignolin (1UAO, `GYDPETGTWG`), which rebuilds from its own torsions at 0.18 A.

---

## The headline, and a correction to the standing analysis

Over all 4,194,304 encodable structures, **two single-weight moves — each one inside the
guardrail box `calibrate_weights` already operates in — flip the objective from
anti-correlated to useful**:

| weights | global minimum | top-100 mean RMSD | enrichment | verdict |
|---|---|---|---|---|
| `DEFAULT_WEIGHTS` | -17.018 @ **5.27 A** | 5.075 | **-0.274** | ANTI-correlated |
| `coop_helix` 2.0 -> **0.667** (`base/3`) | -13.125 @ **1.96 A** | 2.187 | **+2.613** | helps |
| `hbond_longrange` 3.0 -> **9.0** (`base*3`) | — | — | **+2.669** | helps |
| best point on a 3^7 guardrail grid | -31.913 @ **1.96 A** | — | **+2.735** | helps |

The global minimum moves from a 5.27 A helix to a **1.96 A hairpin**, and the whole top-100
lands between 1.81 and 2.30 A — against a representation ceiling of 0.97 A and a mean of
4.80 A over the full space.

**This refutes the recorded conclusion that "weight calibration provably cannot fix
chignolin at 4 states", and it refutes the first draft of this document too.** Both rested
on the same false premise, which is worth naming precisely because it is subtle:

> *"The best encodable chignolin (the snapped native, 2.67 A) scores `hbond_longrange` 0.000
> and `coop_sheet` 0.000. Both hairpin channels are exactly zero, so every weight in
> `FREE_WEIGHTS` multiplies zero."*

The snapped native is **not** the best encodable hairpin. It is the best *per-residue
nearest-angle projection* of the native's torsions, which is a different thing and a worse
one. Enumeration finds `0000110000111001000111` at **1.9632 A** — closer to the native than
the snapped native *and* carrying a full cross-strand ladder:

```
  bits   0000110000111001000111    states [0,0,3,0,0,3,2,1,0,1] chi 11
  CA-RMSD 1.9632 A      H.energy -13.1249 (DEFAULT weights)
  hbond_longrange  raw -3.5187   weighted -10.5560
  matched H-bonds  (7,2) (2,7) (1,8)  -- all three cross-strand
  steric 0.0000    coop_helix 0.0000
```

(Both chi settings on that backbone tie at -13.1249, since its aromatic and ring-steric
terms are zero either way; `results/coop_helix_weight_scan_1UAO.json` records the argmin's
choice, `...110`.)

So the hairpin channel is emphatically **not** structurally zero. It is zero *for one
particular structure* that everyone, including this document's first draft, had been
treating as the ceiling on hairpin quality. Every conclusion derived from that structure —
"tolerance cannot help", "the representation is the binding constraint", "softening widens
the gap" — was measuring an artifact of the projection, not a property of the encoding.

**The real gap is 3.89 kcal/mol, not 14.74:**

| | helix (5.27 A) | hairpin | gap |
|---|---|---|---|
| snapped native as hairpin (the old framing) | -17.018 | -2.279 | 14.739 |
| **true best encodable hairpin (1.96 A)** | -17.018 | **-13.125** | **3.893** |

And `coop_helix` is -10.00 of a 3.89 gap — i.e. **the term is more than twice the size of
the entire deficit it creates.** Remove it and the ordering inverts immediately.

---

## 0. State of the N...O patch (verification)

`python main.py --validate` -> **80/80 passed, `ALL PASSED: True`** with OpenMM installed.
Without it the same run prints `75/75 passed, 5 skipped` — the denominator is
`len(results)`, and skipped tests `continue` before `results.append`, so they leave both
sides of the fraction (`validation.py:2036-2050`). Anyone reading `75/75` as "everything
ran" is reading it wrong; five AMBER gates were inert, and **one of them turned out to be
broken** — see section 5. Install OpenMM before trusting a green suite here.

The suite is 80 registered tests, not 79, because the patch commit added two — both pass:

```
[PASS] dataset rejects natives it cannot rebuild -- 1B03 rebuilds at 11.07 A (rejected), 1UAO at 0.18 A (kept)
```

Exhaustive enumeration confirms the gate. The interpenetrated -69.85 minimum is gone and
the global minimum is the alpha-helix at exactly **-17.0180**, as predicted:

```
  native ranks #387,224 of 4,194,304 by energy  -> 9.23% score better
  global energy minimum       -17.0180    CA-RMSD 5.27 A
  Spearman(energy, RMSD)      : +0.1629
  mean RMSD all 4.80 | top-100 5.07   -> enrichment -0.27 A  (ANTI-correlated)
```

One number came out worse than projected: enrichment was expected at -0.124 ("no useful
signal") and measured **-0.274**, which crosses `energy_quality`'s own threshold into
**"ANTI-correlated"**. The projection was made `rings=None` on 20 bits.

### Two configuration corrections

**(a) The enumeration is 22 bits / 4,194,304 structures, not `4^10` = 1,048,576.**
`TorsionStateRepresentation` defaults to `chi_bits=True` and chignolin has two encoded
aromatics, so two chi bits sit on top of the 20 backbone bits.

**(b) chi1 is part of the bitstring, and `bitstring_from_states([0]*n)` silently zeroes it.**

| all-helix backbone, chi = | energy |
|---|---|
| (0,0) and (1,0) | **+24.775** |
| (0,1) and (1,1) | **-17.018** |

At Trp9 rotamer 0 the indole ring lands on the residue-4 backbone (atom pairs at 1.02-1.65 A,
inside covalent distance) for +41.79 weighted steric. The recorded -17.018 is correct, but
only at chi=(x,1).

---

## 1. Mechanism: why the *snapped native* scores zero, and why that misled everyone

Worth keeping, because it explains the misleading number rather than excusing it.

`hbond_terms` allows each acceptor exactly one bond and matches greedily by **raw** DSSP
energy, while `energy_components` pays long-range pairs `w=3.0` against local `w=1.0`. On
the snapped native at `c <= 0.5`:

```
    E   -1.395  donorN5  accO2   dON  3.201  |i-j|=3  local
    E   -0.966  donorN7  accO2   dON  3.951  |i-j|=5  LONG-RANGE
  greedy match KEEPS: ((5, 2),)
  DROPPED  donorN7 accO2: acceptor O2 already used
```

The matcher discards a pair worth **-2.898** weighted to keep one worth **-1.395**. The
exclusion rule and the objective disagree. Two fixes, both **exactly inert on the helix**:

- **(b) optimal weighted matching** — `linear_sum_assignment` on the weighted contribution.
- **(c) acceptor capacity 2** — a C=O has two lone pairs; bifurcated H-bonds are real, so
  one-bond-per-acceptor is itself the physical error. Donors stay capped at 1.

(c) is preferable on layering grounds: (b) must know the weights to sort, making the
matched-pair list a function of the weight vector — and that list feeds `coop_helix_term`
and `coop_sheet_term`, so `calibrate_weights` would be silently changing which bonds exist.
(c) keeps the pair list a pure function of geometry. Cost of (c): it touches 5.3% of random
encodable structures against 0.7% for (b).

The population statistic from the earlier pass is confirmed — greedy matching is suboptimal
on only **0.0-0.7%** of random structures — but note how badly a rare defect can mislead
when it happens to land on the one structure being used as a reference point.

---

## 2. Softening the H-bond channel: helps, once measured on the right structure

The first draft of this document concluded that softening **widens** the gap. That was
measured against the snapped native and is wrong. Re-measured against the true best
encodable hairpin:

| matcher | c | helix | hairpin (1.96 A) | GAP | vs baseline | bonds h/p |
|---|---|---|---|---|---|---|
| shipped | 1.0 | -17.018 | -13.125 | **3.893** | +0.000 | 6/3 (3 lr) |
| shipped | 0.5 | -20.018 | -17.669 | 2.349 | **-1.544** | 6/4 (3 lr) |
| shipped | 0.0 | -23.018 | -22.669 | 0.349 | **-3.544** | 6/4 (3 lr) |
| cap2 | 0.5 | -20.018 | -17.697 | 2.321 | -1.573 | 6/5 (3 lr) |
| cap2 | 0.0 | -23.018 | -23.197 | **-0.179** | **-4.073** | 6/5 (3 lr) |

Softening closes the gap monotonically, and with acceptor-capacity-2 at `c = 0` the hairpin
**wins outright**. For comparison, the same table computed against the snapped native showed
the gap *growing* from 14.739 to 19.739 — a sign error produced entirely by the choice of
reference structure.

The underlying arithmetic is unchanged and is the thing to remember: the desolvation offset
is refunded **once per matched bond**, so it pays each structure in proportion to the bonds
it holds. The helix holds six local bonds (refund `6 x 1.0`). A hairpin holding three
cross-strand bonds gets `3 x 3.0 = 9.0` and wins the exchange; one holding a single local
bond gets `1.0` and loses it. **Whether softening helps depends on bond count, and the
break-even is two cross-strand bonds** — which real encodable hairpins clear and the
projected one does not.

---

## 3. What calibration can actually reach

Each free weight swept alone to its guardrail limits (`base/3`, `base*3`), enrichment over
the full enumeration:

| weight | base | lo | enr(lo) | hi | enr(hi) |
|---|---|---|---|---|---|
| aromatic | 0.80 | 0.267 | -0.274 | 2.400 | -0.274 |
| torsion | 0.15 | 0.050 | -0.274 | 0.450 | -0.285 |
| compactness | 0.40 | 0.133 | -0.274 | 1.200 | -0.274 |
| solvation | 0.50 | 0.167 | -0.274 | 1.500 | -0.274 |
| **hbond_longrange** | 3.00 | 1.000 | -0.274 | **9.000** | **+2.669** |
| **coop_helix** | 2.00 | **0.667** | **+2.613** | 6.000 | -0.274 |
| coop_sheet | 2.00 | 0.667 | -0.274 | 6.000 | -0.274 |

Only two weights matter, and each is sufficient on its own. Note `coop_sheet` does nothing
in either direction: even the 1.96 A hairpin scores `coop_sheet` exactly 0.000, because its
three rungs `(7,2) (2,7) (1,8)` never satisfy the `(i+2, j-2)` step condition. That term
remains untested by this target.

The best point on a full 3^7 grid reaches **+2.735** at `hbond_longrange=9.0,
coop_helix=0.667, coop_sheet=0.667, solvation=0.167, aromatic=2.400, torsion=0.050,
compactness=0.133` — barely better than moving `coop_helix` alone, which is the honest
summary: **this is a one-parameter problem.**

### Why the ceiling asymmetry produces this

`coop_helix` fires on only **1.56%** of all structures (98.44% score exactly 0.000) — it is
a narrow, deep reward switched on solely by helical topology, worth up to `-10.0` weighted
at `n = 10`. Against a true gap of 3.89 it is roughly 2.6x more than enough to dominate.
The ceiling arithmetic from the earlier pass stands (`coop_helix` max `n-5` vs `coop_sheet`
max `~2*(ceil((n-2)/4)-1)`, i.e. 5 vs 2 at n=10); what changes is the conclusion drawn from
it — the asymmetry is real, and it is also **within calibration's reach**.

---

## 4. Aromatic ring excluded volume is anti-native at 8 states

Independent of the above, and a genuine blocker for the obvious "use a finer library" move.
Sampling near-native encodable structures, taking the **best** of both chi rotamers, and
isolating `steric(rings) - steric(no rings)`:

| CA-RMSD bin | n (8 states) | ring-only steric | frac >20 |
|---|---|---|---|
| 0-1.0 | 50 | **65.046** | **1.00** |
| 1.0-1.5 | 161 | 32.025 | 0.61 |
| 1.5-2.0 | 266 | 22.319 | 0.39 |
| 2.0-2.5 | 265 | 12.387 | 0.20 |
| 3.5-5.0 | 150 | 0.024 | 0.00 |
| 5.0-99 | 96 | **0.000** | 0.00 |

Perfectly monotonic over 1,356 backbones: the closer to native, the larger the penalty.
Confirmed on a dedicated sample — of **81 distinct 8-state backbones under 1.5 A, not one is
clash-free at its best chi** (mean 41.2). The 8-state snapped native sits at 1.478 A and
scores **+19.295**, of which +21.383 is ring steric (0.030 without rings).

Chignolin's fold *is* the Tyr2/Trp9 packing, so approaching the native means bringing those
rings together; one bit per aromatic (two chi1 rotamers, chi2 fixed by a rigid template)
cannot express the real packing, so they interpenetrate.

Mild at 4 states (peak 4.66), and the 4-state ceiling structure at 0.968 A has a completely
clash-free chi — so this does **not** explain the 4-state bias. It means a finer backbone
library without more sidechain freedom would make things worse. The earlier ruling-out
("real native, real rings: steric exactly 0.0") is correct for the deposited native and
should be narrowed to that case rather than withdrawn.

---

## 5. `test_amber_native_below_helix` passes on the clashing rotamer

Running the suite with OpenMM installed (80/80, no skips) surfaced this. The AMBER arm uses
`_AMBER_SEQ = "GYDPETGTWG"` — chignolin, the same target — and the test builds its
comparison helix with `rep.bitstring_from_states([0] * n)`, which zeroes both chi bits.
That is the same silent chi-zeroing described in section 0(b), and here it decides the
result:

| helix chi | AMBER energy | gap vs native (-315.90) | test outcome |
|---|---|---|---|
| **(0,0)** ← what the test uses | -199.44 | -116.46 | **passes** |
| (1,0) | -200.86 | -115.04 | passes |
| (0,1) | **-343.00** | **+27.10** | *fails* |
| (1,1) | **-343.70** | **+27.80** | *fails* |

The assertion `e_native < e_helix` holds only because the helix it is compared against has
Trp9's indole ring buried in the residue-4 backbone. Built at either non-clashing rotamer,
**the helix scores 27.8 kcal/mol below the native** and the assertion is false. The gate
therefore certifies nothing about topology: it would pass unchanged if AMBER preferred a
helix by any margin whatsoever — which, as configured, it does.

Two consequences, and they are separate:

**(a) The gate is invalid as written.** It should build its helix at the minimum over chi,
the way `hamiltonian.FoldingHamiltonian.energy_from_coords` already does for natives via
`MAX_NATIVE_CHI_SCAN_BITS`. Note that fixing it makes it **fail** — so this is not a
one-line correction to slip in, it is a decision about whether the AMBER arm currently
earns a green gate. Left unchanged here deliberately; turning the suite red is the owner's
call, not a side effect of a documentation pass.

**(b) The stated reason for not scanning rotamers is measurably false.**
`amber_hamiltonian.energy_from_coords` says:

> *"the minimizer relaxes the (unrestrained) sidechains anyway, so the starting rotamer
> matters much less here"*

The four chi assignments of one fixed backbone span **144.26 kcal/mol** (-199.44 to
-343.70) after the restrained minimization has run. A tryptophan ring inside the backbone
is separated from its relaxed position by a barrier, not a gradient, and 50 steps of local
minimization at tolerance 2.0 does not cross it. Both `energy` and `energy_from_coords`
route through the same `_evaluate`, so this is not a path mismatch — the comparison is fair
and the starting rotamer simply matters a great deal.

**On the physics claim, stated carefully.** That AMBER-with-OBC2 ranks an ideal helix
27.8 kcal/mol below deposited chignolin should not be read as "chignolin is really a helix".
Chignolin's experimental folding free energy is on the order of 1-2 kcal/mol, so a 27.8
kcal/mol margin is far outside the range where a 50-step restrained minimization of an
idealised build against an NMR model is thermodynamically meaningful. What it does say is
that **the AMBER arm, as configured, has the same helix preference as the knowledge-based
model** — so it cannot serve as the independent physics check on this bias that its name
implies. Fixing `coop_helix` repairs the knowledge-based objective and leaves this
untouched.

---

## 6. A reporting bug in `diagnose_energy_model.py` (fixed here)

The diagnosis printed `closest structure to native   44.4613   0.97`. That +44.46 is an
artifact: **chi1 does not move CA atoms**, so all four chi variants of a backbone have
byte-identical CA-RMSD and `np.argmin(rmsds)` broke the tie arbitrarily, landing on chi=(0,0):

| ceiling backbone, chi = | energy | ring-only steric |
|---|---|---|
| (0,0) | +44.615 | 47.749 |
| **(1,0)** | **-3.491** | **0.000** |
| (0,1) / (1,1) | +23.720 / +24.144 | 27.267 |

The true energy at 0.97 A is **-3.491**, not +44.46. Fixed by breaking the RMSD tie on
energy. Reporting only; no energy, search or weight is affected.

---

## Recommendations

1. **Run `--calibrate-weights` and `--energy-quality` now.** The blocker is gone: this
   objective is fixable inside the existing guardrail box, and `calibrate_weights` can
   already reach the values that fix it. This was believed impossible and is not.

2. **Do not hand-set `coop_helix` to 0.667.** That is the target-shaped fix the project has
   correctly refused before, and section 3 is a single-target measurement. What has changed
   is that calibration is now known to be *capable*; whether it *lands* there must come from
   a train split. Check `split_by_cluster` first — the three 20-mers (1DU1, 1L2Y, 2JOF) are
   the targets a helix-biased objective flatters, and stacking them into train would teach
   the calibrator the opposite lesson.

3. **Decide what to do about `test_amber_native_below_helix`** (section 5). It is green and
   it is not testing what it claims; corrected, it fails. The AMBER arm cannot currently be
   cited as independent physics corroboration on helix bias.

4. **Add a topology gate before calibrating**, so the metric can see this failure class.
   `geo.assign_secondary_structure` and the SS-agreement metric already exist; only the
   assertion is missing. Without it a fully green suite can coexist with a one-fold model —
   which is exactly what happened.

5. **Never use the snapped native as a proxy for "the best encodable structure" again.**
   That single substitution produced three independent wrong conclusions across two
   analyses, including a sign error on whether softening helps. It is a projection, not an
   optimum: on chignolin it is 0.7 A worse in RMSD and 10.8 kcal/mol worse in energy than a
   structure the encoding can express. Where "what can the encoding do" is the question,
   enumerate or search — do not project.

6. **Fix the aromatic representation before attempting 8 states** (section 4), and treat the
   acceptor-capacity rule as a deliberate modelling decision rather than a side effect of a
   greedy loop (section 1).

7. **`desolvation_cost` should become a calibrated parameter** (range `[0.0, 1.5]`). It is
   the only solvent competition in the model and a hardcoded `1.0` no calibration has seen
   is indefensible; section 2 now shows it moves the gap in the *helpful* direction on real
   encodable hairpins.
