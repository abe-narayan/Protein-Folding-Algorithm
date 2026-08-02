# Why the objective does not predict beta structure

Measurements taken 2026-08-02 on `claude/pfa-helix-bias-root-cause-g0kxto`, after the
N...O interpenetration patch (commit `73bb942`). Every number here was produced by running
the code in this repository. Three scripts reproduce the tables:
`diagnose_hbond_channel.py`, `diagnose_calibration.py`, `diagnose_helix_bias.py`.

**Read the corrections log at the bottom before trusting anything in an earlier revision of
this file.** Four claims made during this investigation were retracted after a control was
added. They are listed there with the control that killed each one, because the failure
pattern is more useful than any individual number: every one came from a measurement that
skipped a filter the repo itself applies, or that generalised from a single protein.

---

## The finding, controlled

Chignolin, all 4,194,304 encodable structures, restricted to the 2,574,772 that are
**clash-free** (`steric == 0`):

| set | mean strand | mean CA-RMSD | mean energy |
|---|---|---|---|
| 200 most beta-rich | 0.207 | **3.32 A** | **-2.41** |
| 200 lowest-energy | ~0.00 | **4.69 A** | **-13.67** |

The beta-rich structures are **1.37 A closer to the native** and the objective scores them
**11.26 kcal/mol worse**. Among structures the model itself certifies as physically valid,
it ranks the more native-like ones down. That is the defect, stated without a proxy.

It is one of **two independent blockers**, and both have to be fixed:

**1. Sampling.** Beta geometry is vanishingly rare in feasible space:

| strand fraction | share of clash-free structures | expected hits in a 4000-draw sample |
|---|---|---|
| > 0.1 | 0.780 % | 31.2 |
| > 0.2 | 0.043 % | **1.7** |
| > 0.3 | 0.003 % | **0.1** |
| > 0.4 | 0.000 % | 0.0 |

A sampler drawing 4000 structures finds no beta because there is none to find, not because
the objective rejected it. The VQE and SA arms both sample. **No reweighting can rank a
structure that is never generated**, so search/representation work is not optional here.

**2. Scoring.** The table at the top: when beta *is* available, the objective prefers
structures further from the native.

Attack scoring first -- it is measurable on the enumeration without touching the search,
and the metric is not a proxy: *does the -11.26 kcal/mol gap between the beta-rich and
lowest-energy sets shrink?*

---

## The output is near-constant across sequences

`diagnose_helix_bias.py`, clash-free structures only, top 100 of 4000 by energy:

| pdb | native H | native E | pred H | pred E | bias |
|---|---|---|---|---|---|
| 1UAO | 0.00 | 0.40 | 0.31 | 0.02 | **+0.31** |
| 5AWL | 0.00 | 0.40 | 0.21 | 0.01 | +0.21 |
| 2EVQ | 0.00 | 0.33 | 0.21 | 0.01 | +0.21 |
| 1E0Q | 0.00 | 0.35 | 0.19 | 0.01 | +0.19 |
| 1LE3 | 0.00 | 0.38 | 0.15 | 0.00 | +0.15 |
| 1LE0 / 1LE1 / 1J4M | 0.00 | 0.14-0.33 | 0.11 | 0.01 | +0.11 |
| 1DU1 | 0.35 | 0.00 | 0.14 | 0.00 | **-0.21** |
| 1V4Z | 0.76 | 0.00 | 0.27 | 0.00 | **-0.49** |
| 1L2Y | 0.70 | 0.00 | 0.19 | 0.01 | **-0.51** |
| 2JOF | 0.75 | 0.00 | 0.20 | 0.01 | **-0.55** |

`pred H` sits at **0.11-0.31 regardless of whether the native is 0.00 or 0.76 helix**. The
objective emits roughly the same structural signature for every sequence. "Helix bias"
describes the top half of the table only; on the helical targets it *under*-calls helix by
up to 0.55.

`pred E` is 0.00-0.02 everywhere against a native mean of 0.223 -- but see the sampling
blocker above before reading that as a scoring failure. The feasible pool's own mean strand
is 0.002, and the top-100 at 0.009 is a **4.5x enrichment**: the objective mildly prefers
what little beta exists.

**Aggregates hide all of this.** Mean helix bias is +0.055 unfiltered and -0.031
filtered -- both ~0 -- while per-peptide values run +0.31 to -0.55. Any mean-based gate
passes this model. Report per peptide.

---

## Calibration makes it worse

`diagnose_calibration.py`. Calibrated on 7 train clusters, evaluated on 4 held-out
(1L2Y, 2JOF, 2EVQ, 1J4M):

| max_ratio | solvation | compact | hb_lr | held-out Spearman | enrichment | verdict |
|---|---|---|---|---|---|---|
| **(default)** | 0.500 | 0.400 | 3.00 | **+0.0552** | **+0.94** | **usable** |
| 3 (shipped) | 0.167 | 0.133 | 9.00 | -0.1000 | +0.54 | NOT usable |
| 10 / 30 / 100 / 1000 | 0.078 | 0.072 | 13.36 | -0.1675 | +0.40 | NOT usable |

`DEFAULT_WEIGHTS` beats calibrated on **4/4** held-out targets, on both metrics, and is
positive on 3/4. **Do not run `--calibrate-weights` and then predict with the result.**

`calibrate_weights` matches `weight * std(term)` to the median pinned term -- it balances
*variance*, and no native enters it (by design; `test_calibration_is_sequence_set_derived`
enforces that). Variance balance is orthogonal to "low energy means low RMSD", and on this
term set it is anti-correlated with it. `max_ratio` saturates at 10, so the guardrail was
never binding; the criterion is wrong.

The mechanism, from the train-split term statistics:

| term | std | share of discriminating variance |
|---|---|---|
| solvation | 8.637 | **54.6 %** |
| compactness | 9.333 | **40.8 %** |
| *(all topology-aware terms)* | | **2.4 %** |
| hbond_longrange | 0.050 | 0.1 % |
| aromatic | 0.051 | 0.0 % |
| coop_sheet | 0.000 | 0.0 % |

The objective is ~95 % "be compact". Random torsions essentially never form a hydrogen
bond, so every topology term is numerically zero across the sampled ensemble; the
calibrator reads their tiny std as "underweighted" and drives them to their ceilings,
amplifying rare spurious values, while suppressing the two terms carrying what real signal
exists.

---

## `test_amber_native_below_helix` passes on a clashing rotamer

With OpenMM installed the suite is 80/80 (without it, `75/75 passed, 5 skipped` -- skipped
tests leave *both* sides of the fraction, so five gates were silently inert). One is broken.

`_AMBER_SEQ` is chignolin, and the test builds its comparison helix with
`bitstring_from_states([0] * n)`, which zeroes both chi bits:

| helix chi | Amber energy | gap vs native (-315.90) | |
|---|---|---|---|
| **(0,0)** <- what it uses | -199.44 | -116.46 | **passes** |
| (1,0) | -200.86 | -115.04 | passes |
| (0,1) | **-343.00** | **+27.10** | *fails* |
| (1,1) | **-343.70** | **+27.80** | *fails* |

The assertion holds only against a helix with Trp9's indole ring buried in the residue-4
backbone. Built properly the helix wins by 27.8 kcal/mol. The gate would pass no matter how
strongly Amber preferred a helix, so the Amber arm is **not** the independent physics check
its name implies. Left unfixed deliberately: correcting it makes the suite red, which is the
owner's call.

It also falsifies `energy_from_coords`'s stated reason for not scanning rotamers ("the
minimizer relaxes the sidechains anyway"): four chi assignments of one backbone span
**144.26 kcal/mol** after minimization. A Trp ring inside the backbone is behind a barrier,
not a gradient.

On the physics: 27.8 kcal/mol against chignolin's ~1-2 kcal/mol experimental folding free
energy is far outside where a 50-step restrained minimization of an idealised build against
an NMR model is meaningful. The claim is about the arm as configured, not about chignolin.

---

## Smaller confirmed results

**chi1 is part of the bitstring.** `bitstring_from_states([0]*n)` silently zeroes it, and on
chignolin rotamer 0 puts Trp9 1.02-1.65 A from the residue-4 backbone (+41.79 weighted
steric). The all-helix scores +24.775 at chi=(x,0) and -17.018 at chi=(x,1). The
enumeration is 22 bits / 4,194,304 structures, not `4^10`.

**A `diagnose_energy_model.py` reporting bug (fixed).** It printed the closest-to-native
structure at +44.46; chi1 does not move CA atoms, so all four chi variants tie on RMSD and
`argmin` picked the clashing one. True value -3.491. Tie now breaks on energy.

**The greedy H-bond matcher is suboptimal on 0.0-0.7 % of structures.** It matches by *raw*
energy while the objective pays long-range pairs 3x, so it can discard a pair worth -2.898
weighted to keep one worth -1.395. Rare, but it landed on the snapped native, which is how
it misled an earlier pass.

**`coop_sheet` works.** On real natives it fires at -3.0 (1LE1, 1LE3), -2.0 (1LE0), -1.0
(2EVQ, 1J4M). Where it reads 0.0 on a hairpin native (1UAO, 5AWL, 1E0Q) the cause is
upstream: the H-bond gate finds only 1-2 cross-strand bonds and a ladder rung needs two
adjacent ones.

**`desolvation_cost` is not the lever.** Sweeping 1.0 -> 0.0 moves mean predicted strand
from 0.011 to 0.015 against a native 0.223.

**Aromatic ring excluded volume is anti-native at 8 states.** Of 81 distinct 8-state
backbones under 1.5 A, none is clash-free at its best chi (mean ring-only steric 41.2). Mild
at 4 states, so it does not explain the 4-state behaviour -- but a finer backbone library
without more sidechain freedom would make things worse.

---

## Corrections log

Each of these was stated during the investigation and retracted after a control was added.

| claim | what killed it |
|---|---|
| "`coop_helix` alone inverts the objective; lowering it to 0.667 fixes chignolin" | true on chignolin, which was in the **training** split; on held-out clusters calibration made things worse, and on calibrated weights `coop_helix` changes nothing |
| "Softening the H-bond form widens the helix/hairpin gap" | sign error from using the **snapped native** as "the encodable hairpin". It is a per-residue angle projection, not the encodable optimum -- enumeration finds a structure 0.7 A closer and 10.8 kcal/mol lower |
| "`coop_sheet` is unreachable code" | its std of 0.0000 was over **sampled** structures; on real natives it fires up to -3.0 |
| "The objective rejects beta" | no **feasibility filter**. Beta-rich random structures are almost all clashing: steric is +151.8 of a +150.5 gap, and every topology term mildly *favours* beta |

The common cause: measuring without a filter the repo itself applies (`FEASIBLE_FRACTION`),
or generalising from one protein that was also in train. **Do not conclude from a single
target, and always filter on steric before attributing anything to another term.**

---

## Recommendations

1. **Ship `DEFAULT_WEIGHTS`.** Best on 4/4 held-out targets and the only setting that passes
   the gate. Do not calibrate before predicting.
2. **Fix scoring against the enumeration metric**, not against RMSD proxies: shrink the
   -11.26 kcal/mol gap between the beta-rich and lowest-energy clash-free sets.
3. **Then fix sampling.** At 0.043 % of feasible space above strand 0.2, no search that
   draws thousands of structures will see a hairpin. This bounds the VQE and SA arms
   regardless of the objective.
4. **Decide on `test_amber_native_below_helix`.** Green, not testing what it claims, fails
   when corrected.
5. **Add a per-peptide topology gate.** `assign_secondary_structure` and the SS-agreement
   metric already exist; only the assertion is missing. A mean-based version would pass this
   model, so it must be per peptide.
6. **Never use the snapped native as "the best encodable structure".** Enumerate or search.
