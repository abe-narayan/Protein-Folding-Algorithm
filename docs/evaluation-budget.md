# Shared evaluation budget

**Date:** 2026-07-27
**Status:** landed end-to-end and verified on the legacy model; amber and MPS paths
still unexecuted (no `openmm` / `quimb` in this environment)
**Touches:** `budget.py` (new), `hamiltonian.py`, `amber_hamiltonian.py`, `vqe.py`,
`mps/driver.py`, `classical_baselines.py`, `experiments.py`, `main.py`, `validation.py`

---

## Why

The starting complaint was narrow: the VQE's COBYLA allowance was too small to
optimize. `run_8state_seed0.py` runs 120 parameters (30 qubits x 4 layers) with
`maxiter=200`, and COBYLA spends its first `n_params + 1 = 121` evaluations building an
initial simplex — during which it has done no optimizing at all. That leaves ~79
trust-region steps in 120 dimensions.

Checking `results/main_comparison.csv` to size a fix turned up the larger problem: the
allowance was not just small, it was **set independently per arm**, so the arms were
never cost-comparable in the first place.

`n_energy_evaluations` is the real cost currency — one OpenMM backbone-restrained
minimization each under the amber model, dominating wall-clock for every arm. It was
uncontrolled:

| protein | arm | energy evals | best energy |
| --- | --- | ---: | ---: |
| 1UAO | B_torsion_vqe | 11,080 | −7.4308 |
| 1UAO | C_torsion_sa | 11,245 | −7.4308 |
| 1UAO | D_torsion_random | 19,822 | −5.6583 |
| **1LE0** | **B_torsion_vqe** | **39,599** | **−2.0188** |
| **1LE0** | **C_torsion_sa** | **14,004** | **−2.8887** |
| 1LE0 | D_torsion_random | 19,990 | −1.8399 |

On 1LE0 the VQE spent 2.8x the oracle budget of simulated annealing and returned a
worse answer, and the CSV presented the two rows as comparable. No conclusion in either
direction was supportable from that file.

Two further defects in the same data, **not yet fixed** (see Outstanding):

1. **Mixed configurations.** `append_csv` appends blindly, so the file interleaves runs
   at `maxiter` 150/300/400 and `restarts` 1/2 as one experiment. `A_lattice_vqe` on
   1LE0 appears at 18,365 / 11,455 / 11,455 evaluations under three different configs.
2. **Fabricated error bars.** `_print_summary` prints "mean +/- std CA-RMSD across
   seeds", but every row in the file is seed 0. The spread is across duplicated re-runs
   of one seed, not across seeds.

## What was done

### Cost is now a shared, enforced budget

`budget.py` adds `BudgetedEnergyModel`, mixed into both `FoldingHamiltonian` and
`AmberHamiltonian` so the two energy models account identically. Charges are levied on
**cache misses only** — revisiting a structure costs nothing, which is correct, and is
itself a confound the old iteration-count comparison missed: annealing revisits heavily,
so 20,000 SA steps consumed only ~11–14k evaluations while 20,000 random draws consumed
~19.8k. Same nominal "20,000", ~1.7x the real cost.

Exhausting the budget raises `BudgetExhausted`, which search drivers catch as a normal
termination condition and return best-so-far. `exhaustive_search` is explicitly exempt —
it is the ground-truth reference, not a competing arm.

### This dissolves the original question rather than answering it

The fix deliberately avoids picking a `maxiter` multiple by rule of thumb. `maxiter=None`
is now the default and resolves to `50 x n_params` — high enough that the **shared
budget** is what terminates the search. So "is 15x or 20x `n_params` right?" stops being
a free parameter. Every arm stops when the same number of unique energies has been
computed, and if COBYLA burns its allowance on a 120-dimensional initial simplex while
annealing converges inside the same budget, that is a real result about optimizer choice
rather than an artifact of configuration.

It also makes SPSA-vs-COBYLA an empirical question instead of a judgment call.

### The guard has teeth

`check_optimizer_budget` replaces the old `maxiter >= n_params + 2` check, which passed
any config that could merely *construct* a simplex. It now hard-errors below
`3 x n_params`, and warns when `maxiter` binds because no shared budget was set.

### The harness now uses it, and refuses to present incomparable rows

`experiments.py` passes one `eval_budget` (default 20,000) to every arm. Each row is
stamped with a `config_id` — a hash of the experiment-level settings — and
`_print_summary` reports one block per configuration, never averaging across them.

Two further checks were added there, both aimed at defects already present in
`results/main_comparison.csv`:

- **Duplicate-seed detection.** The seed column shows the number of *distinct* seeds and
  marks it `!` when a std was computed over repeated runs of one seed.
- **Cost-match assertion.** Within a configuration, if arms differ by more than 5% in
  evaluations spent, the summary prints `NOT cost-matched, comparison unsound`.

### Amber is now an optional dependency

`experiments.py`, `main.py` and `validation.py` imported `amber_hamiltonian` — and
therefore `openmm` — at module scope, so on a machine without OpenMM *every* legacy-model
run and the entire validation suite died at import. The import is now lazy, and
`run_all()` skips the five amber tests with an explicit notice rather than failing.
`python main.py --validate` runs here for the first time.

### Two design details worth recording

- **Read-out reservation.** `reserve(k)` withholds part of the budget behind a soft
  limit so a run can always report its answer after the optimizer spends everything;
  `release()` returns it for the final read-out. Charges still stop at the hard limit,
  so the reserve bounds read-out rather than exempting it.
- **Per-restart slicing.** First attempt gave the VQE one shared pool, and restart 1
  consumed all of it — silently reducing a 4-restart algorithm to 1 restart. Fixed by
  slicing the pool cumulatively across restarts: each gets an equal share, an
  early-terminating restart passes its remainder forward, and no single restart can
  swallow the budget.

## Things that broke along the way

Four bugs were introduced and caught while building this. Recorded because three of
them were silent — they produced plausible output rather than errors.

1. **Biased read-out under truncation.** `np.unique` returns basis indices in *sorted*
   order, so when the budget truncated the final read-out, the scan kept an arbitrary
   low-index subset of bitstrings rather than the ones the circuit favoured. Now ordered
   by descending sample count, so truncation keeps the most probable states.
2. **`config_id` split cost-matched arms apart.** Including the *resolved* `maxiter` in
   the hash produced three configuration blocks for one experiment, because `maxiter` is
   now derived per-arm from the qubit count (torsion-VQE 4000, lattice-VQE 3600,
   SA/random `None`). The hash now keys on `maxiter_requested`, and excludes
   `representation`/`n_states` — those are what the arms compare, not configuration
   differences. A validation test pins both directions.
3. **Post-hoc analysis billed to the search budget.** `--predict` crashed with
   `BudgetExhausted` while printing its energy breakdown, and `run_one`'s metrics were
   one cache-miss away from the same fate. Added `end_search()`, which stops charging
   once the search is over while preserving the spent count for reporting.
4. **Spurious duplicate-seed warning.** A shared `flagged` variable made the cost-match
   failure also emit the seed-duplication footer. Cosmetic, but it would have made a
   real warning look untrustworthy.
5. **The read-out reserve quietly penalised the VQE.** Reserving 10% of the budget up
   front capped the VQE's *search* at 18,000 of 20,000 while SA and random search used
   the full 20,000 — reintroducing, at ~7%, precisely the unfairness this work exists to
   remove. Caught only because the summary's own spend check flagged a 1.45x spread on
   the first full run. The read-out turned out to need ~500, because it scans
   most-probable-first and those states are nearly all cache hits after a converged
   search. Reserve cut to 1%.
6. **The cost-match check tested the wrong condition.** It flagged unequal *actual
   spend* as "comparison unsound". But an arm that converges and stops early has gained
   no advantage; the real soundness condition is an equal **allowance**. Now it errors
   on unequal or absent `eval_budget` and reports spend spread as information —
   "under-spending arms terminated before exhausting it", which is itself a finding.

## What was measured

All on this machine, legacy energy model, chignolin `GYDPETGTWG`, 4-state torsion
(20 qubits), seed 0, `layers=4`, `restarts=4`.

**Budget is enforced exactly.** `BudgetExhausted` raised at evaluation 6000 of a 6000
budget.

**All arms now spend the same, and all restarts complete.** Authoritative run:
chignolin, legacy model, 4-state torsion, seeds 0/1/2, budget 20,000
(`results/main_comparison_matched.csv`).

| arm | evals | best energy (the objective) | CA-RMSD (Å) |
| --- | ---: | ---: | ---: |
| B_torsion_vqe | 19,982 | −1.306 ± 0.583 | **4.33 ± 0.66** |
| C_torsion_sa | 20,000 | **−2.282 ± 0.118** | 5.17 ± 0.45 |
| D_torsion_random | 19,820 | −2.153 ± 0.061 | 5.49 ± 0.15 |

Per-seed best energy, which is what every arm is actually minimizing:

| arm | seed 0 | seed 1 | seed 2 |
| --- | ---: | ---: | ---: |
| vqe | −2.130 | −0.872 | −0.918 |
| sa | −2.203 | −2.194 | −2.450 |
| random | −2.069 | −2.213 | −2.176 |

**The guard classifies the historical configs correctly:**

| config | verdict |
| --- | --- |
| `run_8state_seed0.py` — 30q x 4L, maxiter=200 (120 params) | **rejected** (below 3x = 360) |
| 1LE0 in `main_comparison.csv` — 24q x 4L, maxiter=150 (96 params) | **rejected** (below 3x = 288) |
| `main.py` default — 20q x 4L, maxiter=300 (80 params) | accepted, warns (3.8x) |

**No regressions:** `python main.py --validate` reports **35/35 passed** (30 pre-existing
plus 5 new), including `VQE optimality gap on enumerable system` (exact −2.9608, vqe
−2.9608, gap 0.0000). `--predict`, `--scaling` and `--main-comparison` all run clean.

**The MPS driver still matches `vqe.py` exactly** on the shared lightning path
(`sampler=None`), which is the parity property its docstring claims: identical energy
(−1.3186) and identical evaluation count (6000) for the same seed and budget.

**New tests** (`validation.py`), each pinning a property that was violated before:

| test | asserts |
| --- | --- |
| `budget_is_enforced` | the budget is a hard ceiling, not a hint |
| `arms_are_cost_matched` | vqe/sa/random spend within 2% of each other |
| `restarts_survive_shared_budget` | 4/4 restarts complete; none swallows the pool |
| `optimizer_budget_guard_rejects_simplex_only` | the two historical configs are rejected |
| `summary_refuses_to_mix_configs` | configs split; compared arms do *not* split |

**The summary guards fire on the real pathologies.** Fed synthetic rows reproducing what
is in `results/main_comparison.csv`:

| injected defect | output |
| --- | --- |
| three rows all seed 0 | seed column reads `1!` + footer warning |
| maxiter 150 and 300 interleaved | splits into two `CONFIG` blocks |
| 39,599 vs 14,004 evaluations (the real 1LE0 case) | `arms spent 14004-39599 (2.83x spread) — NOT cost-matched, comparison unsound` |

## Did it work?

**Yes for the mechanism.** Arms are now cost-matched to the evaluation, the budget is
enforced, restarts survive, the guard rejects the configs that motivated the work, and
nothing regressed.

**The result is unfavourable to the quantum arm, and the RMSD column is a trap.**

Read the RMSD alone and the VQE looks best (4.33 Å vs 5.17 and 5.49). Read the objective
every arm is actually minimizing and it is clearly worst: mean energy −1.306 against SA's
−2.282 and random search's −2.153, with five times the seed-to-seed variance (±0.583 vs
±0.118). **Plain random search beat the CVaR-VQE on every individual seed.** Only one of
three VQE seeds was competitive; the other two landed near −0.9.

So the better CA-RMSD is not a search win — it is the VQE *failing* to minimize a
known-broken energy function and landing closer to the native structure by accident. On a
landscape whose minimum is in the wrong place, worse optimization can produce better
structures. This is the same point `PLAN.md` makes: "all three find the true minimum of a
wrong Hamiltonian." Quoting the RMSD row as a quantum result would be exactly backwards.

Three caveats before any of this is quoted:

- These runs use the **legacy** energy model, which `PLAN.md` records as broken (native
  scores worse than predictions). What is measured is search efficiency on a known-wrong
  landscape. The amber comparison is the one that counts and has not been run.
- The VQE has **not been retuned** for a fair budget. `alpha`, `layers`, and
  COBYLA-vs-SPSA were all chosen under the old regime. These numbers describe this
  configuration, not the method.
- An earlier version of this note reported the VQE losing more heavily still, at budgets
  3k/6k/12k. Those numbers were distorted by bug 5 below (a 10% read-out reserve that
  shrank only the VQE's search allowance) and have been withdrawn in favour of the table
  above.

## Not verified here

Neither `openmm` nor `quimb` is installed in `.venv` on this machine, so:

- The **Amber path is untested**. Changes to `amber_hamiltonian.py` are mechanical
  (mixin substitution plus one `_charge()` call) and mirror the legacy model exactly,
  but they have not executed. `main.py --validate` cannot run here at all — it imports
  `amber_hamiltonian` at module scope. This failure predates these changes.
- The **MPS path is untested** for the same reason. `mps/driver.py` is not yet wired to
  the budget (see below).
- Oracle cost could not be benchmarked, so the right *absolute* budget number is still
  unknown. 20,000 (matching the existing baselines) remains the suggested default.

## Outstanding

- **Re-run everything that matters on the amber model.** The legacy comparisons here
  are search-efficiency measurements on a landscape `PLAN.md` already calls wrong.
- **Retune the VQE under a fair budget.** `alpha`, `layers`, and COBYLA-vs-SPSA were all
  chosen under the old regime; the current numbers describe this configuration, not the
  method.
- **Verify amber and MPS actually execute.** Both are wired but unrun here.
- **Explain the VQE's under-spend.** It stops at ~19.8k of 20k while SA uses all of it.
  Expected (COBYLA converges or the restart slice ends), but worth confirming it is
  convergence rather than a budget-accounting artifact — the last such "expected"
  discrepancy turned out to be bug 5 above.
- **`results/main_comparison.csv` is still the old, unmatched data.** It should be
  regenerated or deleted; the new matched run is in
  `results/main_comparison_matched.csv`.

## Open decisions

1. **Budget size.** Defaulting to 20,000 keeps the existing classical numbers valid.
   Needs an oracle benchmark on a machine with `openmm` to size properly.
2. **The persisted seed-0 set.** `run_8state_seed0.py` is pinned to `maxiter=200`, which
   the new guard rejects, and `amber_obc2.py` documents that its construction is
   verbatim so energies stay byte-identical to that persisted data. That run cannot be
   re-budgeted without invalidating the set and re-running. Left untouched pending a
   decision — **it will now fail the guard if run as-is.**
3. **COBYLA vs SPSA at 120 parameters.** Now answerable by experiment rather than
   argument.
