# Runtime estimate and optimization plan

**Date:** 2026-07-28
**Machine:** this laptop, 10 cores, `.venv` = Python 3.14 / NumPy 2.5.1 / PennyLane 0.45.1 +
lightning 0.45.0. `openmm` and `quimb` are **not installed**, so the amber and MPS paths
cannot execute here.

Nothing in this document was produced by running an experiment. The numbers come from
three sources:

1. **Runtimes already recorded in the repo** — `results/main_comparison.csv` and
   `results/main_comparison_matched.csv` carry a `runtime` column and an
   `n_objective_evals` column, so per-objective-eval cost is directly measurable for the
   lightning path at 18/20/22/24 qubits.
2. **The profiling captures in `diag_raw/`** — real per-call timings for the amber
   energy (`prof1`, `prof5`), the MPS sampler (`prof2`, `prof2b`), the parallel pools
   (`prof3`, `prof6`), and the cache hit rate (`prof4`).
3. **A bounded micro-benchmark of unit costs only** (single circuit call, single energy
   call, 2000 annealing iterations; ~90 s total, no experiment run). Reported below as
   "isolated".

---

## 1. Measured unit costs

### Legacy energy model (`hamiltonian.FoldingHamiltonian`), N = 10

| operation | cost |
| --- | ---: |
| one uncached torsion energy (`build_coords` + 7 terms) | **0.264 ms** |
| one uncached lattice energy (CA-only, no hbond/torsion) | **0.100 ms** |
| cache hit | ~0 (dict lookup) |
| `representation_ceiling` (20 000 annealing iterations) | **2.1 s** |

### Amber energy model (`AmberHamiltonian`), from `diag_raw/prof1`, `prof5`

| sequence | atoms | per structure | minimize share |
| --- | ---: | ---: | ---: |
| chignolin `GYDPETGTWG` (N=10) | 148 | **22.7 ms** | 90.7 % |
| trpzip `SWTWEGNKWTWK` (N=12) | 218 | **52.5 ms** | 91.2 % |

### Circuit + sampling, per objective evaluation (lightning, 4 layers, 2048 shots)

| qubits | isolated circuit | isolated `rng.choice` | **end-to-end, measured from CSV** |
| ---: | ---: | ---: | ---: |
| 18 (lattice, N=10) | 0.020 s | 0.002 s | **0.037 s** |
| 20 (torsion, N=10) | 0.079 s | 0.009 s | **0.237 s** |
| 22 (lattice, N=12) | 0.430 s | 0.037 s | **0.83 s** |
| 24 (torsion, N=12) | 1.743 s | 0.140 s | **3.38 s** |

Cost grows ~4–5× per two qubits added — worse than the 4× the statevector size implies,
which is the signature of a memory-bandwidth-bound simulation. **Two extra residues cost
roughly 14×.**

### MPS path (`diag_raw/prof2`, `prof2b`, `prof4`)

| qubits | MPS rebuild | sampling @1024 shots | bond dim | per objective eval |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 4.4–5.3 s | 10.5 s | 246 | **14.9 s** (sampling only) |
| 36 | 6.0 s | 15.2 s | 221 | **21.2 s** (sampling only) |

Add the amber energy side: `prof4` measured 303 s for 28 objective evals = **10.8 s/eval**
(455 unique structures per eval, 211 of them new, 53.7 % cache hit rate).

---

## 2. Estimated runtime per entry point

`main.py` defaults: `layers=4`, `maxiter=300`, `restarts=4`, `shots=2048`,
`seeds=0,1,2`, `proteins=1UAO,5AWL`. COBYLA's `maxiter` is a function-evaluation cap, so
**one VQE run = 300 × 4 = 1200 objective evaluations**.

| command | work | **estimate** |
| --- | --- | ---: |
| `--scaling` | analytic table, no optimization | **< 1 s** |
| `--validate` | 35 tests; heaviest are a 2^12 exhaustive search, a 12q×3L / maxiter=400 / restarts=5 VQE, and two ceiling searches | **2–5 min** |
| `--predict` | one 20q torsion VQE run | **~5 min** |
| `--main-comparison` (defaults, 2 × N=10) | 24 runs: 6×lattice-18q, 6×torsion-20q, 6×SA, 6×random | **~35 min** |
| `--main-comparison --proteins 1UAO,1LE0` | 1LE0 is N=12 → 22q and 24q arms | **~4.5 h** |
| `--energy-ablation` (defaults) | 9 variants × 2 proteins × 3 seeds = 54 torsion-20q VQE runs | **~4.3 h** |
| `--hparams` (defaults) | 9 configs × 3 seeds = 27 runs, incl. `layers=8` and an SPSA run costing ~1800 evals | **~2.3 h** |
| `run_8state_seed0.py` | 30q STATES_8, MPS + amber OBC2, 200 × 3 = 600 objective evals | **~4.3 h serial** |

Per-run detail for the default `--main-comparison`:

| arm | qubits | per run | × 6 runs |
| --- | ---: | ---: | ---: |
| `A_lattice_vqe` | 18 | 1200 × 0.037 s + 2.1 s ceiling ≈ 47 s | 4.7 min |
| `B_torsion_vqe` | 20 | 1200 × 0.237 s + 2.1 s ceiling ≈ 286 s | 28.6 min |
| `C_torsion_sa` | 20 | 20 000 steps × ~0.26 ms ≈ 8 s | 0.8 min |
| `D_torsion_random` | 20 | 20 000 draws × ~0.26 ms ≈ 7 s | 0.7 min |

**Running every entry point once ≈ 7.5 hours** with the default N=10 proteins. So yes —
comfortably above an hour, and the two figures that dominate are `--energy-ablation`
(~4.3 h) and `run_8state_seed0.py` (~4.3 h).

Three things dominate the total, in order:

1. **Statevector simulation** — 33 % of an objective eval at 20 qubits, ~55 % at 24.
   This is what makes N=12 cost 14× N=10.
2. **Python-level per-sample bookkeeping** — measured end-to-end cost at 20q (0.237 s) is
   ~0.09 s more than the isolated circuit + sampling + energy costs account for. That
   remainder is `format(int(u), fmt)` for up to 2048 basis indices per eval, string-keyed
   dict lookups, `tracker.offer` in a Python loop, and `np.unique`.
3. **The energy function** — 25 % at 20q, and effectively 100 % for the SA and random
   arms and for the whole amber path.

### Two blockers to note before any of the above can be timed here

- `main.py`, `validation.py` and `experiments.py` all `import amber_hamiltonian` at module
  scope, which imports `openmm`. On this `.venv` **every `main.py` command dies at import**.
  `docs/evaluation-budget.md` claims this was made lazy; that change is not in the tree.
- `budget.py` is **dead code** — `grep` finds no importer anywhere in the repo, yet
  `results/main_comparison_matched.csv` carries `eval_budget` / `config_id` /
  `terminated_by` columns and the doc describes the wiring as landed. The runs in that
  CSV were produced by code that is no longer present. As the tree stands, searches are
  bounded only by `maxiter × restarts`, which is the regime the estimates above describe.

---

## 3. Optimizations

Ranked by (time saved) × (confidence), with the effect on results stated for each. The
first four are the ones that matter.

### O1. Parallelize the experiment loop — up to 8× on every experiment ✦ **[DONE]**

`experiment_main_comparison`, `experiment_energy_ablation` and
`experiment_vqe_hyperparameters` all run a fully sequential triple loop over
(arm/variant, entry, seed). Every iteration is independent: separate Hamiltonian,
separate RNG stream, results appended to a CSV.

- 24 independent runs in main-comparison, **54 in the ablation**, 27 in hparams, on 10 cores.
- Memory per worker is the statevector: 16 MB at 20q, 270 MB at 24q — fine at 8 workers.
- Pin BLAS/OMP threads first (`mps.parallel.pin_threads` already exists) so lightning's
  internal threading does not oversubscribe.
- `append_csv` must be serialized — collect rows in the parent, or take a lock.

Implemented as `--workers N` on `main.py` (or `$PFA_WORKERS`; `0` = cores − 2).
**Default stays 1, i.e. serial**, so nothing changes unless asked.

Three details that make it result-identical rather than merely equivalent:

- **Thread pinning is mandatory, not an optimization.** `experiments.py` now pins
  `OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1` before importing NumPy. Multi-threaded BLAS
  reductions associate nondeterministically, and `representation_ceiling` makes 20 000
  sequential `cs < cur_s` decisions — one comparison flipping on a last-ulp difference
  sends the whole annealing trajectory elsewhere. The arrays here are ~10–20 elements,
  so nothing was gaining from threading anyway.
- **`Pool.imap` preserves order**, so the parent appends rows in task order and the CSV
  is identical, not just equivalent.
- **Exceptions are returned, not raised**, matching the serial loop's per-cell
  `try/except` so one bad cell cannot take down a sweep.

**Verified:** serial vs 4 workers on `--main-comparison` (1UAO, 2 seeds, 4 arms) —
8 rows × 35 columns compared as raw CSV cells, **0 mismatches**, row order preserved.
Serial vs 8 workers on `--energy-ablation` (all 9 variants) — **0 mismatches**, variant
order preserved. `runtime` excluded from both, since it is a wall-clock measurement.

**Measured speedup: 1.66×** (8 tasks / 4 workers) and **1.08×** (9 tasks / 8 workers).
Both tests are startup-dominated — worker spawn costs **1.60 s** (importing NumPy, SciPy
and PennyLane) against deliberately shrunk 3–6 s tasks. Real cells run 47–286 s, where
1.6 s is 1–3 %, so the projections below should be close; **but that is extrapolation, not
something I measured.** One real effect that does cut against it: the O7 ceiling cache is
per-process, so workers each recompute ceilings the serial run cached.

### O2. Turn on the parallel paths that are already written but default to off ✦ **[GATED — cannot verify here]**

`mps/parallel.py` implements process-parallel restarts (REC 1), a minimization pool
(REC 2) and thread pinning (REC 4), documented and verified as *result-identical*, and
`run_8state_seed0.py` gates them behind `PFA_RESTART_PROCS` / `PFA_MINIM_WORKERS`, both
defaulting to serial. The measured speedups are already in `diag_raw`:

- `prof3`: 8-way minimization pool = **2.95×** on the amber energy side.
- `prof6`: 3 concurrent MPS processes = **2.45×** throughput.

**Saving:** `PFA_RESTART_PROCS=3 PFA_MINIM_WORKERS=8` takes `run_8state_seed0.py` from
~4.3 h to **~1.5 h** with no code change at all.

**I did not enable this, and the default is deliberately left serial.** Two reasons:

1. **It cannot be verified in this environment.** `openmm` and `quimb` are not installed,
   so neither the parallel path nor the checks that pin it can execute. "Result-identical"
   here is `mps/parallel.py`'s own claim plus verification done on some other machine.
   The failure modes that would break it — OpenMM platform selection, thread pinning,
   worker Hamiltonian reconstruction — are exactly the environment-dependent ones.
2. **`run_8state_seed0.py`'s seed-0 data is byte-pinned.** `amber_obc2.py` documents that
   its construction is verbatim so energies stay identical to the persisted set. Flipping
   a default that touches that run is not a change to make on an unverified claim.

What I added instead is the gate: **`partest/verify_parallel.py`** runs the three checks
that already exist — `golden_check.py` (energies vs the persisted seed-0 CSVs, max abs
error 0.0), `rec2_pool_check.py` (the same energies through pool workers),
`equiv_check.py` (serial vs parallel driver, bit-for-bit selection) — and prints the
enable command only if all three pass. On a machine without `openmm`/`quimb` it exits 2
and refuses. Run it there, then set the env vars.

The equivalent does not exist for the lightning path — `vqe.run_global_cvar_vqe` runs its
`restarts` loop serially with no parallel option. Adding one gives ~3.5× on every
lightning VQE arm, and is the same pattern already proven in `mps/parallel.py`.

### O3. Cut the MPS bond dimension — the single biggest lever on the 30q/36q runs ✦

`MPSSampler` defaults to `max_bond=8192, cutoff=1e-12`, so the cap never binds and the
measured bond dimension floats to 246 at 30 qubits. Sampling cost scales as O(shots · n · χ²),
and sampling is 70 % of the 14.9 s per-eval MPS cost.

- `prof5` already shows χ = 88 at `init_scale=0.25` giving 12.7 s/eval.
- Capping at χ = 128 is a **3.7× reduction in χ²** versus 246.

**Saving:** potentially 2–3× on the MPS sampling term, i.e. ~1 h off
`run_8state_seed0.py`. **Not result-identical** — this is a controlled fidelity/cost
trade. It must be validated by re-running one seed at both settings and comparing the
sampled energy distribution, not adopted blind. Note `run_8state_seed0.py`'s persisted
seed-0 data is explicitly byte-pinned (`amber_obc2.py`), so this cannot be applied there
without invalidating that set.

### O4. Memoize the Ramachandran penalty — removes ~12 % of the legacy energy ✦ **[DONE]**

`torsion_term` loops over residues calling `rama_penalty`, which runs a 6-basin Python
loop with `math.exp` and `math.degrees` per residue, per structure. But phi/psi come from
a **fixed discrete library of `n_states` values**, so `rama_penalty(sequence[i], state_k)`
takes only `n_residues × n_states` distinct values — 40 of them for chignolin at 4 states.

Implemented as an `lru_cache` on `rama_penalty` rather than a precomputed table — it is
bit-identical for *any* input (continuous native angles simply miss) instead of only for
the discrete library, and it needs no plumbing of state indices into `energy_terms`.

**Measured:** `energy_components` 0.263 → **0.232 ms/structure** (12 %). Less than the
20 % predicted, because the surrounding per-residue loop and the cache lookups remain.
Verified bit-identical against `HEAD` over a 74 420-point (aa, phi, psi) grid and 500
random structures in each of the torsion and lattice representations: **0 mismatches
under exact equality.**

### O5. Batch the energy evaluation over the unique set

Both `vqe._run_single` and `mps/driver._run_single` already compute `np.unique(idx)` and
then loop over the (up to 2048) unique bitstrings one at a time. At N=10 the arrays are
45–100 elements, so **NumPy call overhead is essentially the entire cost** — ten-odd
`np.linalg.norm` / `np.where` calls on 45-element arrays per structure.

Stack the whole unique batch into `(B, n, 3)` and compute each term once for all B
structures. `mps/driver.py` already has the `energy_batch` hook for exactly this shape of
change; `vqe.py` does not.

**Saving:** 5–20× on the energy term at these sizes; ~20 % off a torsion-VQE eval, and it
is the dominant win for the SA/random arms and for `--energy-ablation`. **Results
identical** if the reduction order is preserved (the `energy_batch` docstring already
records the offer-order constraint).

`geo.build_backbone` should be batched with it — it is a pure-Python chain of ~40
`_place_atom` calls with `math.cos`/`math.sin` inside. Since the per-residue transforms
come from a fixed `n_states`-element library, the rigid transforms can be precomputed once
and composed, eliminating all trig from the hot path.

### O6. Key the energy cache by integer basis index, not by bitstring

Every objective eval does `format(int(u), fmt)` for up to 2048 unique indices, then uses
those strings as dict keys (hashing a 20–36 char string), then `state_indices` re-parses
them with `int(bitstring[o:o+b], 2)` per residue. The sampler produces integers and the
representation wants integers; the string is a detour.

Keying on the int and decoding states by bit-shifting removes ~2048 string constructions,
2048 string hashes and 10× 2048 substring parses per eval.

**Saving:** a meaningful slice of the unexplained ~0.09 s/eval remainder identified in §2.
**Results identical**, but it touches the public `energy(bitstring)` signature used by
`validation.py` and the classical baselines, so keep the string form as a wrapper.

### O7. Cache `representation_ceiling` — pure waste, trivially removed **[DONE]**

`representation_ceiling` runs 20 000 annealing iterations (2.1 s) and depends only on
`(representation kind, n_states, native structure, seed)`. It does **not** depend on the
search arm or on the search result. Yet `run_one` calls it fresh for every row:

- `--main-comparison`: 4 arms × 2 proteins × 3 seeds = 24 calls where 6 distinct values
  exist. **18 redundant computations ≈ 38 s.**
- `--energy-ablation`: called with the default `seed=0` for all 54 rows, where **1**
  distinct value exists. **53 redundant computations ≈ 111 s.**

Memoized on `(rep kind, n_bits, n_states, seed, iterations, native CA/phi/psi bytes)`.
**Measured:** 0.403 s cold → **15 µs** warm at 4000 iterations; the production setting is
20 000 iterations (~2.1 s), so this removes ~38 s from a default `--main-comparison` and
~111 s from `--energy-ablation`. Verified that a forced recomputation returns the same
bitstring and RMSD, and that distinct seeds and distinct representations do not collide.

One deliberate difference: a cache hit returns the *original* call's `runtime` field. No
caller reads it — `run_one` takes only `ceiling_ca_rmsd` — but it means the field now
measures the first computation rather than the current call.

### O8. Share the component cache across ablation variants

The nine ablation variants (`full`, seven `no_<term>`, `raw_mj`) differ only in the weight
vector. For a given bitstring, `energy_components` returns the *same dict* for the first
eight; only the weighted sum differs. Currently each variant builds a fresh
`FoldingHamiltonian` and recomputes coordinates and all seven terms from scratch.

Cache the component dict keyed by `(sequence, use_corrected_mj, bitstring)` and share it
across variants. `raw_mj` needs its own key since `use_corrected_mj` differs.

**Saving:** up to the full energy share of `--energy-ablation` (~25 % of that experiment)
wherever the eight searches revisit the same structures. **Results identical.**

⚠️ Do **not** extend this idea to sharing a cache across the *arms* of
`--main-comparison`. The comparison is explicitly quantum-vs-classical at equal cost, with
unique energy evaluations as the currency (`docs/evaluation-budget.md`). A cross-arm cache
would make arms C and D nearly free in wall-clock **and** in charged evaluations,
destroying the cost-match. If it is done, the charged count must be kept separate from the
wall-clock cache.

### O9. Sample from a shots-based device instead of materializing 2^n probabilities

`build_global_circuit` returns `qml.probs(wires=range(n))`, so every objective eval
allocates and normalizes the full 2^n vector, and `rng.choice(probs.size, size=shots, p=probs)`
then builds a 2^n cumulative sum. At 24 qubits that is a 16.7M-element cumsum per eval —
**0.140 s measured, 4 % of the eval at 24q and 11 % at 20q**, on top of the allocation.

A shots-based lightning device samples without ever forming the full vector.

**Saving:** the sampling term outright, plus allocation pressure that is likely feeding the
superlinear scaling in §1. **Not result-identical** — it changes the RNG stream, so every
persisted seed-0 result would need regenerating. The `distribution_top1_prob` /
`distribution_entropy_bits` outputs would also have to move to the empirical estimator the
MPS branch of `mps/driver.py` already uses.

### O10. Re-wire `budget.py`, with a clear-eyed note on what it does and does not bound

Re-connecting `BudgetedEnergyModel` caps every arm at a fixed number of unique energy
evaluations (20 000 by default) instead of the current `maxiter × restarts`.

**But it does not fix the 24-qubit blowup.** The budget bounds *energy* evaluations, and
at ≥22 qubits the statevector simulation dominates — the 1LE0 torsion run spent 507 s on
150 objective evals, of which the energy side was ~40 s. A budget stops the energy side
growing; only O1/O9 touch the circuit side. Worth doing for experimental soundness, and it
does bound the N=10 arms, but it is not the runtime lever the doc's framing suggests.

### O11. Small, safe cleanups

- `geo.contact_map` is an O(n²) Python double loop calling `np.linalg.norm` on 3-vectors;
  `geo.dssp_hbonds` is the same with four norms per pair. Both are one `cdist` call.
  ~3 calls per row, so a few ms — cosmetic, but free.
- `_run_single` constructs `np.random.default_rng(np.random.SeedSequence([seed, k]))` on
  every objective eval (~50–100 µs of SeedSequence hashing). This is load-bearing for
  reproducibility, so it can only be replaced by an equivalent pre-spawned stream — worth
  it only if O5/O6 land first and it becomes visible.
- SPSA calls the objective **three** times per iteration (`fp`, `fm`, `fx`), where the
  third exists only for best-tracking. Tracking `(fp+fm)/2` instead saves 33 % of the SPSA
  configuration in `--hparams`. This *is* a behavior change to the optimizer.
- `AmberHamiltonian._assemble` rebuilds every hydrogen's local frame in a Python loop with
  `np.cross` / `np.linalg.norm` on 3-vectors — 1.45 ms of the 22.7 ms per structure (6.6 %),
  rising to 2.59 ms at N=12. Fully vectorizable into one batched `(H, 3)` computation.
  Likewise `_evaluate` calls `setParticleParameters` in a Python loop over all 3N restraint
  particles every evaluation.

---

## 4. Status and suggested order

**Landed** (all verified bit-identical, default behaviour unchanged):

| | what | verification |
| --- | --- | --- |
| **O4** | `lru_cache` on `rama_penalty` | 74 420-point grid + 1000 structures, 0 mismatches |
| **O7** | memoized `representation_ceiling` | forced recompute matches; seeds/reps don't collide |
| **O1** | `--workers N` on the experiment loops | 8 rows × 35 cols and 9 ablation variants, 0 mismatches |

**Gated:** **O2** — `partest/verify_parallel.py` added; run it on a machine with
`openmm`+`quimb`, then set `PFA_RESTART_PROCS` / `PFA_MINIM_WORKERS`.

**Remaining, in order:**

1. **O5** — batch the energy over the unique set. Biggest remaining identical-result win,
   and it compounds with O1. Needs a bit-identity check, since batching changes NumPy
   reduction shapes and `np.sum`'s pairwise blocking depends on array shape and strides.
2. **O6, O8** — good, more invasive.
3. **O3, O9** — real speedups but they change numbers. Validate against a pinned seed
   before adopting.
4. **O10** — do it for soundness, not for speed.

Also outstanding and blocking everything on a machine without OpenMM: the module-scope
`import amber_hamiltonian` in `main.py`, `validation.py` and `experiments.py` (§2).

Fixing the module-scope `openmm` imports (§2) is a prerequisite for any of this being
runnable on a machine without OpenMM.

Projected effect of steps 1–4 on the worst offenders:

| | before | after |
| --- | ---: | ---: |
| `--energy-ablation` | 4.3 h | **~25 min** |
| `--hparams` | 2.3 h | **~15 min** |
| `--main-comparison` (defaults) | 35 min | **~5 min** |
| `run_8state_seed0.py` | 4.3 h | **~1.5 h** |
