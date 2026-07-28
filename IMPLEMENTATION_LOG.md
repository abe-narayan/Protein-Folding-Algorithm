# Implementation Log — Safe Speedups (REC 1/2/4 from DIAGNOSIS.md)

Durable on the C: drive. One entry per change, with the automatic golden-check result
and any measured speedup. A session cutoff loses nothing.

**NON-NEGOTIABLE INVARIANT:** REC 1/2/4 must be byte-identical — same energies, same
selected bitstring, same RMSD, determinism exactly 0.0, `--validate` 35/35. After every
change: golden check (≥30 bitstring→energy pairs from each of the chignolin + trpzip
seed-0 CSVs; **max abs error must be 0.0**) + determinism + validate. On any failure:
revert that change from backup, log why, move on.

**FORBIDDEN as speed levers** (change results, DIAGNOSIS §7): minimizer steps/tolerance,
collapse floor, energy math, shots, MPS cutoff. Not touched.

---
## Environment / references
- Interpreter (owns openmm 8.5.2 + quimb 1.14.0): WindowsApps Python 3.11
  `C:\Users\abena\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python3.11.exe`
- Repo: `C:\Users\abena\Protein-Folding-Algorithm` (git HEAD `9c1c53d`).
- **Backup (verified identical, 78 files):** `C:\Users\abena\PFA_backups\backup_20260727_135627`
- Golden CSVs (durable copies): `C:\Users\abena\PFA_backups\golden\{seed0_full_eval_set.csv, trpzip_seed0_full_eval_set.csv}`
- Golden harness: `partest/golden_check.py`  (`python partest/golden_check.py`)
- Persisted BASELINE wall-times (ground truth, not re-run):
  - chignolin (30q): **2.362 h** (`seed0_logged_meta.json`, 95,643 unique minimizations)
  - trpzip (36q): **~4.23 h** (133,801 unique minimizations)

---
## Baseline capture (unmodified code)
- **Golden check:** chignolin max_abs_err **0.0** / inf_mismatch 0; trpzip max_abs_err
  **0.0** / inf_mismatch 0. Determinism 0.0 both. (~8.5 s)  → PASS
- **--validate:** 35/35 passed (ALL PASSED: True), 2m22s.  → PASS

---
## Change log

<!-- entries appended below, newest last -->

### REC 4 — pin OPENBLAS/OMP threads (free enabler) — ✅ DONE
**What changed:**
- New `mps/parallel.py`: `pin_threads()` sets `OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/
  MKL_NUM_THREADS/NUMEXPR_NUM_THREADS = 1` (auto-run at import); + picklable `OBC2Builder`.
- `run_8state_seed0.py`: inline env pin at the very top **before `import numpy`** (must
  precede OpenBLAS init to take effect in the main process).
- Mechanism for workers: Windows `spawn` children inherit the parent env at creation, so
  pinning the parent pins workers; each worker initializer also calls `pin_threads()`.
- De-risk confirmed: spawned workers report `OPENBLAS_NUM_THREADS=1`; threadpoolctl is
  not installed, so env-var inheritance is the (working) mechanism.
**Golden check:** chignolin max_abs_err **0.0** / inf 0; trpzip **0.0** / inf 0;
determinism 0.0 both. **--validate:** 35/35. → PASS. Zero result change.

### REC 1 — 3 COBYLA restarts as independent processes — ✅ DONE
**What changed:**
- `mps/parallel.py`: `run_parallel_restarts` launches `restarts` non-daemonic `spawn`
  processes, one per restart, each rebuilding its own OBC2 Hamiltonian via `OBC2Builder`
  and running the unchanged `_run_single` COBYLA trajectory. A `Manager().dict()` shared
  cache preserves cross-restart reuse (local H cache first → shared cache → compute).
- `mps/driver.py`: `_run_single` gained an optional `energy_batch` hook (serial path
  unchanged — same bitstrings, same np.unique order, same `tracker.offer` sequence).
  `run_global_cvar_vqe` gained opt-in `hamiltonian_builder / restart_procs /
  minim_workers / use_shared_cache`; `restart_procs==1` keeps the original serial path.
  Best-run reduction and best-seen merge run in restart order with strict `<`, exactly
  reproducing the serial first-wins tie-break.
- `run_8state_seed0.py`: opt-in via `PFA_RESTART_PROCS` / `PFA_MINIM_WORKERS` env
  (default 1/0 → serial → byte-identical to the persisted seed-0 run); `freeze_support`.
**Verification (`partest/equiv_check.py`, small real OBC2, 12q):** serial vs parallel
selection **byte-identical** — vqe/modal/best-seen bitstrings + energies, final_objective,
and full history all match with **worst |d| = 0.0**, for restart_procs=3 with
minim_workers=0, with shared cache OFF, and (below) with the nested pool. This also
empirically shows BLAS thread count does **not** perturb the MPS trajectory (serial main =
default threads vs workers = 1 thread → identical samples).
**Golden check:** 0.0 both. **--validate:** 35/35. → PASS.

### REC 2 — per-eval OpenMM minimizations in a process pool — ✅ DONE
**What changed:**
- `mps/parallel.py`: `MinimizationPool` — a persistent `spawn` pool whose workers each
  own their own OBC2 Hamiltonian (Threads=1, BLAS pinned). `compute(bitstrings)` maps the
  unique structures round-robin across workers → `{bitstring: energy}`. Wired into the
  restart worker's `energy_batch` (used for cache misses) so results populate the shared
  cache; enabled by `minim_workers>1` and composes with REC 1 (nested, non-daemon parent).
**Verification (`partest/rec2_pool_check.py`):** the 48 golden bitstrings per protein,
computed through the pool, reproduce the CSVs with **0 mismatches, max abs error 0.0**
(chignolin + trpzip). The nested REC 1+2 config in `equiv_check.py` is byte-identical too
(worst |d| = 0.0). **Golden check:** 0.0 both. **--validate:** 35/35. → PASS.
Note: `n_energy_evaluations` / `n_unique_structures_cached` counters may differ from serial
(a modest increase in redundant, identical-valued minimizations from lost cross-restart
reuse) — expected per DIAGNOSIS §5 and NOT part of the byte-identical invariant.

### Combined end-to-end measurement — IN PROGRESS
**Config:** `partest/measure.py <protein> 3 2` = config A (layers 4 / maxiter 200 /
restarts 3 / shots 1024, seed 0) with `restart_procs=3, minim_workers=2` (REC 1+2 nested,
BLAS pinned). Baselines from the persisted golden runs: chignolin 8503.6 s (2.362 h),
trpzip 15231.19 s (4.231 h). Each run cross-checks selected + best-seen bitstring/energy
against the golden targets (must be byte-identical); RMSD cross-check verified to reproduce
golden (chignolin 3.5288, trpzip 4.789368) before launch.
**Housekeeping (clean-machine):** a leftover `prof4_cache_traj.py` from a prior diagnosis
session was found consuming ~1 full core (3876 CPU-s over ~1 h). DIAGNOSIS.md states prof4
is optional/non-essential ("all conclusions stand"). It was terminated so the wall-time
measurement is uncontended and honest; the first (contended) chignolin launch was
discarded and the run relaunched on an idle machine.

**CHIGNOLIN (30q) combined result — restart_procs=3, minim_workers=2, clean machine:**
- **WALL 2280.3 s (0.633 h)** vs baseline 8503.6 s (2.362 h) → **SPEEDUP 3.73×**
  (DIAGNOSIS estimate ~3× — **exceeded**).
- **BYTE-IDENTICAL SELECTION: True.** selected `001000010000000000000000001000`
  E=-368.0638953527846 (golden |dE|=0.0); best_seen `100000011011111111100000000101`
  E=-396.4426263461881 (golden |dE|=0.0); CA-RMSD 3.5288 (golden 3.53); audit CLEAN.
- Counters: `n_unique_structures_cached` 95643 == golden 95643;
  `n_energy_evaluations` 95853 (+210 redundant, identical-valued minimizations from
  cross-restart cache races — expected, not part of the invariant).
- Artifact: `partest/measure_chignolin_r3_m2.json`.

**TRPZIP (36q) — pre-existing repo bug hit + fixed (byte-identical), re-running:**
- First trpzip run completed all 3 restarts (the expensive ~99%) then **crashed at the
  final-stats step** in `mps/driver.py`: `np.bincount(final_idx)` allocates max(index)+1
  entries — at 36 qubits ~2^36 (512 GiB) → `_ArrayMemoryError`. This is **original repo
  code, not a REC 1/2/4 change**; it is exactly the "bincount→unique infra fix" the golden
  36q run used via a scratchpad driver (the repo `mps/driver.py` was never fixed for 36q;
  at 30q the max sampled index fit in RAM so chignolin succeeded).
- **Fix (byte-identical):** count over the *observed* indices —
  `vals, vcounts = np.unique(final_idx, return_counts=True); freq = vcounts/final_shots;
  modal = vals[argmax(vcounts)]`. Produces the identical `freq` vector that
  `bincount[bincount>0]/N` gives and the same lowest-value argmax tie-break; only feeds
  reported distribution stats + modal bitstring (selection untouched). Verified byte-
  identical to the old path on random + forced-tie cases (modal/top1/top16/entropy all ==).
- **Golden check after fix:** 0.0 both. **--validate:** 35/35. Re-ran trpzip clean.

**TRPZIP (36q) combined result — restart_procs=3, minim_workers=2, clean machine:**
- **WALL 4256.0 s (1.182 h)** vs baseline 15231.2 s (4.231 h) → **SPEEDUP 3.58×**
  (DIAGNOSIS estimate ~4× — **fell ~0.4× short**, honest sublinear scaling on this
  power-limited 4P+4E laptop; trpzip's heavier OpenMM load, 46% of wall at 2.31× the
  per-min cost, saturates the E-cores harder than chignolin's MPS-bound profile).
- **BYTE-IDENTICAL SELECTION: True.** selected `001010000000000000000000001000000000`
  E=-395.792567917652 (golden |dE|=0.0); best_seen `011011100000000000000000001000000000`
  E=-398.80968070881033 (golden |dE|=0.0); CA-RMSD 4.789368 (golden 4.789368); audit CLEAN.
- Counters: `n_unique_structures_cached` 133801 == golden 133801;
  `n_energy_evaluations` 134066 (+265 redundant identical-valued minimizations).
- Artifact: `partest/measure_trpzip_r3_m2.json`.

---
## FINAL — combined end-to-end speedup vs DIAGNOSIS estimate

| protein | baseline | combined (r=3, m=2) | **measured speedup** | DIAGNOSIS est. | byte-identical |
|---|---|---|---|---|---|
| chignolin (30q) | 2.362 h | **0.633 h** | **3.73×** | ~3× (exceeded) | ✅ sel+best-seen \|dE\|=0.0 |
| trpzip (36q)    | 4.231 h | **1.182 h** | **3.58×** | ~4× (short ~0.4×) | ✅ sel+best-seen \|dE\|=0.0 |

Honest read: **~3.6–3.7× combined on this laptop.** Chignolin beat the estimate; trpzip
landed short, consistent with DIAGNOSIS §5/§8's explicit sublinear-scaling caveat for the
power-limited 4P+4E mobile chip (a homogeneous many-core workstation would scale closer to
linear). Every selected/best-seen bitstring, energy, and CA-RMSD is byte-identical to the
persisted golden serial runs; determinism 0.0; `--validate` 35/35 throughout. All measured
on a clean machine (leftover prof4 profiler terminated first). REC 3 left unimplemented by
design — see `DIAGNOSIS_REC3.md`.
