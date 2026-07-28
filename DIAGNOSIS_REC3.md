# REC 3 — Faster *batched* exact-MPS sampler (proposal, NOT implemented)

**Status: PROPOSAL for your decision.** This is the one recommendation deliberately left
unimplemented under autonomy, because — unlike REC 1/2/4 — it is **not byte-identical**.
It changes which specific samples are drawn, so it cannot reproduce the persisted seed-0
CSVs that are this project's reproducibility baseline, and the automatic golden check
therefore **cannot gate it** (there is nothing byte-identical to check against). That is
exactly why it is excluded from autonomous editing and handed back to you.

Everything below is derived from DIAGNOSIS.md §3b–§3c and §7.3 and the measured
micro-benchmarks in `diag_raw/` — no code was changed for this doc.

---
## 1. What is slow, and why REC 1/2 don't fully fix it

The dominant sink for the small peptides is the **classical MPS sampler**, not OpenMM:
**~67 % of wall for chignolin (30q), ~54 % for trpzip (36q)** (DIAGNOSIS §TL;DR, §3b).
Per objective eval the driver calls `sampler.draw_indices(params, 1024)`, which:

1. rebuilds a `CircuitMPS` (4 layers of RY + CNOT-chain + ring, SVD-compressed), then
2. draws **1024 shots one at a time** via `psi.sample(1024)` — sequential per-shot MPS
   sweeps, **10–15 ms/shot**, i.e. ~10.5 s (30q) / ~15.2 s (36q) per eval, **600× per run**
   (DIAGNOSIS §3c table).

REC 1 (3 restart processes) parallelises this **3×** across restarts, but each restart
still runs its 200 objective evals' MPS **sequentially** — the hard floor is
**~1900 s ≈ 0.53 h per restart** for chignolin (DIAGNOSIS §5, "Floor"). Below the
~2.5–3× restart-parallel ceiling, the *only* remaining lever for MPS-bound small peptides
is to **make each 1024-shot draw itself cheaper**. That is REC 3.

The cost is intrinsic to *how* quimb draws shots, not to a wrapper: DIAGNOSIS §3c measured
`psi.sample` ≈ `CircuitMPS.sample` (11.8 vs 10.5 s) and sample time **linear in shots**
(256→3.0 s, 512→5.7 s, 1024→10.5 s, 2048→24.2 s at 30q). Each shot re-walks the chain.

## 2. What REC 3 changes

Replace the **per-shot sequential** sampler with a **batched** sampler that shares one
right-to-left canonical-form sweep across all shots:

- **Canonicalise once.** Put the MPS into right-canonical form and precompute, per site,
  the right-environment needed for the conditional bit probabilities — done **once per
  eval**, not once per shot.
- **Sample all shots together, site by site.** Walk qubits 0→n-1. At each site hold a
  batch of `shots` partial prefixes; compute each prefix's conditional P(bit=1) from the
  cached environments; draw all `shots` bits in **one vectorised numpy step**; update the
  carried left-boundary vectors for all shots at once (grouping identical prefixes so work
  is shared). This is the standard "perfect sampling of MPS" (Ferris–Vidal) done in batch.
- **Same distribution, same everything downstream.** It targets the *identical* exact-to-
  cutoff Born distribution |⟨x|ψ⟩|²; every energy, the collapse floor, CVaR, best-seen,
  restart logic and shot count (1024) are untouched. It only draws the shots **faster**.

Expected cost model: one O(n·χ²) canonical sweep + O(n·shots·χ) vectorised sampling,
versus today's O(shots · n · χ²) of independent sweeps — the χ² work moves from
"per shot" to "once per eval".

## 3. Expected speedup

- DIAGNOSIS §7.3 estimate: a **4× faster sampler → ~2.0× overall on its own** for
  chignolin (because the sampler is ~67 % of wall: 1/((0.67/4)+0.33) ≈ 1.9×). It
  **multiplies with REC 1/2**: e.g. combined ~3× (REC 1+2) × ~1.7–2.0× (REC 3) could
  approach **~5–6×** for chignolin, pushing under the 0.53 h restart-MPS floor that REC 1
  alone cannot break.
- Smaller marginal benefit for trpzip, where OpenMM (46 %) is closer to co-dominant and
  REC 2 already attacks that half; still meaningful on the 54 % MPS share.
- **Confidence: MEDIUM.** The 4× sampler factor is an estimate; the realised factor
  depends on χ (88–246 here) and how well the batched site-update vectorises. It should be
  prototyped and measured against the real per-eval times in DIAGNOSIS §3c before relying
  on the number.

## 4. Why it is NOT byte-identical (the crux)

Today's sampler is seeded deterministically (`draw_indices` derives `seed` from the
per-eval RNG and passes it to `circ.sample`). The persisted seed-0 CSVs are the exact set
of structures that *this specific sampling procedure* visited. A batched sampler consumes
randomness in a **different order** (all shots' site-0 bits, then all site-1 bits, …,
instead of shot-by-shot), so **even with the same seed it draws a different specific
multiset of bitstrings**. The distribution is the same; the realised samples are not.

Consequences:
- `vqe_bitstring`, `best_seen`, the per-eval CVaR trajectory and the full eval-set CSV
  would **all differ** from the persisted seed-0 artifacts — not because anything is wrong,
  but because a different (equally valid) sample was drawn.
- The golden check (`partest/golden_check.py`) verifies `energy(bitstring)` against the
  CSVs; it stays valid as an **energy-function** check, but it does **not** and **cannot**
  certify REC 3's *selection*, because there is no byte-identical selection to match.
- This project's reproducibility ("the gate") rests on those exact seed-0 artifacts, so
  REC 3 requires a deliberate, one-time **re-baseline**, not a silent swap.

## 5. Artifacts that would need re-baselining

If you adopt REC 3, regenerate and re-bless (the current copies live in prior-session
scratchpads; durable golden copies are in `C:\Users\abena\PFA_backups\golden\`):

- `seed0_full_eval_set.csv` (chignolin) and `trpzip_seed0_full_eval_set.csv` — the full
  bitstring→energy eval sets.
- `seed0_logged_meta.json`, `seed0_logged_run.log`, `seed0_lowtail.json` (chignolin) and
  `trpzip_seed0_summary.json`, `trpzip_seed0_lowtail.json`, `trpzip_phase3.json` (trpzip) —
  selected/best-seen bitstrings, energies, CA-RMSDs, wall times, collapse audit.
- Any downstream numbers quoted from those (README/PLAN, the ~5 Å chignolin selection, the
  4.79 Å trpzip selection, the native-vs-prediction gaps, the trpzip native-rank verdicts).
- The energy values themselves do **not** change (same force field / minimizer), so a new
  eval-set CSV built by REC 3 must still reproduce the *shared* bitstrings at max abs
  error 0.0 — a useful partial cross-check during re-baselining.

## 6. Distribution-level verification that would establish correctness

Since bit-exactness is impossible, correctness = **the batched sampler draws from the same
distribution as the current per-shot sampler**. Recommended acceptance suite (build as
`partest/` scripts, mirroring the existing guardrails):

1. **Exact-distribution match at small n.** For n ≤ ~12 qubits use `MPSSampler.probs`
   (full 2ⁿ vector). Draw a large N (e.g. 2²⁰) with both samplers and check empirical
   frequencies agree with the exact Born probabilities: per-state |p̂−p| within binomial
   tolerance, and total-variation distance between the two samplers → 0 as N grows.
   (`vqe.cvar_from_distribution` already gives the exact CVaR to compare the sampled CVaR
   against — cf. `validation.test_cvar_sampled_converges_to_exact`.)
2. **Statistical equivalence at production n (30/36q).** No tractable exact vector, so
   compare the two samplers to **each other** at fixed params: KS / chi-square /TV on the
   top-k basis-state frequencies over many seeds; assert no significant difference.
3. **Energy-invariance cross-check.** Every drawn bitstring, fed to `hamiltonian.energy`,
   must match the golden CSV at max abs error 0.0 for any bitstring present in both sets
   (proves REC 3 changed *sampling only*, not physics).
4. **CVaR-trajectory distribution.** Over ≥20 seeds, the distribution of final
   `vqe_energy` / selected CA-RMSD from REC 3 must be statistically indistinguishable from
   the current sampler's (same mean/spread within CI) — the selection is unchanged *in
   distribution* even though any single seed differs.
5. **Determinism & speed.** Same seed → same output (reproducible); and measure the real
   per-eval draw time vs DIAGNOSIS §3c to confirm the ≥4× target on this hardware.

Only after (1)–(4) pass would a re-baseline of §5 be justified.

## 7. Effort / risk

- **Effort: HIGH** (a correct batched perfect-sampler over quimb's MPS canonical form,
  plus the acceptance suite and the re-baseline). **Risk: MEDIUM** — subtle bugs in the
  batched conditionals can distort the tail that CVaR/selection depend on, and there is no
  byte-identical gate to catch them; hence the distribution suite above is mandatory.
- **Recommendation:** worthwhile *only* if you need to go below the ~2.5–3× restart floor
  for the small MPS-bound peptides and are willing to re-baseline. For larger peptides
  (≥~14 residues) OpenMM overtakes the sampler (DIAGNOSIS §6), so REC 2 matters more there
  and REC 3's payoff shrinks. **Decision left to you.**
