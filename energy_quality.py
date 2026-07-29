"""Is this objective worth minimising?

The repo could measure whether a search found the minimum (`classical_baselines`,
`validation`) and whether a prediction was close to native (`evaluation`). It could not
measure whether the *function being minimised points at the answer* -- so
Spearman(energy, RMSD) = -0.351 survived undetected through an entire budget-fairness
project, and every number that project produced was denominated in an inverted currency.
Finding it took a hand-written one-off script and a markdown write-up.

This module makes that a first-class, testable property. `validation.py` gates on it, so
a regression here fails the suite instead of quietly inverting the results.

Three numbers decide whether a weight vector is usable:

    spearman    rank correlation of energy with CA-RMSD-to-native. Must be > 0, and a
                model that is any use has it well above 0.
    enrichment  mean RMSD over all structures minus mean RMSD over the lowest-energy
                fraction. Positive means optimising the energy moves you toward the
                native fold. The pre-rewrite model scored -0.25 A.
    native_pct  percentage of the space scoring *better* than the snapped native.
                Pre-rewrite: 12.05%.

`calibrate_weights` then sets the free weights from a *set* of sequences by variance
balancing, so no weight is ever fitted against a single target structure. Use
`holdout_report` to calibrate on train clusters and report on held-out ones -- that is
the only form in which a re-weighting is a result rather than a fit.
"""
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import protein_geometry as geo
import representations as reps
import energy_terms as et
import hamiltonian as ham


__all__ = ["spearman", "energy_quality", "term_statistics", "variance_shares",
           "separable_share", "calibrate_weights", "holdout_report", "print_quality",
           "print_terms"]

#: Fraction of the space treated as "what a converged search returns" when computing
#: enrichment.
LOW_ENERGY_FRACTION = 0.001
#: Minimum sample when the fraction rounds too small.
MIN_LOW_ENERGY = 100


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation. No scipy dependency; ties broken by argsort order."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra @ ra) * (rb @ rb)))
    return float((ra @ rb) / denom) if denom > 0 else float("nan")


#: Enumerate only below this many bits; sample above it. 20 bits is ~1e6 structures, a
#: couple of minutes single-threaded. Chi1 encoding pushes a 10-mer with one aromatic pair
#: to 22 bits (4.2e6), which is 4x the work for no extra information about CA-RMSD --
#: chi1 does not move any CA atom. Use `diagnose_energy_model.py --workers N` for a genuine
#: exhaustive pass.
ENUMERATE_MAX_BITS = 20


def _scan(H, rep, native_ca: np.ndarray, n_sample: Optional[int],
          seed: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return (energies, rmsds, exhaustive). Enumerates when the space is small."""
    exhaustive = n_sample is None and rep.n_bits <= ENUMERATE_MAX_BITS
    nat = np.asarray(native_ca, dtype=float)

    def rmsd_of(bits: str) -> float:
        ca = np.asarray(rep.build_coords(bits)["CA"], dtype=float)[:len(nat)]
        return geo.ca_rmsd(ca, nat[:len(ca)])

    if exhaustive:
        total = 1 << rep.n_bits
        fmt = f"0{rep.n_bits}b"
        e = np.empty(total, dtype=np.float32)
        r = np.empty(total, dtype=np.float32)
        for k in range(total):
            bits = format(k, fmt)
            e[k] = H.energy(bits)
            r[k] = rmsd_of(bits)
        return e, r, True

    n = int(n_sample or 20000)
    rng = np.random.default_rng(seed)
    e = np.empty(n, dtype=np.float32)
    r = np.empty(n, dtype=np.float32)
    for k in range(n):
        bits = rep.random_bitstring(rng)
        e[k] = H.energy(bits)
        r[k] = rmsd_of(bits)
    return e, r, False


def energy_quality(H, rep, native_ca: np.ndarray,
                   native_phi=None, native_psi=None,
                   n_sample: Optional[int] = None, seed: int = 0) -> Dict:
    """Correlation of an energy model with distance-to-native.

    Enumerates the configuration space when `rep.n_bits <= ENUMERATE_MAX_BITS` and
    `n_sample` is None, otherwise draws `n_sample` random structures. Sampling is enough
    for the correlation statistics; `native_pct` from a sample is an estimate and is
    flagged as such by the `exhaustive` field.
    """
    energies, rmsds, exhaustive = _scan(H, rep, native_ca, n_sample, seed)
    finite = np.isfinite(energies)
    energies_f, rmsds_f = energies[finite], rmsds[finite]

    k = max(MIN_LOW_ENERGY, int(LOW_ENERGY_FRACTION * energies_f.size))
    k = min(k, energies_f.size)
    low = np.argsort(energies_f)[:k]

    out = {
        "n_structures": int(energies.size),
        "exhaustive": bool(exhaustive),
        "spearman": spearman(energies_f, rmsds_f),
        "mean_rmsd_all": float(rmsds_f.mean()),
        "mean_rmsd_low_energy": float(rmsds_f[low].mean()),
        "enrichment": float(rmsds_f.mean() - rmsds_f[low].mean()),
        "n_low_energy": int(k),
        "best_rmsd_anywhere": float(rmsds_f.min()),
        "e_global_min": float(energies_f.min()),
        "rmsd_at_global_min": float(rmsds_f[int(np.argmin(energies_f))]),
        "e_at_best_rmsd": float(energies_f[int(np.argmin(rmsds_f))]),
    }

    if native_phi is not None and not getattr(rep, "is_lattice", False):
        snap = rep.angles_to_bits(native_phi, native_psi)
        e_snap = H.energy(snap)
        out["e_native_snapped"] = float(e_snap)
        out["native_snapped_rmsd"] = float(
            geo.ca_rmsd(rep.build_coords(snap)["CA"], np.asarray(native_ca, float)))
        out["native_pct"] = float(100.0 * (energies_f < e_snap).sum() / energies_f.size)
    return out


#: Fraction of random structures kept as "feasible". Chosen as a percentile rather than
#: an absolute steric cutoff so the filter adapts to whatever the steric term's scale
#: happens to be -- an absolute threshold silently keeps everything or nothing when the
#: term is reparameterised.
FEASIBLE_FRACTION = 0.5

#: Ceiling on the separable term's share of the discriminating variance. The pre-rewrite
#: model had `torsion` outvoting `contact` and `hbond` by roughly 8:1 and 23:1, and a
#: separable term's minimum is reached residue-by-residue -- so a landscape it dominates
#: has no fold in it at any weight. This is the invariant, not a tuning preference.
MAX_SEPARABLE_SHARE = 0.35


def term_statistics(sequences: Sequence[str], n_states: int = 4,
                    n_sample: int = 2000, seed: int = 0,
                    feasible_fraction: float = FEASIBLE_FRACTION
                    ) -> Dict[str, Dict[str, float]]:
    """Unweighted mean/std of every term over random *feasible* structures.

    Pooled across `sequences`, so the result describes the term rather than one peptide.

    "Feasible" means in the least-clashing `feasible_fraction` of the sample by steric
    energy. Filtering matters more than it looks: across *all* random structures steric
    carried 87% of the variance, but it is exactly zero for both the native fold and the
    global minimum, so among the clash-free structures a search actually explores it is
    flat. Balancing against unfiltered variance balances against a term that does not vary
    where it matters.
    """
    rows: List[Dict[str, float]] = []
    for si, seq in enumerate(sequences):
        rep = reps.TorsionStateRepresentation(len(seq), n_states=n_states,
                                              sequence=seq)
        H = ham.FoldingHamiltonian(seq, rep)
        rng = np.random.default_rng(np.random.SeedSequence([seed, si]))
        seq_rows = [H.components(rep.random_bitstring(rng)) for _ in range(n_sample)]
        # Percentile per sequence, so a long peptide with systematically more contacts
        # does not swamp a short one.
        steric = np.array([r["steric"] for r in seq_rows], dtype=float)
        k = max(1, int(round(feasible_fraction * len(seq_rows))))
        keep = np.argsort(steric, kind="stable")[:k]
        rows.extend(seq_rows[i] for i in keep)

    out = {}
    for t in et.TERM_NAMES:
        arr = np.array([float(r.get(t, 0.0)) for r in rows], dtype=float)
        if arr.size == 0:
            arr = np.zeros(1)
        out[t] = {"mean": float(arr.mean()), "std": float(arr.std()),
                  "n": int(arr.size)}
    return out


def variance_shares(stats: Dict[str, Dict[str, float]],
                    weights: Dict[str, float]) -> Dict[str, float]:
    """Each discriminating term's share of the weighted variance.

    `steric` is excluded -- it is a feasibility filter, not a discriminator
    (`energy_terms.FILTER_TERMS`).
    """
    scaled = {t: weights.get(t, 0.0) * stats[t]["std"]
              for t in et.TERM_NAMES if t not in et.FILTER_TERMS}
    total = sum(v ** 2 for v in scaled.values()) or 1.0
    return {t: (v ** 2) / total for t, v in scaled.items()}


def separable_share(stats: Dict[str, Dict[str, float]],
                    weights: Dict[str, float]) -> float:
    """Fraction of the discriminating variance carried by separable terms."""
    shares = variance_shares(stats, weights)
    return float(sum(shares[t] for t in et.SEPARABLE_TERMS if t in shares))


def calibrate_weights(sequences: Sequence[str], n_states: int = 4,
                      n_sample: int = 2000, seed: int = 0,
                      base: Optional[Dict[str, float]] = None,
                      free: Sequence[str] = et.FREE_WEIGHTS,
                      max_ratio: float = 3.0,
                      max_separable_share: float = MAX_SEPARABLE_SHARE
                      ) -> Dict[str, float]:
    """Variance-balance the free weights over a *set* of sequences.

    Two target-independent criteria, in order:

    1. **Balance.** Each free weight is set so that `weight * std(term)` matches the
       median `weight * std(term)` of the pinned terms, then clamped to `max_ratio` of its
       starting value so one pass cannot swing a weight arbitrarily far.
    2. **The separable term may not dominate.** `torsion` is a sum of independent
       per-residue penalties, so its minimum is reached residue by residue and a landscape
       it dominates has no fold in it *at any weight*. Its weight is scaled down until it
       carries at most `max_separable_share` of the discriminating variance. This is the
       invariant that the pre-rewrite model violated -- it is not a preference.

    Pinned terms keep their `base` weights because their scale is fixed by their units:
    `contact` is the MJ potential at unit weight, `hbond` is DSSP kcal/mol at unit weight,
    `electrostatic` is a screened Coulomb in kcal/mol, and `steric` must dominate because
    hard-core overlap is forbidden rather than merely costly.

    No native structure enters this function -- the signature takes sequences only, and
    `validation.test_calibration_is_sequence_set_derived` asserts that it reads no PDB.
    `docs/energy-model-diagnosis.md` warns that "with seven weights and one target
    structure you can always succeed, and you will have fit the answer rather than a
    model"; `holdout_report` is what closes that off, by reporting on clusters that were
    not used for calibration.
    """
    base = dict(et.DEFAULT_WEIGHTS if base is None else base)
    stats = term_statistics(sequences, n_states=n_states, n_sample=n_sample, seed=seed)

    pinned = [t for t in et.TERM_NAMES
              if t not in free and t not in et.FILTER_TERMS]
    scales = [base[t] * stats[t]["std"] for t in pinned if stats[t]["std"] > 0]
    out = dict(base)

    if scales:
        target = float(np.median(scales))
        for t in free:
            s = stats[t]["std"]
            if s <= 0:
                continue
            lo, hi = base[t] / max_ratio, base[t] * max_ratio
            out[t] = float(min(hi, max(lo, target / s)))

    # Criterion 2. Solve directly rather than iterating: with v_sep the separable terms'
    # weighted variance and v_cpl the coupled terms', share = v_sep / (v_sep + v_cpl), so
    # holding the coupled terms fixed the admissible separable variance is
    # v_cpl * m / (1 - m). Scale every separable weight by the same factor.
    shares = variance_shares(stats, out)
    v_sep = sum(shares[t] for t in et.SEPARABLE_TERMS if t in shares)
    v_cpl = sum(shares[t] for t in shares if t not in et.SEPARABLE_TERMS)
    m = float(max_separable_share)
    if v_sep > m and v_cpl > 0 and 0 < m < 1:
        allowed = v_cpl * m / (1.0 - m)
        factor = math.sqrt(allowed / v_sep)
        for t in et.SEPARABLE_TERMS:
            if t in out:
                out[t] = float(out[t] * factor)
    return out


def holdout_report(train: Sequence, test: Sequence, n_states: int = 4,
                   n_sample: Optional[int] = None, calib_sample: int = 2000,
                   seed: int = 0, out_json: Optional[str] = None) -> Dict:
    """Calibrate on `train`, report energy quality on `test`.

    `train`/`test` are `dataset.PeptideEntry` lists that must come from disjoint identity
    clusters -- use `dataset.split_by_cluster` and assert `dataset.check_no_cluster_leak`
    before calling. Reporting a calibrated model on the sequences it was calibrated on is
    not a result.
    """
    import dataset as ds

    train_seqs = [e.sequence for e in train]
    weights = calibrate_weights(train_seqs, n_states=n_states,
                                n_sample=calib_sample, seed=seed)

    rows = []
    for split_name, entries in (("train", train), ("test", test)):
        for entry in entries:
            seq = entry.sequence
            _, ncoords, nphi, npsi = ds.load_native(entry)
            rep = reps.TorsionStateRepresentation(len(seq), n_states=n_states,
                                                  sequence=seq)
            H = ham.FoldingHamiltonian(seq, rep, weights=weights)
            q = energy_quality(H, rep, ncoords["CA"], nphi, npsi,
                               n_sample=n_sample, seed=seed)
            q.update(split=split_name, protein=entry.pdb_id, sequence=seq,
                     cluster=entry.cluster)
            rows.append(q)

    test_rows = [r for r in rows if r["split"] == "test"]
    summary = {
        "weights": weights,
        "n_train": len(train), "n_test": len(test),
        "test_mean_spearman": float(np.nanmean([r["spearman"] for r in test_rows]))
                              if test_rows else float("nan"),
        "test_mean_enrichment": float(np.nanmean([r["enrichment"] for r in test_rows]))
                                if test_rows else float("nan"),
        "rows": rows,
    }
    if out_json:
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary


def print_quality(q: Dict, label: str = "") -> None:
    tag = f" -- {label}" if label else ""
    mode = "exhaustive" if q.get("exhaustive") else "sampled"
    print(f"  ENERGY QUALITY{tag}  ({q['n_structures']:,} structures, {mode})")
    print(f"    Spearman(energy, RMSD)     : {q['spearman']:+.4f}   "
          f"(> 0 required; +1 = energy tracks correctness)")
    print(f"    mean RMSD, all             : {q['mean_rmsd_all']:.2f} A")
    print(f"    mean RMSD, lowest-energy   : {q['mean_rmsd_low_energy']:.2f} A "
          f"({q['n_low_energy']} structures)")
    verdict = ("helps" if q["enrichment"] > 0.25 else
               "no useful signal" if q["enrichment"] > -0.25 else "ANTI-correlated")
    print(f"    enrichment                 : {q['enrichment']:+.2f} A   ({verdict})")
    if "native_pct" in q:
        print(f"    structures better than native: {q['native_pct']:.2f} %")
        print(f"    native (snapped)           : E {q['e_native_snapped']:.4f}  "
              f"RMSD {q['native_snapped_rmsd']:.2f} A")
    print(f"    global minimum             : E {q['e_global_min']:.4f}  "
          f"RMSD {q['rmsd_at_global_min']:.2f} A")
    print(f"    best RMSD anywhere         : {q['best_rmsd_anywhere']:.2f} A  "
          f"(E {q['e_at_best_rmsd']:.4f})")


def print_terms(stats: Dict[str, Dict[str, float]],
                weights: Optional[Dict[str, float]] = None) -> None:
    """Per-term variance table. `steric` is shown but excluded from the shares, since it
    is a feasibility filter rather than a discriminator."""
    weights = dict(et.DEFAULT_WEIGHTS if weights is None else weights)
    shares = variance_shares(stats, weights)
    print(f"  {'term':<16}{'kind':>10}{'weight':>8}{'mean':>10}{'std':>10}"
          f"{'w*std':>10}{'var share':>11}")
    for t in et.TERM_NAMES:
        s = stats[t]
        w = weights.get(t, 0.0)
        kind = ("filter" if t in et.FILTER_TERMS
                else "separable" if t in et.SEPARABLE_TERMS else "coupled")
        share = "" if t in et.FILTER_TERMS else f"{100.0 * shares[t]:>9.1f} %"
        print(f"  {t:<16}{kind:>10}{w:>8.3f}{s['mean']:>10.3f}{s['std']:>10.3f}"
              f"{w * s['std']:>10.3f}{share:>11}")
    sep = separable_share(stats, weights)
    verdict = "OK" if sep <= MAX_SEPARABLE_SHARE else "TOO HIGH -- no fold in this landscape"
    print(f"  {'separable share':<16}{'':>10}{'':>8}{'':>10}{'':>10}{'':>10}"
          f"{100.0 * sep:>9.1f} %   [{verdict}]")
