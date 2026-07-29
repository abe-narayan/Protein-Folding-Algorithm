"""Representation ceiling and predicted-vs-native metrics.

2026-07-28 changes:

* `evaluate_structure` raises on a sequence/representation length mismatch instead of
  silently truncating to the shorter one. A mismatch is a dataset bug.
* NMR ensembles are scored as ensembles (`ensemble` argument): min/mean CA-RMSD over
  deposited models, plus the ensemble's own spread, which is the resolution floor below
  which a prediction is not distinguishable from the experiment.
* `well_determined_mask` derives the structured core from the deposited models rather
  than from any assumption about a particular peptide, and `core_ca_rmsd` reports RMSD
  over it. On a 10-mer, global RMSD is dominated by flexible termini, so the global
  number understates how good a prediction of the ordered part is -- but the core must be
  identified from data, never hardcoded.
* The lattice no longer gets a free similarity scale: `representations.decode` now
  returns Angstroms, so `allow_scale` is unnecessary, and allowing it made lattice RMSD
  incomparable with the torsion arm's.
"""
import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

import protein_geometry as geo


_CEILING_CACHE: Dict[tuple, Dict] = {}

#: Across-model CA RMSF below which a residue counts as well determined, Angstroms.
CORE_RMSF_TOL = 1.5


def _ceiling_key(rep, native_ca, native_phi, native_psi, seed, iterations) -> tuple:
    """Everything the annealed ceiling search actually depends on."""
    def _b(a):
        return None if a is None else np.ascontiguousarray(
            np.asarray(a, dtype=float)).tobytes()
    return (getattr(rep, "name", type(rep).__name__), rep.n_bits,
            getattr(rep, "n_states", None), tuple(getattr(rep, "classes", ())),
            int(seed), int(iterations),
            _b(native_ca), _b(native_phi), _b(native_psi))


def clear_ceiling_cache() -> None:
    _CEILING_CACHE.clear()


def representation_ceiling(rep, native_ca: np.ndarray,
                           native_phi: Optional[np.ndarray] = None,
                           native_psi: Optional[np.ndarray] = None,
                           seed: int = 0, iterations: int = 20000) -> Dict:
    """Best CA-RMSD the representation can express, by annealed search.

    Memoized on everything the search depends on -- including the per-residue class list,
    which now affects the available states. The RNG is seeded from `seed` alone, so a
    cache hit returns exactly what a recomputation would.
    """
    key = _ceiling_key(rep, native_ca, native_phi, native_psi, seed, iterations)
    hit = _CEILING_CACHE.get(key)
    if hit is not None:
        return dict(hit)

    t0 = time.time()
    rng = np.random.default_rng(seed)
    is_lat = getattr(rep, "is_lattice", False)

    def score(bits: str) -> float:
        ca = rep.decode(bits) if is_lat else rep.build_coords(bits)["CA"]
        return geo.ca_rmsd(ca, native_ca, allow_scale=False)

    if is_lat:
        current = rep.random_bitstring(rng)
        n_slots, width, n_choices = rep.n_bonds, 2, 4
    else:
        current = rep.angles_to_bits(native_phi, native_psi)
        n_slots, width, n_choices = (rep.n_residues, rep.bits_per_residue,
                                     rep.n_states)

    cur_s = score(current)
    best, best_s = current, cur_s
    projection_rmsd = cur_s if not is_lat else float("nan")

    T0, T1 = 3.0, 1e-3
    for k in range(iterations):
        frac = k / max(1, iterations - 1)
        T = T0 * (1 - frac) + T1 * frac
        bits = list(current)
        off = int(rng.integers(0, n_slots)) * width
        bits[off:off + width] = list(
            format(int(rng.integers(0, n_choices)), f"0{width}b"))
        cand = "".join(bits)
        cs = score(cand)
        if cs < cur_s or rng.random() < math.exp(-(cs - cur_s) / max(T, 1e-9)):
            current, cur_s = cand, cs
            if cs < best_s:
                best, best_s = cand, cs

    out = {
        "ceiling_bitstring": best,
        "ceiling_ca_rmsd": float(best_s),
        "projection_ca_rmsd": float(projection_rmsd),
        "method": "annealed search on CA-RMSD ({} iters)".format(iterations),
        "runtime": time.time() - t0,
    }
    _CEILING_CACHE[key] = out
    return dict(out)

def _superpose_on_subset(mobile: np.ndarray, target: np.ndarray,
                         mask: np.ndarray) -> np.ndarray:
    """Kabsch fit computed from the masked residues only, applied to every residue."""
    P = np.asarray(mobile, dtype=float)
    Q = np.asarray(target, dtype=float)
    pc = P[mask].mean(axis=0)
    qc = Q[mask].mean(axis=0)
    H = (P[mask] - pc).T @ (Q[mask] - qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (R @ (P - pc).T).T + qc


def well_determined_mask(ensemble_ca: Sequence[np.ndarray],
                         tol: float = CORE_RMSF_TOL,
                         max_rounds: int = 10) -> np.ndarray:
    """Boolean mask of residues the experiment actually resolves.

    Superposes every deposited model on the first and flags residues whose across-model
    root-mean-square fluctuation is below `tol`. This is a data-derived definition of
    "the structured core" -- it works for any peptide and requires no assumption about
    which residues matter. With a single model everything is flagged as determined.

    The superposition is *iterated on the current core*. Fitting over all residues lets
    disordered termini dominate the rotation and smear the ordered part past `tol` -- on a
    12-residue test ensemble with 4 A terminal noise and 0.15 A core noise, a single
    all-residue fit loses 2 of the 8 genuinely ordered residues. Refitting on the retained
    subset is what makes the core self-consistent rather than an artifact of the flexible
    ends.
    """
    models = [np.asarray(m, dtype=float) for m in ensemble_ca]
    if not models:
        raise ValueError("well_determined_mask needs at least one model")
    n = len(models[0])
    if len(models) < 2:
        return np.ones(n, dtype=bool)
    ref = models[0]
    mask = np.ones(n, dtype=bool)
    for _ in range(max_rounds):
        aligned = np.stack([_superpose_on_subset(m, ref, mask) for m in models])
        rmsf = np.sqrt(((aligned - aligned.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))
        new = rmsf < tol
        if np.array_equal(new, mask):
            return new
        if new.sum() < 3:
            return new          # nothing resolvable; core_ca_rmsd reports nan
        mask = new
    return mask

def core_ca_rmsd(pred_ca: np.ndarray, native_ca: np.ndarray,
                 mask: np.ndarray) -> float:
    """CA-RMSD over the masked residues only, superposed on those residues."""
    if mask.sum() < 3:
        return float("nan")
    return geo.ca_rmsd(np.asarray(pred_ca)[mask], np.asarray(native_ca)[mask])


def evaluate_structure(bitstring: str, rep, hamiltonian,
                       native_seq: str, native_coords: Dict[str, np.ndarray],
                       native_phi: Optional[np.ndarray] = None,
                       native_psi: Optional[np.ndarray] = None,
                       contact_threshold: float = 8.0,
                       ensemble: Optional[List] = None) -> Dict:
    """Full evaluation of one predicted bitstring against a native structure.

    `ensemble`, when given, is the list returned by `dataset.load_ensemble`; the
    ensemble-aware RMSD numbers are then reported alongside the model-1 value.
    """
    if len(native_seq) != rep.n_residues:
        raise ValueError(
            f"native sequence length {len(native_seq)} != representation "
            f"n_residues {rep.n_residues}. Previously this was silently truncated to "
            "the shorter of the two, which quietly changed what the RMSD measured.")
    n = rep.n_residues
    is_lat = getattr(rep, "is_lattice", False)

    pred_coords = rep.build_coords(bitstring)
    pred_ca = np.asarray(pred_coords["CA"])[:n]
    pred_cb = np.asarray(pred_coords.get("CB", pred_coords["CA"]))[:n]
    nat_ca = np.asarray(native_coords["CA"])[:n]
    nat_cb = np.asarray(native_coords["CB"])[:n]

    ca_r = geo.ca_rmsd(pred_ca, nat_ca)

    if is_lat:
        bb_r = float("nan")
        ss_pred = "unavailable"
        ss_agree = float("nan")
        pred_contacts = geo.contact_map(pred_cb, contact_threshold, 3)
    else:
        pred_bb = np.concatenate([np.asarray(pred_coords[a])[:n]
                                  for a in ("N", "CA", "C")])
        nat_bb = np.concatenate([np.asarray(native_coords[a])[:n]
                                 for a in ("N", "CA", "C")])
        bb_r = geo.rmsd(geo.kabsch_superpose(pred_bb, nat_bb), nat_bb)
        ss_pred = geo.assign_secondary_structure(pred_coords)
        ss_agree = geo.ss_agreement(
            ss_pred, geo.assign_secondary_structure(native_coords))
        pred_contacts = geo.contact_map(pred_cb, contact_threshold, 3)

    nat_contacts = geo.contact_map(nat_cb, contact_threshold, 3)
    cp, cr, cf1 = geo.contact_metrics(pred_contacts, nat_contacts)
    lr_pred = {(i, j) for i, j in pred_contacts if j - i >= 5}
    lr_nat = {(i, j) for i, j in nat_contacts if j - i >= 5}
    _, lr_recall, _ = geo.contact_metrics(lr_pred, lr_nat)

    pred_energy = hamiltonian.energy(bitstring)
    if is_lat:
        native_energy = float("nan")
    else:
        native_energy = hamiltonian.energy_from_coords(
            native_coords, native_phi, native_psi)

    out = {
        "bitstring": bitstring,
        "ca_rmsd_angstrom": float(ca_r),
        "backbone_rmsd_angstrom": float(bb_r),
        "contact_precision": float(cp),
        "contact_recall": float(cr),
        "contact_f1": float(cf1),
        "longrange_contact_recall": float(lr_recall),
        "ss_predicted": ss_pred,
        "ss_agreement": float(ss_agree),
        "rg_predicted": float(geo.radius_of_gyration(pred_ca)),
        "rg_native": float(geo.radius_of_gyration(nat_ca)),
        "predicted_energy": float(pred_energy),
        "native_energy": float(native_energy),
        "energy_gap_pred_minus_native": float(pred_energy - native_energy),
        "n_predicted_contacts": len(pred_contacts),
        "n_native_contacts": len(nat_contacts),
    }

    if ensemble:
        models = [np.asarray(c["CA"])[:n] for _, c, _, _ in ensemble]
        ens = geo.ca_rmsd_to_ensemble(pred_ca, models)
        mask = well_determined_mask(models)
        out.update({
            "ca_rmsd_ensemble_min": ens["min"],
            "ca_rmsd_ensemble_mean": ens["mean"],
            "ca_rmsd_best_model": ens["best_model"],
            "n_native_models": ens["n_models"],
            "ensemble_spread": geo.ensemble_spread(models),
            "n_core_residues": int(mask.sum()),
            # Measured against the same model `ca_rmsd_ensemble_min` reports, so the two
            # numbers describe one comparison rather than two different ones.
            "core_ca_rmsd": core_ca_rmsd(pred_ca, models[ens["best_model"]], mask),
        })
    return out


def summarize_seeds(rows: List[Dict], key: str) -> Dict[str, float]:
    vals = np.array([r[key] for r in rows
                     if r.get(key) is not None and np.isfinite(r[key])])
    if vals.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "best": float("nan"), "worst": float("nan"), "n": 0}
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "best": float(vals.min()), "worst": float(vals.max()),
            "n": int(vals.size)}
