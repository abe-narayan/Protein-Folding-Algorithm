"""Post-optimization structural evaluation. STRICTLY SEPARATED FROM SEARCH.

Everything in this module reads native PDB structures. Nothing in this module
may be imported by hamiltonian.py or vqe.py -- validation.py enforces this by
auditing the PDB access log around optimization calls.

ALL RMSD VALUES ARE IN ANGSTROMS. There is no normalized-coordinate path.
"""
import time
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

import protein_geometry as geo
import representations as reps


def representation_ceiling(rep, native_ca: np.ndarray,
                           native_phi: Optional[np.ndarray] = None,
                           native_psi: Optional[np.ndarray] = None,
                           seed: int = 0, iterations: int = 20000) -> Dict:
    """Best CA-RMSD achievable by ANY bitstring under this representation.

    Estimated by simulated annealing directly on CA-RMSD to the native.

    WHY SEARCH RATHER THAN PROJECTION: an earlier version projected each
    residue's native torsions onto the nearest library state independently.
    That is locally optimal but NOT globally optimal, because NeRF
    accumulates coordinates along the chain -- a small angular error early
    displaces every downstream residue. At N=12 the projection produced a
    "ceiling" of 9.50 A that three optimizers beat, which is impossible for
    a true ceiling and exposed the error.

    This function reads native coordinates and is EVALUATION ONLY. It must
    never be called from the optimization path.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    is_lat = getattr(rep, "is_lattice", False)

    def score(bits: str) -> float:
        if is_lat:
            return geo.ca_rmsd(rep.decode(bits), native_ca, allow_scale=True)
        return geo.ca_rmsd(rep.build_coords(bits)["CA"], native_ca,
                           allow_scale=False)

    # Seed the search with the greedy projection when available -- it is a
    # reasonable starting point even though it is not the optimum.
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

    return {
        "ceiling_bitstring": best,
        "ceiling_ca_rmsd": float(best_s),
        "projection_ca_rmsd": float(projection_rmsd),
        "method": "annealed search on CA-RMSD ({} iters)".format(iterations),
        "runtime": time.time() - t0,
    }


def evaluate_structure(bitstring: str, rep, hamiltonian,
                       native_seq: str, native_coords: Dict[str, np.ndarray],
                       native_phi: Optional[np.ndarray] = None,
                       native_psi: Optional[np.ndarray] = None,
                       contact_threshold: float = 8.0) -> Dict:
    """Full evaluation of one predicted bitstring against a native structure."""
    n = min(len(native_seq), rep.n_residues)
    is_lat = getattr(rep, "is_lattice", False)

    pred_coords = rep.build_coords(bitstring)
    pred_ca = np.asarray(pred_coords["CA"])[:n]
    pred_cb = np.asarray(pred_coords.get("CB", pred_coords["CA"]))[:n]
    nat_ca = np.asarray(native_coords["CA"])[:n]
    nat_cb = np.asarray(native_coords["CB"])[:n]

    ca_r = geo.ca_rmsd(pred_ca, nat_ca, allow_scale=is_lat)

    if is_lat:
        bb_r = float("nan")   # lattice has no N/C atoms
        ss_pred = "unavailable"
        ss_agree = float("nan")
        pred_cb_scaled, scale = geo.kabsch_superpose_with_scale(pred_cb, nat_cb)
        pred_contacts = geo.contact_map(pred_cb_scaled, contact_threshold, 3)
    else:
        pred_bb = np.concatenate([np.asarray(pred_coords[a])[:n]
                                  for a in ("N", "CA", "C")])
        nat_bb = np.concatenate([np.asarray(native_coords[a])[:n]
                                 for a in ("N", "CA", "C")])
        bb_r = geo.rmsd(geo.kabsch_superpose(pred_bb, nat_bb), nat_bb)
        ss_pred = geo.assign_secondary_structure(pred_coords)
        ss_agree = geo.ss_agreement(
            ss_pred, geo.assign_secondary_structure(native_coords))
        scale = 1.0
        pred_contacts = geo.contact_map(pred_cb, contact_threshold, 3)

    nat_contacts = geo.contact_map(nat_cb, contact_threshold, 3)
    cp, cr, cf1 = geo.contact_metrics(pred_contacts, nat_contacts)
    lr_pred = {(i, j) for i, j in pred_contacts if j - i >= 5}
    lr_nat = {(i, j) for i, j in nat_contacts if j - i >= 5}
    _, lr_recall, _ = geo.contact_metrics(lr_pred, lr_nat)

    # --- energy comparison under the SAME Hamiltonian ---
    pred_energy = hamiltonian.energy(bitstring)
    if is_lat:
        # Lattice coordinates are dimensionless; the native cannot be scored
        # in the same units. Reported as NaN rather than a misleading number.
        native_energy = float("nan")
    else:
        native_energy = hamiltonian.energy_from_coords(
            native_coords, native_phi, native_psi)

    return {
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
        "lattice_scale_factor": float(scale),
        "n_predicted_contacts": len(pred_contacts),
        "n_native_contacts": len(nat_contacts),
    }


def summarize_seeds(rows: List[Dict], key: str) -> Dict[str, float]:
    vals = np.array([r[key] for r in rows
                     if r.get(key) is not None and np.isfinite(r[key])])
    if vals.size == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "best": float("nan"), "worst": float("nan"), "n": 0}
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "best": float(vals.min()), "worst": float(vals.max()),
            "n": int(vals.size)}