"""Rigorous structural validation of the lattice folding model.

A single RMSD-to-reference number is not a scientific result on this lattice: the
energy ground state is highly degenerate (many symmetry-equivalent folds with
different RMSDs), so a bare RMSD reflects which degenerate fold the sampler
happened to pick, not the model's skill. This module replaces that with proper
statistics:

  * a null RMSD distribution over the whole (enumerated or sampled) fold space,
  * an empirical p-value / percentile for any selected fold against that null,
  * the Spearman correlation between energy and RMSD -- the model's *predictive
    validity*: only a positive correlation means "lower energy => more native",
  * the effect of imposing an experimentally-known disulfide bond as a hard
    topological constraint, tested with a Mann-Whitney U test against the
    unconstrained ensemble.

Everything here is measured, not tuned: no parameter is fit to the reference
structure. The functions return numbers whichever way they point.
"""

import itertools

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

from encoding import bits_to_coords
from hamiltonian import path_energy_specific, find_disulfide_pairs
from real_structure import get_ca_coords, normalize_coords, kabsch_align, rmsd


# On this lattice a "contact" is squared-distance 8 (the nearest non-bonded
# shell). A disulfide is a covalent bond, so we treat two cysteines as bonded
# when they are within that shell or closer.
CONTACT_D2 = 8


def enumerate_folds(sequence, max_qubits=20):
    """Yield every non-self-overlapping fold as (bitstring, coords).

    Exhaustive for small sequences. Raises for sequences whose fold space is too
    large to enumerate exactly -- use ``sample_folds`` there instead.
    """
    n_qubits = 2 * (len(sequence) - 1)
    if n_qubits > max_qubits:
        raise ValueError(
            f"{n_qubits} qubits is too large to enumerate exactly; "
            f"use sample_folds()."
        )
    fmt = f"0{n_qubits}b"
    for idx in range(2 ** n_qubits):
        bits = format(idx, fmt)
        coords = bits_to_coords(bits)
        if len(set(coords)) == len(coords):  # reject self-overlapping chains
            yield bits, coords


def sample_folds(sequence, n_samples, rng):
    """Yield ``n_samples`` random non-overlapping folds (for large sequences)."""
    n_qubits = 2 * (len(sequence) - 1)
    fmt = f"0{n_qubits}b"
    seen = 0
    while seen < n_samples:
        idx = int(rng.integers(0, 2 ** n_qubits))
        bits = format(idx, fmt)
        coords = bits_to_coords(bits)
        if len(set(coords)) == len(coords):
            seen += 1
            yield bits, coords


def cys_pair_d2(coords, pair):
    i, j = pair
    return sum((coords[i][k] - coords[j][k]) ** 2 for k in range(3))


def build_fold_table(sequence, pdb_path, chain_id="A", n_qubits_exact=20,
                     n_samples=50000, seed=0):
    """Compute per-fold RMSD-to-reference and disulfide geometry once.

    Returns a dict of parallel numpy arrays plus metadata. Energies for specific
    model variants are computed cheaply on top of this table by ``variant_energy``.
    """
    real = normalize_coords(get_ca_coords(pdb_path, chain_id=chain_id))
    n_qubits = 2 * (len(sequence) - 1)
    exact = n_qubits <= n_qubits_exact
    source = (enumerate_folds(sequence) if exact
              else sample_folds(sequence, n_samples, np.random.default_rng(seed)))

    disulfides = find_disulfide_pairs(sequence)

    bitstrings, rmsds, coords_list, dsat = [], [], [], []
    for bits, coords in source:
        aligned = kabsch_align(normalize_coords(coords), real)
        bitstrings.append(bits)
        rmsds.append(rmsd(aligned, real))
        coords_list.append(coords)
        # disulfide satisfied = every inferred Cys-Cys pair within the bond shell
        dsat.append(all(cys_pair_d2(coords, p) <= CONTACT_D2 for p in disulfides)
                    if disulfides else False)

    return {
        "sequence": sequence,
        "pdb_path": pdb_path,
        "exact": exact,
        "n_folds": len(bitstrings),
        "bitstrings": bitstrings,
        "coords": coords_list,
        "rmsd": np.array(rmsds),
        "disulfide_pairs": disulfides,
        "disulfide_satisfied": np.array(dsat, dtype=bool),
        "best_possible_rmsd": float(np.min(rmsds)) if rmsds else float("nan"),
    }


def variant_energy(table, compactness_weight=0.5, disulfide_pairs=None,
                   disulfide_weight=0.5):
    return np.array([
        path_energy_specific(
            b, table["sequence"],
            compactness_weight=compactness_weight,
            disulfide_pairs=disulfide_pairs,
            disulfide_weight=disulfide_weight,
        )
        for b in table["bitstrings"]
    ])


def empirical_pvalue(value, null):
    """Fraction of the null distribution at least as native (<=) as ``value``."""
    return float((null <= value).mean())


def predictive_validity(energies, rmsds):
    """Spearman(energy, RMSD). Positive => lower energy predicts lower RMSD."""
    rho, p = spearmanr(energies, rmsds)
    return float(rho), float(p)


def ground_state_ensemble(energies, rmsds, tol=1e-6):
    """RMSD stats over all degenerate energy minima (the honest 'selected' set)."""
    mn = energies.min()
    gs = np.flatnonzero(energies <= mn + tol)
    r = rmsds[gs]
    return {
        "min_energy": float(mn),
        "degeneracy": int(gs.size),
        "rmsd_min": float(r.min()),
        "rmsd_median": float(np.median(r)),
        "rmsd_max": float(r.max()),
    }


def disulfide_effect(table):
    """Mann-Whitney U test: does imposing the disulfide shift RMSD toward native?

    Compares the RMSD distribution of disulfide-satisfying folds against the
    complement. One-sided (constrained folds are *more* native = lower RMSD).
    """
    r = table["rmsd"]
    sat = table["disulfide_satisfied"]
    if not sat.any() or sat.all():
        return None
    con, unc = r[sat], r[~sat]
    u, p = mannwhitneyu(con, unc, alternative="less")
    return {
        "n_constrained": int(sat.sum()),
        "n_unconstrained": int((~sat).sum()),
        "median_constrained": float(np.median(con)),
        "median_unconstrained": float(np.median(unc)),
        "best_constrained": float(con.min()),
        "mannwhitney_u": float(u),
        "p_value": float(p),
    }
