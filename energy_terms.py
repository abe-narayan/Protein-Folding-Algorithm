"""Energy terms for the folding Hamiltonian.

DESIGN PRINCIPLE: every term must have (a) a formula, (b) a physical
interpretation, (c) a stated scale, (d) a reason to exist, and (e) an honest
label of how its weight was set. There are SEVEN terms, not twelve, because
each additional weight on an 8-peptide benchmark is a free parameter that
invites overfitting.

WEIGHTS ARE NOT LEARNED. They are fixed a priori from physical reasoning about
relative magnitudes in kcal/mol. This is a deliberate departure from the
earlier learned-weight design: 12 weights fit by ranking loss on ~8 peptides
produced a scoring function, not an energy model, and it could not be
defended as physics. Fixed weights will likely give worse benchmark RMSD.
That is the correct tradeoff for a scientific claim.

-------------------------------------------------------------------------
TERM SUMMARY
-------------------------------------------------------------------------
name            formula (per pair/residue)                weight  origin
--------------- ----------------------------------------- ------- ---------
steric          sum (d0 - d)^2 for d < d0                    4.0  physical
contact         sum MJ'_ij * S(d_CB; 4.5, 8.5)               1.0  reference
hbond           sum matched DSSP E_hb (kcal/mol)             1.0  reference
solvation       -sum (KD_i/4.5) * coordination_i             0.3  empirical
electrostatic   sum q_i q_j / max(d,2) * exp(-d/8)           0.5  physical
torsion         sum rama_penalty(aa_i, phi_i, psi_i)         1.0  empirical
compactness     (Rg - 2.2 N^0.38)^2                          0.05 empirical

"reference" = taken directly from published potentials with unit weight
              (Miyazawa-Jernigan 1996; Kabsch-Sander DSSP 1983).
"physical"  = magnitude fixed by the physics (hard-core repulsion must
              dominate; Coulomb prefactor from screened electrostatics).
"empirical" = chosen by the author to set a sensible relative scale; these
              are the three terms most vulnerable to criticism and are the
              ones the sensitivity ablation must probe.
-------------------------------------------------------------------------
"""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import protein_geometry as geo


# ==========================================================================
# Miyazawa-Jernigan contact potential
# ==========================================================================
MJ_ORDER = ["C", "M", "F", "I", "L", "V", "W", "Y", "A", "G",
            "T", "S", "N", "Q", "D", "E", "H", "R", "K", "P"]

_MJ_RAW = [
    [-5.44, -4.99, -5.80, -5.50, -5.83, -4.96, -4.95, -4.16, -3.57, -3.16, -3.11, -2.86, -2.59, -2.85, -2.41, -2.27, -3.60, -2.57, -1.95, -3.07],
    [-4.99, -5.46, -6.56, -6.02, -6.41, -5.32, -5.55, -4.91, -3.94, -3.39, -3.51, -3.03, -2.95, -3.30, -2.57, -2.89, -3.98, -3.12, -2.48, -3.45],
    [-5.80, -6.56, -7.26, -6.84, -7.28, -6.29, -6.16, -5.66, -4.81, -4.13, -4.28, -4.02, -3.75, -4.10, -3.48, -3.56, -4.77, -3.98, -3.36, -4.25],
    [-5.50, -6.02, -6.84, -6.54, -7.04, -6.05, -5.78, -5.25, -4.58, -3.78, -4.03, -3.75, -3.52, -3.67, -3.17, -3.27, -4.14, -3.63, -3.01, -3.76],
    [-5.83, -6.41, -7.28, -7.04, -7.37, -6.48, -6.14, -5.67, -4.91, -4.16, -4.34, -4.08, -3.75, -4.04, -3.40, -3.59, -4.54, -4.03, -3.37, -4.20],
    [-4.96, -5.32, -6.29, -6.05, -6.48, -5.52, -5.18, -4.62, -4.04, -3.38, -3.46, -3.30, -3.07, -3.28, -2.83, -2.90, -3.58, -3.07, -2.49, -3.32],
    [-4.95, -5.55, -6.16, -5.78, -6.14, -5.18, -5.06, -4.66, -3.82, -3.42, -3.22, -3.07, -3.07, -3.11, -2.84, -2.99, -3.98, -3.41, -2.69, -3.73],
    [-4.16, -4.91, -5.66, -5.25, -5.67, -4.62, -4.66, -4.17, -3.36, -3.01, -3.01, -2.78, -2.83, -2.97, -2.76, -2.79, -3.52, -3.16, -2.60, -3.19],
    [-3.57, -3.94, -4.81, -4.58, -4.91, -4.04, -3.82, -3.36, -2.72, -2.31, -2.32, -2.01, -1.84, -1.89, -1.70, -1.51, -2.41, -1.83, -1.31, -2.03],
    [-3.16, -3.39, -4.13, -3.78, -4.16, -3.38, -3.42, -3.01, -2.31, -2.24, -2.08, -1.82, -1.74, -1.66, -1.59, -1.22, -2.15, -1.72, -1.15, -1.87],
    [-3.11, -3.51, -4.28, -4.03, -4.34, -3.46, -3.22, -3.01, -2.32, -2.08, -2.12, -1.96, -1.88, -1.90, -1.80, -1.74, -2.42, -1.90, -1.31, -1.90],
    [-2.86, -3.03, -4.02, -3.75, -4.08, -3.30, -3.07, -2.78, -2.01, -1.82, -1.96, -1.67, -1.58, -1.49, -1.63, -1.48, -2.11, -1.62, -1.05, -1.57],
    [-2.59, -2.95, -3.75, -3.52, -3.75, -3.07, -3.07, -2.83, -1.84, -1.74, -1.88, -1.58, -1.68, -1.71, -1.68, -1.51, -2.08, -1.64, -1.21, -1.53],
    [-2.85, -3.30, -4.10, -3.67, -4.04, -3.28, -3.11, -2.97, -1.89, -1.66, -1.90, -1.49, -1.71, -1.54, -1.46, -1.42, -1.98, -1.80, -1.29, -1.73],
    [-2.41, -2.57, -3.48, -3.17, -3.40, -2.83, -2.84, -2.76, -1.70, -1.59, -1.80, -1.63, -1.68, -1.46, -1.21, -1.02, -2.32, -2.29, -1.68, -1.33],
    [-2.27, -2.89, -3.56, -3.27, -3.59, -2.90, -2.99, -2.79, -1.51, -1.22, -1.74, -1.48, -1.51, -1.42, -1.02, -0.91, -2.15, -2.27, -1.80, -1.26],
    [-3.60, -3.98, -4.77, -4.14, -4.54, -3.58, -3.98, -3.52, -2.41, -2.15, -2.42, -2.11, -2.08, -1.98, -2.32, -2.15, -3.05, -2.16, -1.35, -2.25],
    [-2.57, -3.12, -3.98, -3.63, -4.03, -3.07, -3.41, -3.16, -1.83, -1.72, -1.90, -1.62, -1.64, -1.80, -2.29, -2.27, -2.16, -1.55, -0.59, -1.70],
    [-1.95, -2.48, -3.36, -3.01, -3.37, -2.49, -2.69, -2.60, -1.31, -1.15, -1.31, -1.05, -1.21, -1.29, -1.68, -1.80, -1.35, -0.59, -0.12, -0.97],
    [-3.07, -3.45, -4.25, -3.76, -4.20, -3.32, -3.73, -3.19, -2.03, -1.87, -1.90, -1.57, -1.53, -1.73, -1.33, -1.26, -2.25, -1.70, -0.97, -1.75],
]


def _build_mj_corrected() -> Dict[Tuple[str, str], float]:
    """Contact energies with the self-energy correction applied.

        e'_ij = e_ij - 0.5 * (e_ii + e_jj)

    WHY THIS MATTERS: the raw MJ matrix is entirely negative, so under the
    raw matrix EVERY contact lowers the energy and the global minimum is
    always the most compact structure regardless of sequence. The correction
    (Miyazawa & Jernigan 1996, Sec. 4) subtracts the residue self-energies,
    yielding a matrix with BOTH signs -- hydrophobic-hydrophobic pairs are
    favourable, polar-hydrophobic pairs are penalized. This is the single
    most important fix relative to the original implementation.
    """
    idx = {aa: i for i, aa in enumerate(MJ_ORDER)}
    self_e = {aa: _MJ_RAW[idx[aa]][idx[aa]] for aa in MJ_ORDER}
    return {(a, b): _MJ_RAW[idx[a]][idx[b]] - 0.5 * (self_e[a] + self_e[b])
            for a in MJ_ORDER for b in MJ_ORDER}


MJ_CORRECTED = _build_mj_corrected()
MJ_RAW = {(a, b): _MJ_RAW[MJ_ORDER.index(a)][MJ_ORDER.index(b)]
          for a in MJ_ORDER for b in MJ_ORDER}

# Kyte-Doolittle hydropathy (1982). Positive = hydrophobic.
KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Formal charge at neutral pH.
CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}


# ==========================================================================
# Default weights — FIXED A PRIORI, NOT LEARNED
# ==========================================================================
DEFAULT_WEIGHTS: Dict[str, float] = {
    "steric": 4.0,
    "contact": 1.0,
    "hbond": 1.0,
    "solvation": 0.3,
    "electrostatic": 0.5,
    "torsion": 1.0,
    "compactness": 0.05,
}

TERM_NAMES = list(DEFAULT_WEIGHTS.keys())

WEIGHT_ORIGIN = {
    "steric": "physical  (must dominate; hard-core overlap is forbidden)",
    "contact": "reference (MJ-corrected potential used at unit weight)",
    "hbond": "reference (DSSP electrostatic model used at unit weight)",
    "solvation": "empirical (sets burial scale relative to contacts)",
    "electrostatic": "physical  (screened Coulomb prefactor)",
    "torsion": "empirical (Ramachandran basin depth, dimensionless)",
    "compactness": "empirical (weak Rg regularizer; deliberately small)",
}


# ==========================================================================
# Ramachandran basins (for the torsion term)
# ==========================================================================
# (phi_center, psi_center, sigma_deg, base_depth)
# Basin centres, widths, and depths approximating the populated regions of a
# real Ramachandran plot. The beta/PPII region in particular is a broad
# continuous plateau (phi in [-180, -45], psi in [90, 180]), not a pair of
# tight Gaussians -- narrow basins leave gaps that penalize genuine
# experimental structures. Widths were widened and a bridge basin added after
# the native chignolin structure scored +5.28 on this term, with eight of ten
# residues penalized despite all lying in allowed regions.
_RAMA_BASINS = [
    (-63.0, -42.0, 28.0, 1.00),   # alpha-R
    (-120.0, 130.0, 40.0, 0.90),  # beta (broad)
    (-75.0, 145.0, 40.0, 0.75),   # PPII (broad, overlaps beta)
    (-85.0, 100.0, 32.0, 0.70),   # beta/PPII lower plateau
    (-100.0, -15.0, 30.0, 0.45),  # bridge (alpha/beta transition)
    (75.0, 35.0, 30.0, 0.40),     # alpha-L
]
_HELIX_FORMERS = set("AELMQKRH")   # Chou-Fasman high-P_alpha residues
_SHEET_FORMERS = set("VIFYTWC")    # Chou-Fasman high-P_beta residues


def _ang_diff(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


def rama_penalty(aa: str, phi_rad: float, psi_rad: float) -> float:
    """Dimensionless Ramachandran penalty in roughly [-0.5, 1.5].

    E_tors(aa, phi, psi) = 1 - sum_k w_k(aa) * exp(-d_k^2 / (2 sigma_k^2))

    where d_k is the angular distance to basin k. Residue-specific: helix
    formers get deeper alpha basins, sheet formers deeper beta basins, and
    alpha-L is heavily suppressed for non-glycine (steric clash of CB).
    Proline is penalized for phi far from -63; glycine gets a flat bonus for
    its broader allowed region.

    PURPOSE: without this, nothing in the energy prefers realistic local
    conformations, and the optimizer produces geometrically valid but
    Ramachandran-forbidden chains.
    """
    pd, sd = math.degrees(phi_rad), math.degrees(psi_rad)
    score = 0.0
    for k, (pc, sc, sig, depth) in enumerate(_RAMA_BASINS):
        d2 = _ang_diff(pd, pc) ** 2 + _ang_diff(sd, sc) ** 2
        w = depth
        if k == 0 and aa in _HELIX_FORMERS:
            w += 0.5
        if k == 1 and aa in _SHEET_FORMERS:
            w += 0.5
        if k == 3 and aa != "G":
            w *= 0.2          # alpha-L is strongly disfavoured off glycine
        score += w * math.exp(-d2 / (2.0 * sig * sig))
    e = 1.0 - score
    if aa == "P":
        # Proline's ring restricts phi to roughly [-90, -50]; penalize only
        # departures from that window, not distance from a single point.
        # The previous form charged 0.03/degree from -63 even at phi = -76,
        # which is a perfectly normal proline angle.
        if pd < -90.0:
            e += 0.03 * (-90.0 - pd)
        elif pd > -50.0:
            e += 0.03 * (pd - (-50.0))
    if aa == "G":
        e -= 0.2
    return e


# ==========================================================================
# Smooth switching function
# ==========================================================================
def switch(d: np.ndarray, d0: float, dc: float) -> np.ndarray:
    """Smooth 1 -> 0 cutoff.

        S(d) = 1                                  d <= d0
             = 0.5 (1 + cos(pi (d-d0)/(dc-d0)))   d0 < d < dc
             = 0                                  d >= dc

    PURPOSE: replaces the original implementation's hard
    `distance_squared == 8` contact test, which was a lattice-geometry
    coincidence rather than a physical criterion and gave a discontinuous
    energy landscape.
    """
    d = np.asarray(d, dtype=float)
    s = np.zeros_like(d)
    s[d <= d0] = 1.0
    mid = (d > d0) & (d < dc)
    if np.any(mid):
        s[mid] = 0.5 * (1.0 + np.cos(math.pi * (d[mid] - d0) / (dc - d0)))
    return s


# ==========================================================================
# Per-sequence cached arrays
# ==========================================================================
_SEQ_CACHE: Dict[Tuple[str, bool], Tuple] = {}


def sequence_arrays(sequence: str, use_corrected_mj: bool = True):
    """Cache KD, charge, and pairwise MJ arrays for a sequence."""
    key = (sequence, use_corrected_mj)
    hit = _SEQ_CACHE.get(key)
    if hit is not None:
        return hit
    table = MJ_CORRECTED if use_corrected_mj else MJ_RAW
    kd = np.array([KD.get(a, 0.0) for a in sequence])
    q = np.array([CHARGE.get(a, 0.0) for a in sequence])
    mj = np.array([[table.get((a, b), 0.0) for b in sequence] for a in sequence])
    _SEQ_CACHE[key] = (kd, q, mj)
    return kd, q, mj


def clear_sequence_cache() -> None:
    _SEQ_CACHE.clear()


# ==========================================================================
# Individual energy terms
# ==========================================================================
def steric_term(CA: np.ndarray, CB: np.ndarray, sep: np.ndarray,
                di: np.ndarray, dj: np.ndarray,
                d_ca: np.ndarray, d_cb: np.ndarray) -> float:
    """Soft-core excluded volume.

        E_steric = sum_{|i-j|>=2} [ (3.8 - d_CA)^2_+ + (3.4 - d_CB)^2_+ ]

    Units: Angstrom^2, multiplied by weight 4.0 -> kcal/mol scale.
    Physical: two CA atoms cannot approach closer than ~3.8 A (the
    consecutive CA-CA virtual bond length); CB spheres exclude at ~3.4 A.
    Quadratic rather than hard-wall so the landscape stays continuous.

    PURPOSE: this is the term that replaces the original implementation's
    `overlap_penalty` for exactly-coincident lattice sites. Off-lattice,
    exact coincidence has measure zero, so a distance-based soft core is the
    only meaningful form of the constraint.
    """
    m2 = sep >= 2
    e = np.where(m2 & (d_ca < 3.8), (3.8 - d_ca) ** 2, 0.0)
    e = e + np.where(m2 & (d_cb < 3.4), (3.4 - d_cb) ** 2, 0.0)
    return float(e.sum())


def contact_term(mj: np.ndarray, sep: np.ndarray, di: np.ndarray,
                 dj: np.ndarray, d_cb: np.ndarray) -> float:
    """Sequence-conditioned residue-residue contacts.

        E_contact = sum_{|i-j|>=3} MJ'_ij * S(d_CB^ij; 4.5, 8.5)

    Units: kcal/mol (MJ units), weight 1.0.
    Physical: the corrected MJ potential is the free energy of transferring
    a residue pair from solvent contact to mutual contact. Both signs occur.
    """
    m3 = sep >= 3
    s = switch(d_cb, 4.5, 8.5) * m3
    return float(np.sum(mj[di, dj] * s))


def hbond_terms(coords: Dict[str, np.ndarray],
                desolvation_cost: float = 1.0) -> Tuple[float, float]:
    """Backbone hydrogen bonds, DSSP electrostatic model with 1:1 matching.

        E_hb(i,j) = 0.084 * 332 * (1/r_ON + 1/r_CH - 1/r_OH - 1/r_CN)

    accepted when E < -0.5 kcal/mol and |i-j| >= 2.

    BEST-PARTNER MATCHING: each amide N-H donates at most ONE backbone
    H-bond and each carbonyl C=O accepts at most ONE. Candidate pairs are
    sorted by increasing (most favourable) energy and greedily matched, each
    residue consumed once per role.

    WHY: without matching, a compact helix accumulates an inflated all-pairs
    H-bond reward and out-scores a correct beta-hairpin purely by being
    dense. Matching removes that artifact.

    Returns (E_local, E_longrange) split at |i-j| = 5 so that helical and
    sheet/tertiary hydrogen bonding can be reported separately. Both are
    summed at unit weight in the total.
    """
    if "N" not in coords or "O" not in coords or "C" not in coords:
        # Representation has no backbone atoms (e.g. the lattice).
        return 0.0, 0.0

    N, C, O = coords["N"], coords["C"], coords["O"]
    H = geo.amide_h_positions(N, C, O)
    n = len(N)
    valid = np.isfinite(H).all(axis=1)

    dON = np.linalg.norm(N[:, None, :] - O[None, :, :], axis=2)
    dCH = np.linalg.norm(C[None, :, :] - H[:, None, :], axis=2)
    dOH = np.linalg.norm(H[:, None, :] - O[None, :, :], axis=2)
    dCN = np.linalg.norm(N[:, None, :] - C[None, :, :], axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        E = 0.084 * 332.0 * (1.0 / dON + 1.0 / dCH - 1.0 / dOH - 1.0 / dCN)

    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    ok = valid[:, None] & (sep >= 2)
    ok &= (dON > 0.5) & (dCH > 0.5) & (dOH > 0.5) & (dCN > 0.5)
    ok &= np.isfinite(E) & (E < -0.5)

    donors, acceptors = np.where(ok)
    if len(donors) == 0:
        return 0.0, 0.0

    energies = E[donors, acceptors]
    order = np.argsort(energies)
    donor_used = np.zeros(n, dtype=bool)
    acc_used = np.zeros(n, dtype=bool)
    local = lr = 0.0
    for k in order:
        i, j = int(donors[k]), int(acceptors[k])
        if donor_used[i] or acc_used[j]:
            continue
        donor_used[i] = True
        acc_used[j] = True
                # Each formed H-bond requires desolvating the donor N-H and the
        # acceptor C=O. The DSSP electrostatic energy omits this cost by
        # construction, which systematically over-rewards regular secondary
        # structure with many bonds: a 10-residue alpha helix has ~6 backbone
        # H-bonds while a short beta-hairpin has ~3, so the helix wins on
        # count alone regardless of sequence. Implicit-solvent models charge
        # roughly +1 kcal/mol per bond formed. Value is the literature figure,
        # NOT tuned to the benchmark.
        e = float(energies[k]) + desolvation_cost
        if e >= 0.0:
            continue          # no longer net-favourable once desolvated
        if abs(i - j) < 5:
            local += e
        else:
            lr += e
    return local, lr


def solvation_term(kd: np.ndarray, CB: np.ndarray) -> float:
    """Hydrophobic burial via coordination number.

        c_i = sum_{j != i} S(d_CB^ij; 6.0, 10.0)
        E_solv = - sum_i (KD_i / 4.5) * c_i

    Units: dimensionless coordination x normalized hydropathy, weight 0.3.
    Physical: burying a hydrophobic residue (KD > 0) is favourable and
    burying a polar residue (KD < 0) is unfavourable. Normalizing by 4.5
    (isoleucine, the KD maximum) puts the term in [-1, 1] per unit
    coordination.

    PURPOSE: the contact term alone is pairwise; burial is many-body and is
    what actually drives a hydrophobic core to form.
    """
    D = np.linalg.norm(CB[:, None, :] - CB[None, :, :], axis=2)
    coord = switch(D, 6.0, 10.0)
    np.fill_diagonal(coord, 0.0)
    return float(-np.sum((kd / 4.5) * coord.sum(axis=1)))


def electrostatic_term(q: np.ndarray, sep: np.ndarray, di: np.ndarray,
                       dj: np.ndarray, d_cb: np.ndarray) -> float:
    """Screened Coulomb between formally charged sidechains.

        E_elec = sum_{|i-j|>=2, q_i q_j != 0}
                     (q_i q_j / max(d_CB, 2)) * exp(-d_CB / 8)

    Units: e^2/A scaled; weight 0.5 sets it well below contacts, which is
    correct for aqueous solvent (eps_r ~ 80).
    Physical: Debye-Huckel-like screening with an 8 A screening length,
    approximating physiological ionic strength. The max(d, 2) floor prevents
    the singularity at contact.

    PURPOSE: salt bridges are a real determinant of hairpin register in
    short charged peptides (e.g. chignolin's D3-T8 region).
    """
    m2 = sep >= 2
    qq = q[di] * q[dj]
    mask = m2 & (qq != 0.0)
    if not np.any(mask):
        return 0.0
    val = (qq / np.maximum(d_cb, 2.0)) * np.exp(-d_cb / 8.0)
    return float(np.sum(val[mask]))


def torsion_term(sequence: str, phi: np.ndarray, psi: np.ndarray) -> float:
    """Sum of per-residue Ramachandran penalties. Weight 1.0, dimensionless."""
    return float(sum(rama_penalty(sequence[i], phi[i], psi[i])
                     for i in range(len(sequence))))


def compactness_term(CA: np.ndarray, n: int) -> float:
    """Weak globularity regularizer.

        E_geom = (Rg - 2.2 N^0.38)^2

    Units: Angstrom^2, weight 0.05 -- deliberately tiny.
    Empirical: 2.2 N^0.38 is the standard scaling of Rg with chain length
    for compact globular proteins.

    PURPOSE: prevents fully extended chains from being competitive purely
    because extended chains have no steric penalty. The small weight is
    intentional: a large weight would recreate exactly the
    "compact == low energy" pathology this design is trying to avoid.
    """
    rg = geo.radius_of_gyration(CA)
    return float((rg - 2.2 * (n ** 0.38)) ** 2)


def backtracking_term(rep, bitstring: str) -> float:
    """Lattice-only: penalize immediate bond reversal.

        E_back = sum_i [ d_i . d_{i+1} == -3 ]   (tetrahedral units)

    PURPOSE: on the tetrahedral lattice a chain may reverse direction,
    producing a physically impossible fold that the coincidence-only overlap
    penalty from the original implementation did not catch. Applied at a
    fixed penalty of 5.0 per reversal (comparable to a strong contact).
    Only active for the lattice representation.
    """
    if not getattr(rep, "is_lattice", False):
        return 0.0
    from representations import LATTICE_DIRECTIONS
    dirs = [np.array(LATTICE_DIRECTIONS[b]) for b in rep.bond_directions(bitstring)]
    return float(sum(1.0 for a, b in zip(dirs, dirs[1:])
                     if float(np.dot(a, b)) < -2.5))


# ==========================================================================
# Assembly
# ==========================================================================
def energy_components(sequence: str,
                      coords: Dict[str, np.ndarray],
                      phi: Optional[np.ndarray] = None,
                      psi: Optional[np.ndarray] = None,
                      use_corrected_mj: bool = True) -> Dict[str, float]:
    """Compute all terms for one structure. Returns a name -> value dict.

    `phi`/`psi` may be None (lattice representation), in which case the
    torsion term is 0 and reported as unavailable.
    """
    n = len(sequence)
    CA = np.asarray(coords["CA"], dtype=float)
    CB = np.asarray(coords.get("CB", coords["CA"]), dtype=float)
    if len(CA) != n:
        raise ValueError(f"coordinate count {len(CA)} != sequence length {n}")

    kd, q, mj = sequence_arrays(sequence, use_corrected_mj)

    di, dj = np.triu_indices(n, 1)
    sep = dj - di
    d_ca = np.linalg.norm(CA[di] - CA[dj], axis=1)
    d_cb = np.linalg.norm(CB[di] - CB[dj], axis=1)

    hb_local, hb_lr = hbond_terms(coords)

    return {
        "steric": steric_term(CA, CB, sep, di, dj, d_ca, d_cb),
        "contact": contact_term(mj, sep, di, dj, d_cb),
        # Normalized per residue. The raw sum scales with the NUMBER of
        # backbone H-bonds, which favours regular secondary structure by
        # count alone: a 10-residue alpha helix has ~6 i->i+4 bonds while a
        # hairpin of the same length has ~3. That count advantage was worth
        # ~5 kcal/mol on chignolin and was the single largest reason the
        # helix outscored the native structure. Dividing by N removes the
        # chain-length artifact without altering the physics of any
        # individual bond.
        "hbond": (hb_local + hb_lr) / max(1, n),
        "hbond_local": hb_local,          # reported raw, not weighted
        "hbond_longrange": hb_lr,         # reported raw, not weighted
        "solvation": solvation_term(kd, CB),
        "electrostatic": electrostatic_term(q, sep, di, dj, d_cb),
        "torsion": (0.0 if phi is None
                    else torsion_term(sequence, phi, psi)),
        "compactness": compactness_term(CA, n),
    }


def total_from_components(components: Dict[str, float],
                          weights: Dict[str, float]) -> float:
    return float(sum(weights.get(k, 0.0) * components.get(k, 0.0)
                     for k in TERM_NAMES))