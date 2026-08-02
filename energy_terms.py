"""Legacy (knowledge-based) energy model.

Rewritten 2026-07-28 to remove the defects that made this objective *anti-correlated*
with correctness: `docs/energy-model-diagnosis.md` measured Spearman(energy, RMSD) =
-0.351 by exhaustive enumeration, meaning a search that optimised it did worse than
drawing structures at random. See `docs/rmsd-accuracy-fixes.md` for the full account.

Every change here is a correction that is stated in terms of a residue *class* or a
literature scale covering all twenty residues. Nothing is tuned to a particular
sequence, and the two genuinely free weights (`torsion`, `compactness`, `aromatic`) are
meant to be set by `energy_quality.calibrate_weights` over a *set* of training
sequences, never by hand against one target. `DEFAULT_WEIGHTS` below are the
variance-balanced starting point, not a fit.
"""
import math
from functools import lru_cache
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
    idx = {aa: i for i, aa in enumerate(MJ_ORDER)}
    self_e = {aa: _MJ_RAW[idx[aa]][idx[aa]] for aa in MJ_ORDER}
    return {(a, b): _MJ_RAW[idx[a]][idx[b]] - 0.5 * (self_e[a] + self_e[b])
            for a in MJ_ORDER for b in MJ_ORDER}


MJ_CORRECTED = _build_mj_corrected()
MJ_RAW = {(a, b): _MJ_RAW[MJ_ORDER.index(a)][MJ_ORDER.index(b)]
          for a in MJ_ORDER for b in MJ_ORDER}


# ==========================================================================
# Burial scale
# ==========================================================================
#: Fauchere-Pliska octanol/water pi (1983), all twenty residues.
#:
#: This replaces Kyte-Doolittle, and the replacement is the single largest accuracy fix
#: in this rewrite. KD is a membrane-spanning propensity scale and places the aromatics
#: on the *hydrophilic* side (W -0.9, Y -1.3, F +2.8 only by comparison). Any peptide
#: whose hydrophobic core is an aromatic cluster is then penalised for forming it, and
#: for a peptide with no aliphatic core every KD value can be negative -- in which case
#: `solvation_term` becomes purely expansive and actively drives the search to an
#: extended chain. Fauchere-Pliska puts W and Y at +2.25 and +0.96, so burying an
#: aromatic cluster is rewarded, which is what the physics says.
#:
#: This is a property of the scale, not of any one sequence: it changes the sign of the
#: solvation term for every aromatic-core peptide in `dataset.CANDIDATE_PDB_IDS`.
BURIAL: Dict[str, float] = {
    "W": 2.25, "I": 1.80, "F": 1.79, "L": 1.70, "C": 1.54, "M": 1.23, "V": 1.22,
    "Y": 0.96, "P": 0.72, "A": 0.31, "T": 0.26, "H": 0.13, "G": 0.00, "S": -0.04,
    "Q": -0.22, "N": -0.60, "E": -0.64, "D": -0.77, "K": -0.99, "R": -1.01,
}
#: Trp, so the normalised scale spans [-0.45, 1.0] and is dimensionless.
BURIAL_NORM = 2.25

CHARGE = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5}

#: Aromatic residues, as a class. Ring-ring stacking is real physics that the MJ
#: contact term represents only weakly (its weighted spread is 0.29 against torsion's
#: 2.27), and it is the dominant tertiary interaction in short hairpin peptides
#: generally -- not in any one of them specifically.
AROMATIC = frozenset("FYWH")

#: Coulomb constant, kcal A / (mol e^2). The previous electrostatic term omitted this
#: entirely, so a salt bridge at 4 A scored -0.076 kcal/mol against a physical -1 to -3
#: and the term had the smallest spread of the seven (0.011). It was inert.
COULOMB = 332.0637
#: Effective dielectric for a solvent-exposed peptide. Combined with the existing
#: exp(-d/8) screening this is a coarse Debye-Huckel treatment, so the term's weight is
#: still a calibration target -- but its *magnitude* is now physical.
DIELECTRIC = 40.0

#: van der Waals radii, A. Used by the all-atom backbone clash test.
VDW_RADIUS = {"N": 1.55, "C": 1.70, "O": 1.52, "S": 1.80}
#: Soft-sphere prefactor. Contacts are penalised below SOFTNESS * (r_i + r_j); 0.80
#: leaves room for real close contacts (a C-C pair may legitimately reach 3.4 A) while
#: still catching interpenetration.
SOFTNESS = 0.80

#: |i-j| at or above which an H-bond counts as long-range. A definition, not a weight.
HB_LONGRANGE_SEP = 5

#: Re-exported from `protein_geometry` so the two copies of the DSSP form cannot drift.
#: Below HB_MIN_ON the donor and acceptor heavy atoms are interpenetrating, and nothing else
#: in the model says so: `_steric_layout` exempts every hetero N/O pair, and the amide H is
#: not in the steric layout at all, so the `-1/dOH` term is unbounded below. As dOH -> 0.5
#: (the old numerical guard) that term alone reaches -55.8 kcal/mol against a maximum
#: possible weighted steric of 27.0, so the clash can never win at any weight. Measured: the
#: SA arm at default budget drove into it on 2 of 3 seeds, scoring -58 against an ideal
#: helix's -28.
HB_MIN_ON = geo.HB_MIN_ON
HB_E_FLOOR = geo.HB_E_FLOOR

#: Aromatic ring-ring geometry. Two wells rather than one, because pi-stacking has two
#: distinct favourable arrangements and a single distance well cannot tell them apart:
#:
#:   parallel-displaced  ring planes roughly parallel, centroids ~3.6-4.2 A apart
#:   T-shaped (edge-to-face)  planes roughly perpendicular, centroids ~5.0-5.5 A
#:
#: `theta` is the angle between ring normals folded into [0, 90] degrees, since a normal's
#: sign is arbitrary. The relative depths are a modelling choice under the term's single
#: calibrated weight; the geometry is not.
COOP_LADDER_MIN_SEP =2
AROM_PD_DIST = 4.00
AROM_PD_DEPTH = 1.00
AROM_T_DIST = 5.20
AROM_T_DEPTH = 0.70
AROM_DIST_WIDTH = 1.10
AROM_ANGLE_WIDTH = 35.0
#: Fallback CB-CB well, used for aromatics whose ring this repo cannot build (His) or when
#: chi1 is not encoded so no ring geometry exists.
AROM_CB_DIST = 5.50
AROM_CB_WIDTH = 1.80


# ==========================================================================
# Weights
# ==========================================================================
#: Starting weights. `physical` entries are fixed by the units of the term they scale;
#: `empirical` entries are free and should be set by
#: `energy_quality.calibrate_weights` over a train split of sequences.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "steric": 4.0,
    "contact": 1.0,
    "hbond_local": 1.0,
    "hbond_longrange": 3.0,
    "coop_helix": 2.0,
    "coop_sheet": 2.0,
    "solvation": 0.5,
    "electrostatic": 1.0,
    "aromatic": 0.8,
    "torsion": 0.15,
    "compactness": 0.4,
}

TERM_NAMES = list(DEFAULT_WEIGHTS.keys())

#: Which weights `calibrate_weights` is allowed to move. The rest are pinned by the units
#: of the term they scale.
#:
#: `hbond_longrange` is here rather than fixed at 3.0 deliberately. Weighting the
#: long-range ladder above local bonds is right for a beta hairpin and gives a helix
#: nothing -- every helical H-bond is i->i+4 -- so a hardcoded multiplier is a standing bet
#: that the targets are hairpins. Most of `dataset.CANDIDATE_PDB_IDS` are, but 1DU1 is a
#: charged helix and the trp-cages are mixed, so the multiplier has to be measured on a
#: train split and reported on held-out clusters like any other free parameter. Splitting
#: `hbond` into two terms also means `experiment_energy_ablation` produces separate
#: `no_hbond_local` and `no_hbond_longrange` arms, which is how the hairpin bias becomes
#: visible instead of assumed.
FREE_WEIGHTS = ("aromatic", "torsion", "compactness", "solvation",
                "hbond_longrange", "coop_helix", "coop_sheet")

#: Terms that couple residues: they cannot be evaluated without a three-dimensional
#: arrangement, so they are the only ones carrying tertiary-structure information.
COUPLED_TERMS = ("contact", "hbond_local", "hbond_longrange", "coop_helix",
                 "coop_sheet", "solvation", "electrostatic", "aromatic",
                 "compactness")

#: Terms that are a sum of independent per-residue contributions. Their minimum is found
#: residue by residue, so a landscape they dominate has no fold in it -- which is exactly
#: how the pre-2026-07-28 model ended up anti-correlated with correctness. The invariant
#: worth defending is COUPLED variance > SEPARABLE variance among feasible structures;
#: `energy_quality` measures it and `validation` gates on it.
SEPARABLE_TERMS = ("torsion",)

#: `steric` is in neither list: it is a feasibility filter, not a discriminator. It
#: carried 87% of the variance across *all* random structures while being exactly zero for
#: both the native fold and the global minimum, so among the clash-free structures a
#: search actually explores it is flat. Any variance accounting that does not exclude it
#: balances against a term that does not vary where it matters.
FILTER_TERMS = ("steric",)

WEIGHT_ORIGIN = {
    "steric": "physical  (must dominate; hard-core overlap is forbidden)",
    "contact": "reference (MJ-corrected potential used at unit weight)",
    "hbond_local": "reference (DSSP electrostatic model, kcal/mol, at unit weight)",
    "hbond_longrange": "empirical (hairpin-ladder emphasis; MUST be calibrated -- a fixed "
                       "multiplier is a bet that the targets are hairpins)",
    "coop_helix": "empirical (consecutive n-turns; MUST be calibrated)",
    "coop_sheet": "empirical (consecutive ladder rungs; MUST be calibrated)",
    "solvation": "empirical (burial scale is dimensionless; sets burial vs contact)",
    "electrostatic": "physical  (screened Coulomb, with the 332 prefactor)",
    "aromatic": "empirical (ring-ring well depth, kcal/mol-ish)",
    "torsion": "empirical (Ramachandran basin depth; small because the state library "
               "is already restricted to favourable basins -- see docstring)",
    "compactness": "empirical (one-sided Rg restraint)",
}


# ==========================================================================
# Ramachandran
# ==========================================================================
_RAMA_BASINS = [
    (-63.0, -42.0, 28.0, 1.00),
    (-120.0, 130.0, 40.0, 0.90),
    (-75.0, 145.0, 40.0, 0.75),
    (-85.0, 100.0, 32.0, 0.70),
    (-100.0, -15.0, 30.0, 0.45),
    (75.0, 35.0, 30.0, 0.40),
]
_HELIX_FORMERS = set("AELMQKRH")
_SHEET_FORMERS = set("VIFYTWC")


def _ang_diff(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


@lru_cache(maxsize=100_000)
def rama_penalty(aa: str, phi_rad: float, psi_rad: float) -> float:
    """Ramachandran basin penalty for one residue.

    Memoized: the torsion representations draw phi/psi from a fixed library of discrete
    values, so across a whole search this has only n_residues x n_states distinct
    arguments but is called once per residue per structure. Pure function of its
    arguments, so the cache returns bit-identical values.

    NOTE on the weight this term carries. `representations` restricts phi/psi to a
    curated library of *favourable* basins, so a term whose job is to keep torsions
    physical has little left to do -- while being a sum of independent per-residue
    penalties, so its minimum is reached by choosing each residue's best basin in
    isolation. That minimum is locally ideal everywhere and globally meaningless, and at
    weight 1.0 it accounted for +6.329 of native chignolin's +7.963 penalty relative to
    the global minimum. Hence `DEFAULT_WEIGHTS["torsion"] = 0.15`: it survives as a
    tiebreaker carrying amino-acid specificity, and the *library* (per-residue-class in
    `representations`) now carries the rest.
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
            w *= 0.2
        score += w * math.exp(-d2 / (2.0 * sig * sig))
    e = 1.0 - score
    if aa == "P":
        if pd < -90.0:
            e += 0.03 * (-90.0 - pd)
        elif pd > -50.0:
            e += 0.03 * (pd - (-50.0))
    if aa == "G":
        e -= 0.2
    return e


def switch(d: np.ndarray, d0: float, dc: float) -> np.ndarray:
    """Cosine switch: 1 below d0, 0 above dc, smooth in between."""
    d = np.asarray(d, dtype=float)
    s = np.zeros_like(d)
    s[d <= d0] = 1.0
    mid = (d > d0) & (d < dc)
    if np.any(mid):
        s[mid] = 0.5 * (1.0 + np.cos(math.pi * (d[mid] - d0) / (dc - d0)))
    return s


# ==========================================================================
# Per-sequence caches
# ==========================================================================
_SEQ_CACHE: Dict[Tuple[str, bool], Tuple] = {}


def sequence_arrays(sequence: str, use_corrected_mj: bool = True):
    """Cache burial, charge, and pairwise MJ arrays for a sequence.

    Returns ``(burial, charge, mj)``. The first element was Kyte-Doolittle hydropathy
    before this rewrite; it is now the Fauchere-Pliska scale (see `BURIAL`).
    """
    key = (sequence, use_corrected_mj)
    hit = _SEQ_CACHE.get(key)
    if hit is not None:
        return hit
    table = MJ_CORRECTED if use_corrected_mj else MJ_RAW
    burial = np.array([BURIAL.get(a, 0.0) for a in sequence])
    q = np.array([CHARGE.get(a, 0.0) for a in sequence])
    mj = np.array([[table.get((a, b), 0.0) for b in sequence] for a in sequence])
    _SEQ_CACHE[key] = (burial, q, mj)
    return burial, q, mj


@lru_cache(maxsize=4096)
def aromatic_indices(sequence: str) -> Tuple[int, ...]:
    return tuple(i for i, a in enumerate(sequence) if a in AROMATIC)


@lru_cache(maxsize=64)
def pair_index(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cached upper-triangle pair indices and separations for `n` residues.

    ``np.triu_indices`` was being rebuilt on every energy evaluation. The arrays are
    returned from a cache, so treat them as read-only -- nothing here mutates them.
    """
    di, dj = np.triu_indices(n, 1)
    return di, dj, (dj - di)


def clear_sequence_cache() -> None:
    _SEQ_CACHE.clear()
    _STERIC_LAYOUT_CACHE.clear()
    rama_penalty.cache_clear()
    aromatic_indices.cache_clear()
    pair_index.cache_clear()


# ==========================================================================
# Terms
# ==========================================================================
def steric_term(coords: Dict[str, np.ndarray], sequence: str,
                rings: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
                min_sep: int = 2) -> float:
    """Soft-sphere overlap over every available heavy backbone atom, plus CB and rings.

    Replaces the previous CA-CA (< 3.8 A) and CB-CB (< 3.4 A) pair test, which had two
    problems. It never looked at N, C or O, so backbone carbonyls could interpenetrate
    for free -- the main geometric constraint that should force an aromatic pair into a
    stacked rather than an overlapping arrangement. And its 3.8 A CA threshold is below
    the geometric floor for |i-j| = 2 (2 * 3.8 * sin(40 deg) ~ 4.9 A), so in the torsion
    representation the sep-2 test could never fire at all.

    N/O pairs are exempt: those are exactly the H-bond donor/acceptor pairs, which
    legitimately approach 2.8-3.0 A, and penalising them would put this term in direct
    conflict with `hbond_terms`.

    `rings`, when supplied, adds aromatic ring atoms. This is what stops two rings simply
    passing through each other -- previously nothing did, because `sidechains` built the
    rings and the energy never looked at them. Note the soft-sphere limit for a carbon pair
    is 0.80 * (1.70 + 1.70) = 2.72 A while parallel-displaced stacking puts the closest
    ring atoms at ~3.5 A, so this forbids interpenetration without forbidding the stack.

    The pair list, element radii and exemption mask depend only on the *layout* -- which
    atom names are present, the residue count, and which ring atoms exist -- never on the
    coordinates. They are cached by `_steric_layout`, so a structure costs one gather, one
    norm and one dot product. Rebuilding them per call made this the most expensive term in
    the model once ring atoms took the pair count from 45 to ~2000.
    """
    names = tuple(k for k in ("N", "CA", "C", "O", "CB") if k in coords)
    if not names:
        return 0.0
    n_res = len(coords[names[0]])
    if n_res != len(sequence):
        raise ValueError(f"coordinate count {n_res} != sequence length {len(sequence)}")

    # Insertion order, which `sidechains.ring_atom_names` fixes to template order, so the
    # gather below lines up with the cached layout. Deterministic, and avoids a per-call
    # sort of the atom names.
    ring_keys = tuple((i, nm) for i, atoms_i in (rings or {}).items() for nm in atoms_i)
    ii, jj, limit = _steric_layout(names, n_res, ring_keys, min_sep)
    if ii.size == 0:
        return 0.0

    blocks = [np.asarray(coords[k], dtype=float) for k in names]
    if ring_keys:
        blocks.append(np.array([rings[i][nm] for i, nm in ring_keys], dtype=float))
    atoms = np.concatenate(blocks, axis=0)

    over = np.maximum(0.0, limit - np.linalg.norm(atoms[ii] - atoms[jj], axis=1))
    return float(over @ over)


_STERIC_LAYOUT_CACHE: Dict[tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _steric_layout(names: Tuple[str, ...], n_res: int,
                   ring_keys: Tuple[Tuple[int, str], ...], min_sep: int):
    """Cached ``(ii, jj, contact_limit)`` for one atom layout. Read-only."""
    key = (names, n_res, ring_keys, min_sep)
    hit = _STERIC_LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit

    elem: List[str] = []
    res_of: List[int] = []
    for k in names:
        # CB is carbon; Gly has no real CB but carries a virtual one, which is the right
        # excluded volume for a CA-only pseudo-atom anyway.
        elem.extend([("N" if k == "N" else "O" if k == "O" else "C")] * n_res)
        res_of.extend(range(n_res))
    for i, nm in ring_keys:
        elem.append(nm[0] if nm[0] in VDW_RADIUS else "C")
        res_of.append(i)

    elem_arr = np.asarray(elem)
    radius = np.array([VDW_RADIUS[e] for e in elem])
    is_no = np.isin(elem_arr, ("N", "O"))
    res_arr = np.asarray(res_of)

    ii, jj = np.triu_indices(len(elem), 1)
    keep = np.abs(res_arr[ii] - res_arr[jj]) >= min_sep
    # exempt N...O and O...N (H-bond partners)
    keep &= ~(is_no[ii] & is_no[jj] & (elem_arr[ii] != elem_arr[jj]))
    ii, jj = ii[keep], jj[keep]
    out = (ii, jj, SOFTNESS * (radius[ii] + radius[jj]))
    _STERIC_LAYOUT_CACHE[key] = out
    return out


def contact_term(mj: np.ndarray, sep: np.ndarray, di: np.ndarray,
                 dj: np.ndarray, d_cb: np.ndarray) -> float:
    """MJ-corrected contact energy over CB pairs at |i-j| >= 3."""
    m3 = sep >= 3
    return float(np.dot(mj[di[m3], dj[m3]], switch(d_cb[m3], 4.5, 8.5)))


def hbond_terms(coords: Dict[str, np.ndarray],
                desolvation_cost: float = 1.0
                ) -> Tuple[float, float, Tuple[Tuple[int, int], ...]]:
    """DSSP electrostatic H-bond energies, split into local and long-range sums.

    Returns ``(local, longrange, matched_pairs)`` where each pair is ``(donor, acceptor)``.
    The pair list is what makes the cooperativity terms possible -- see
    `coop_helix_term` / `coop_sheet_term`.

    Vectorised greedy one-donor-one-acceptor matching. `energy_components` reports the two
    sums as the separate weighted terms `hbond_local` and `hbond_longrange`, and -- the
    part that matters -- does *not* divide either by the residue count. `HB_LONGRANGE_SEP`
    is the boundary.
    """
    if not all(k in coords for k in ("N", "C", "O")):
        return 0.0, 0.0, ()      # representation has no backbone (e.g. the lattice)

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
    # Bound the 1/r divergence. Without this the term is unbounded below while `steric_term`
    # is quadratic and exempts N/O pairs outright, so no weight can ever make the clash win.
    E = np.maximum(E, HB_E_FLOOR)

    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    ok = valid[:, None] & (sep >= 2)
    ok &= (dON > HB_MIN_ON) & (dCH > 0.5) & (dOH > 0.5) & (dCN > 0.5)
    ok &= np.isfinite(E) & (E < -0.5)

    donors, acceptors = np.where(ok)
    if len(donors) == 0:
        return 0.0, 0.0, ()

    energies = E[donors, acceptors]
    # Stable: HB_E_FLOOR manufactures exact ties at -4.0 and this greedy match is
    # order-dependent, so an unstable sort could map one bitstring to two energies across
    # platforms -- which is exactly the invariant budget.py's cache relies on.
    order = np.argsort(energies, kind="stable")
    donor_used = np.zeros(n, dtype=bool)
    acc_used = np.zeros(n, dtype=bool)
    local = lr = 0.0
    matched = []
    for k in order:
        i, j = int(donors[k]), int(acceptors[k])
        if donor_used[i] or acc_used[j]:
            continue
        donor_used[i] = True
        acc_used[j] = True
        e = float(energies[k]) + desolvation_cost
        if e >= 0.0:
            continue
        matched.append((i, j))
        if abs(i - j) < HB_LONGRANGE_SEP:
            local += e
        else:
            lr += e
    return local, lr, tuple(matched)


def coop_helix_term(pairs: Tuple[Tuple[int, int], ...]) -> float:
    """Reward *consecutive* n-turns, which is what a helix actually is.

    `hbond_local` is a sum over independent matched pairs, so two scattered i->i+4 bonds
    score exactly the same as two adjacent ones. That degeneracy is the reason an additive
    potential cannot tell a helix from a compact tangle with the same bond count. DSSP
    itself defines a helix by *repeating* n-turns, not by their number.

    In DSSP's convention the donor is C-terminal to the acceptor, so a helical bond has
    ``donor - acceptor`` in {3, 4}; the run condition is that ``(d+1, a+1)`` is also bonded.
    An n-residue run scores n-1, against 0 for the same bonds scattered.
    """
    s = {(d, a) for d, a in pairs if 3 <= d - a <= 4}
    return -float(sum(1 for (d, a) in s if (d + 1, a + 1) in s))


def coop_sheet_term(pairs: Tuple[Tuple[int, int], ...],
                    min_sep: int = 2) -> float:
    """Reward *consecutive* rungs of an antiparallel ladder.

    The same additivity problem as `coop_helix_term`, and the one that most likely bounds
    RMSD on a hairpin: `hbond_longrange` cannot distinguish two scattered long-range bonds
    from two adjacent rungs forming a correct register. In an antiparallel ladder the
    H-bonded pairs step as ``(i, j) -> (i+2, j-2)``, so that offset is the run condition.

    Purely topological -- no amino acid, no distance scale, no parameter. Requiring a
    ladder also constrains the turn at the small-|i-j| end, which is the other half of
    register error -- which is why `min_sep` is 2 and NOT `HB_LONGRANGE_SEP`. That
    threshold classifies bond *energy* into local vs long-range; used here it discards
    every pair with |d - a| < 5, i.e. the rungs nearest the turn, truncating each ladder
    from the inside. A 3-rung ladder then scored -1 instead of -2, and on a 10-mer the
    term had almost nothing left to reward. 2 is the floor `hbond_terms` already enforces
    when matching, so every matched pair is admissible and the (i+2, j-2) step does all
    the discrimination.

    Only antiparallel is modelled: hairpins are antiparallel, and parallel sheets do not
    occur in peptides of this length.
    """
    s = {(d, a) for d, a in pairs if abs(d - a) >= min_sep}
    return -float(sum(1 for (d, a) in s if (d + 2, a - 2) in s))


def solvation_term(burial: np.ndarray, d_cb: np.ndarray,
                   di: np.ndarray, dj: np.ndarray, n: int) -> float:
    """Burial reward, proportional to the Fauchere-Pliska value of each residue.

    `burial` used to be Kyte-Doolittle. See the `BURIAL` docstring for why that made
    aromatic-core peptides unfoldable by construction.

    Coordination numbers are accumulated from the upper-triangle distances
    `energy_components` has already computed, rather than from a fresh (n, n) matrix plus a
    `fill_diagonal`. Identical result -- the switch is symmetric and the triangle excludes
    the diagonal -- at half the distance work and no square allocation.
    """
    s = switch(d_cb, 6.0, 10.0)
    coord = (np.bincount(di, weights=s, minlength=n)
             + np.bincount(dj, weights=s, minlength=n))
    return float(-(burial / BURIAL_NORM) @ coord)


def ring_frame(ring_xyz) -> Tuple[np.ndarray, np.ndarray]:
    """Centroid and unit normal of a planar ring.

    The normal is the least-variance direction of the atom cloud, which for a ring that is
    planar to 1e-6 A (as `sidechains` builds them, pinned by
    `validation.test_sidechain_rings_planar`) is the ring normal to machine precision.

    Uses `eigh` on the 3x3 scatter matrix rather than an SVD of the (k, 3) coordinates:
    same eigenvector, order-independent, and several times cheaper -- which matters because
    this runs once per aromatic residue per energy evaluation. `eigh` returns eigenvalues
    ascending, so column 0 is the smallest-variance direction.
    """
    P = np.asarray(ring_xyz, dtype=float)
    M = P - P.mean(axis=0)
    _, vecs = np.linalg.eigh(M.T @ M)
    return P.mean(axis=0), vecs[:, 0]


def aromatic_term(sequence: str, coords: Dict[str, np.ndarray],
                  rings: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
                  min_sep: int = 3) -> float:
    """Ring-ring attraction between aromatic sidechains (F, Y, W, H).

    Two regimes, and which one applies is a property of the *encoding*:

    * **Real geometry** when `rings` supplies ring atoms for both partners -- available
      once chi1 is encoded (`representations.TorsionStateRepresentation.build_rings`). The
      potential is then a function of centroid distance *and* the angle between ring
      normals, with separate wells for parallel-displaced and T-shaped stacking. This is
      the point of the chi1 bits: three terms in this model (burial, long-range H-bond, and
      this one) all say "bring the aromatics together", but only this one can say *in what
      orientation*, and orientation is what separates a 2 A prediction from a 4 A one.
    * **CB-CB fallback** otherwise -- His, whose imidazole template is not implemented, or
      any representation without chi1 bits. A single soft distance well: crude, but honest
      about the fact that without chi1 the ring's direction is not determined.

    This is the one *new* term rather than a correction, and the one most at risk of
    encoding a preconception about what these peptides look like. It is therefore in
    `TERM_NAMES`, so `experiment_energy_ablation` automatically produces a `no_aromatic`
    arm, and its weight is a `FREE_WEIGHTS` entry to be set on a train split and reported
    on held-out clusters. If it does not survive that, delete it.
    """
    idx = aromatic_indices(sequence)
    if len(idx) < 2:
        return 0.0
    rings = rings or {}
    CB = np.asarray(coords.get("CB", coords["CA"]), dtype=float)

    # dict insertion order is the template order from `sidechains.ring_atom_names`, so this
    # is deterministic without a per-call sort; `ring_frame` is order-independent anyway.
    frames = {i: ring_frame(np.array(list(rings[i].values()), dtype=float))
              for i in idx if i in rings and len(rings[i]) >= 3}

    total = 0.0
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if j - i < min_sep:
                continue
            if i in frames and j in frames:
                ci, ni = frames[i]
                cj, nj = frames[j]
                d = float(np.linalg.norm(ci - cj))
                # Normals are sign-ambiguous, so fold the angle into [0, 90] degrees.
                cos = min(1.0, abs(float(np.dot(ni, nj))))
                theta = math.degrees(math.acos(cos))
                pd = (AROM_PD_DEPTH
                      * math.exp(-((d - AROM_PD_DIST) / AROM_DIST_WIDTH) ** 2)
                      * math.exp(-(theta / AROM_ANGLE_WIDTH) ** 2))
                t = (AROM_T_DEPTH
                     * math.exp(-((d - AROM_T_DIST) / AROM_DIST_WIDTH) ** 2)
                     * math.exp(-((theta - 90.0) / AROM_ANGLE_WIDTH) ** 2))
                total -= pd + t
            else:
                d = float(np.linalg.norm(CB[i] - CB[j]))
                total -= math.exp(-((d - AROM_CB_DIST) / AROM_CB_WIDTH) ** 2)
    return float(total)


def electrostatic_term(q: np.ndarray, sep: np.ndarray, di: np.ndarray,
                       dj: np.ndarray, d_cb: np.ndarray) -> float:
    """Screened Coulomb between formal sidechain charges.

    Now carries the `COULOMB / DIELECTRIC` prefactor the previous version omitted, which
    made it 13-40x too weak and gave it the smallest spread of any term.

    Masked before the exponential rather than after: most peptides have a handful of charged
    pairs out of n(n-1)/2, and the old form evaluated `exp` over every pair to then discard
    almost all of it.
    """
    qq = q[di] * q[dj]
    mask = (sep >= 2) & (qq != 0.0)
    if not np.any(mask):
        return 0.0
    d = d_cb[mask]
    # The 2.0 A floor guards the 1/r singularity only; the screening exponential keeps the
    # true distance, exactly as before. Clamping both would silently change the term for
    # sub-2 A pairs -- which are severe clashes, but changing them is still a change.
    return float(np.sum((COULOMB / DIELECTRIC) * (qq[mask] / np.maximum(d, 2.0))
                        * np.exp(-d / 8.0)))


def torsion_term(sequence: str, phi: np.ndarray, psi: np.ndarray) -> float:
    """Sum of per-residue Ramachandran penalties. Separable by construction -- see the
    weight note in `rama_penalty`."""
    return float(sum(rama_penalty(sequence[i], phi[i], psi[i])
                     for i in range(len(sequence))))


def compactness_term(CA: np.ndarray, n: int) -> float:
    """One-sided radius-of-gyration restraint.

    Penalises expansion past the expected Rg for a folded chain of this length but not
    over-compaction, which is `steric_term`'s job. A two-sided well made the two terms
    fight, and at the old weight of 0.05 an extended chain paid only 0.69 for being ~4 A
    too large while gaining ~6 in torsion -- the one term that knew the answer should be
    compact was the second-weakest in the model.
    """
    rg = geo.radius_of_gyration(CA)
    target = 2.2 * (n ** 0.38)
    return float(max(0.0, rg - target) ** 2)


def backtracking_term(rep, bitstring: str) -> float:
    if not getattr(rep, "is_lattice", False):
        return 0.0
    from representations import LATTICE_DIRECTIONS
    dirs = [np.array(LATTICE_DIRECTIONS[b]) for b in rep.bond_directions(bitstring)]
    return float(sum(1.0 for a, b in zip(dirs, dirs[1:])
                     if float(np.dot(a, b)) < -2.5))


# ==========================================================================
def energy_components(sequence: str,
                     coords: Dict[str, np.ndarray],
                     phi: Optional[np.ndarray] = None,
                     psi: Optional[np.ndarray] = None,
                     use_corrected_mj: bool = True,
                     rings: Optional[Dict[int, Dict[str, np.ndarray]]] = None
                     ) -> Dict[str, float]:
    """Unweighted term values for one structure.

    All coordinates must be in ANGSTROMS. `representations` guarantees this for both
    representations; before this rewrite the tetrahedral lattice returned coordinates in
    lattice units (bond norm sqrt(3) ~ 1.73 where CA-CA should be 3.80), which made
    every threshold here wrong by 2.19x on that representation.

    `rings` maps residue index -> aromatic ring atoms, from
    `representations.TorsionStateRepresentation.build_rings`. Supplying it turns the
    aromatic term into a real orientation-dependent pi-stacking potential and gives ring
    atoms excluded volume; omitting it falls back to CB proxies.

    Note `hbond_local` and `hbond_longrange` are separate weighted terms. They used to be
    combined -- and divided by the chain length -- which suppressed the only long-range
    structural signal by 10x at N=10 and let a helix's local bonds substitute for a
    hairpin's ladder. Keeping them separate also makes the long-range emphasis a calibrated
    parameter rather than a hardcoded bet that every target is a hairpin.

    `coop_helix` and `coop_sheet` then add the one thing an additive pairwise potential
    structurally cannot express: cooperativity. Both are deliberately symmetric -- a helix
    run and a sheet ladder are rewarded by the same mechanism with the same default weight
    -- so the model gains cooperativity without gaining a preference for either topology.
    """
    n = len(sequence)
    CA = np.asarray(coords["CA"], dtype=float)
    CB = np.asarray(coords.get("CB", coords["CA"]), dtype=float)
    if len(CA) != n:
        raise ValueError(f"coordinate count {len(CA)} != sequence length {n}")

    burial, q, mj = sequence_arrays(sequence, use_corrected_mj)

    di, dj, sep = pair_index(n)          # cached; read-only
    d_cb = np.linalg.norm(CB[di] - CB[dj], axis=1)

    hb_local, hb_lr, hb_pairs = hbond_terms(coords)

    return {
        "steric": steric_term(coords, sequence, rings=rings),
        "contact": contact_term(mj, sep, di, dj, d_cb),
        "hbond_local": hb_local,
        "hbond_longrange": hb_lr,
        "coop_helix": coop_helix_term(hb_pairs),
        "coop_sheet": coop_sheet_term(hb_pairs),
        "solvation": solvation_term(burial, d_cb, di, dj, n),
        "electrostatic": electrostatic_term(q, sep, di, dj, d_cb),
        "aromatic": aromatic_term(sequence, coords, rings=rings),
        "torsion": (0.0 if phi is None else torsion_term(sequence, phi, psi)),
        "compactness": compactness_term(CA, n),
    }


def total_from_components(components: Dict[str, float],
                         weights: Dict[str, float]) -> float:
    return float(sum(weights.get(k, 0.0) * components.get(k, 0.0)
                     for k in TERM_NAMES))
