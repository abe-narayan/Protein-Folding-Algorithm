"""Backbone geometry, structural metrics, and PDB I/O.

Pure geometry and structural bookkeeping. No energy, no representation, no
optimization. This module is the single source of truth for:

  * NeRF backbone construction from (phi, psi)
  * Kabsch superposition and RMSD in ANGSTROMS
  * PDB parsing (with an access log for leakage auditing)
  * contact maps and secondary-structure assignment

Every RMSD produced anywhere in this project must come from `ca_rmsd` or
`rmsd` here, on unnormalized Angstrom coordinates. There is deliberately no
coordinate-normalization function in this project.
"""
import math
import os
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from Bio.PDB import PDBParser
    _HAVE_BIOPYTHON = True
except Exception:  # pragma: no cover
    _HAVE_BIOPYTHON = False


# --------------------------------------------------------------------------
# Ideal backbone geometry (Engh & Huber 1991 values, Angstroms / radians)
# --------------------------------------------------------------------------
BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
BOND_C_O = 1.231
ANGLE_N_CA_C = math.radians(111.0)
ANGLE_CA_C_N = math.radians(116.2)
ANGLE_C_N_CA = math.radians(121.7)
ANGLE_CA_C_O = math.radians(120.8)
OMEGA_TRANS = math.pi

DEFAULT_PHI = math.radians(-60.0)
DEFAULT_PSI = math.radians(-45.0)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


# --------------------------------------------------------------------------
# PDB access log — used by validation to prove no native structure is read
# during optimization.
# --------------------------------------------------------------------------
_PDB_ACCESS_LOG: List[str] = []


def reset_pdb_log() -> None:
    _PDB_ACCESS_LOG.clear()


def get_pdb_log() -> List[str]:
    return list(_PDB_ACCESS_LOG)


# --------------------------------------------------------------------------
# NeRF backbone construction
# --------------------------------------------------------------------------
def _place_atom(a, b, c, length: float, angle: float, torsion: float):
    """Natural Extension Reference Frame: place atom D given A-B-C.

    D is placed at distance `length` from C, with angle B-C-D = `angle`,
    and dihedral A-B-C-D = `torsion`.
    """
    bcx, bcy, bcz = c[0] - b[0], c[1] - b[1], c[2] - b[2]
    nb = math.sqrt(bcx * bcx + bcy * bcy + bcz * bcz)
    if nb < 1e-9:
        nb = 1e-9
    bcx, bcy, bcz = bcx / nb, bcy / nb, bcz / nb

    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    nx = aby * bcz - abz * bcy
    ny = abz * bcx - abx * bcz
    nz = abx * bcy - aby * bcx
    nn = math.sqrt(nx * nx + ny * ny + nz * nz)
    if nn < 1e-9:
        nx, ny, nz = 0.0, 0.0, 1.0
    else:
        nx, ny, nz = nx / nn, ny / nn, nz / nn

    mx = ny * bcz - nz * bcy
    my = nz * bcx - nx * bcz
    mz = nx * bcy - ny * bcx

    d0 = -length * math.cos(angle)
    sa = length * math.sin(angle)
    d1 = sa * math.cos(torsion)
    d2 = sa * math.sin(torsion)

    return (
        c[0] + d0 * bcx + d1 * mx + d2 * nx,
        c[1] + d0 * bcy + d1 * my + d2 * ny,
        c[2] + d0 * bcz + d1 * mz + d2 * nz,
    )


def place_cb(n, ca, c):
    """Ideal CB position from backbone N, CA, C (standard L-amino-acid geometry).

    The fixed coefficients encode the tetrahedral CB direction for an
    L-residue; this is what makes the representation CHIRAL.
    """
    bx, by, bz = ca[0] - n[0], ca[1] - n[1], ca[2] - n[2]
    dx, dy, dz = c[0] - ca[0], c[1] - ca[1], c[2] - ca[2]
    ax = by * dz - bz * dy
    ay = bz * dx - bx * dz
    az = bx * dy - by * dx
    return (
        -0.58273431 * ax + 0.56802827 * bx - 0.54067466 * dx + ca[0],
        -0.58273431 * ay + 0.56802827 * by - 0.54067466 * dy + ca[1],
        -0.58273431 * az + 0.56802827 * bz - 0.54067466 * dz + ca[2],
    )


def build_backbone(phi: np.ndarray, psi: np.ndarray,
                   omega: float = OMEGA_TRANS) -> Dict[str, np.ndarray]:
    """Build full backbone coordinates (Angstroms) from torsion angles.

    Returns dict with keys N, CA, C, CB, O, each an (n_res, 3) float array.
    """
    n_res = len(phi)
    if n_res < 1:
        raise ValueError("build_backbone requires at least one residue")

    N = [(0.0, 0.0, 0.0)] * n_res
    CA = [(0.0, 0.0, 0.0)] * n_res
    C = [(0.0, 0.0, 0.0)] * n_res

    N[0] = (0.0, 0.0, 0.0)
    CA[0] = (BOND_N_CA, 0.0, 0.0)
    C[0] = (
        CA[0][0] + BOND_CA_C * (-math.cos(ANGLE_N_CA_C)),
        CA[0][1] + BOND_CA_C * math.sin(ANGLE_N_CA_C),
        0.0,
    )

    for i in range(n_res - 1):
        N[i + 1] = _place_atom(N[i], CA[i], C[i], BOND_C_N, ANGLE_CA_C_N, psi[i])
        CA[i + 1] = _place_atom(CA[i], C[i], N[i + 1], BOND_N_CA, ANGLE_C_N_CA, omega)
        C[i + 1] = _place_atom(C[i], N[i + 1], CA[i + 1], BOND_CA_C, ANGLE_N_CA_C,
                               phi[i + 1])

    CB = [place_cb(N[i], CA[i], C[i]) for i in range(n_res)]
    O = [_place_atom(N[i], CA[i], C[i], BOND_C_O, ANGLE_CA_C_O, psi[i] + math.pi)
         for i in range(n_res)]

    return {
        "N": np.array(N, dtype=float),
        "CA": np.array(CA, dtype=float),
        "C": np.array(C, dtype=float),
        "CB": np.array(CB, dtype=float),
        "O": np.array(O, dtype=float),
    }


def amide_h_positions(N: np.ndarray, C: np.ndarray, O: np.ndarray) -> np.ndarray:
    """Amide H placed anti-parallel to the preceding C=O (DSSP convention).

    Residue 0 has no preceding carbonyl -> NaN row (excluded downstream).
    """
    N = np.asarray(N, dtype=float)
    C = np.asarray(C, dtype=float)
    O = np.asarray(O, dtype=float)
    n_res = len(N)
    H = np.full((n_res, 3), np.nan)
    for i in range(1, n_res):
        d = C[i - 1] - O[i - 1]
        nd = np.linalg.norm(d)
        if nd > 1e-6:
            H[i] = N[i] + d / nd
    return H


def dihedral(p0, p1, p2, p3) -> float:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    p2, p3 = np.asarray(p2, float), np.asarray(p3, float)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    nb1 = np.linalg.norm(b1)
    if nb1 < 1e-9:
        return 0.0
    b1n = b1 / nb1
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    return math.atan2(np.dot(np.cross(b1n, v), w), np.dot(v, w))


def extract_torsions(N, CA, C):
    """Extract (phi, psi) in radians from backbone coordinates."""
    N, CA, C = np.asarray(N, float), np.asarray(CA, float), np.asarray(C, float)
    n_res = len(CA)
    phi = np.zeros(n_res)
    psi = np.zeros(n_res)
    phi[0] = DEFAULT_PHI
    psi[n_res - 1] = DEFAULT_PSI
    for i in range(n_res):
        if i > 0:
            phi[i] = dihedral(C[i - 1], N[i], CA[i], C[i])
        if i < n_res - 1:
            psi[i] = dihedral(N[i], CA[i], C[i], N[i + 1])
    return phi, psi


# --------------------------------------------------------------------------
# Superposition and RMSD — ANGSTROMS ONLY
# --------------------------------------------------------------------------
def kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Optimally rotate+translate `mobile` onto `target`. No scaling.

    Both inputs are (n, 3) in Angstroms. Returns the transformed `mobile`
    in the frame of `target`. This is the ONLY superposition routine in the
    project; it always centers both sets internally.
    """
    P = np.asarray(mobile, dtype=float)
    Q = np.asarray(target, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch: {P.shape} vs {Q.shape}")
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return (R @ Pc.T).T + Q.mean(axis=0)


def kabsch_superpose_with_scale(mobile: np.ndarray,
                                target: np.ndarray) -> Tuple[np.ndarray, float]:
    """Superpose with a uniform isotropic scale factor.

    Used ONLY for the tetrahedral lattice representation, whose coordinates
    are in dimensionless lattice units and must be scaled to Angstroms before
    RMSD is meaningful. The scale is set so that mean consecutive-CA distance
    matches the target's, then Kabsch is applied. This generalizes the
    `fit_lattice_to_real` routine from the original implementation.

    Returns (transformed_mobile_in_angstroms, scale_factor).
    """
    P = np.asarray(mobile, dtype=float)
    Q = np.asarray(target, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"shape mismatch: {P.shape} vs {Q.shape}")
    p_bond = np.linalg.norm(np.diff(P, axis=0), axis=1).mean()
    q_bond = np.linalg.norm(np.diff(Q, axis=0), axis=1).mean()
    scale = (q_bond / p_bond) if p_bond > 1e-9 else 1.0
    return kabsch_superpose(P * scale, Q), float(scale)


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD in Angstroms between two already-superposed coordinate sets."""
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def ca_rmsd(pred_ca: np.ndarray, native_ca: np.ndarray,
            allow_scale: bool = False) -> float:
    """CA-RMSD in Angstroms after optimal superposition.

    `allow_scale=True` is ONLY for lattice coordinates in arbitrary units.
    """
    if allow_scale:
        aligned, _ = kabsch_superpose_with_scale(pred_ca, native_ca)
    else:
        aligned = kabsch_superpose(pred_ca, native_ca)
    return rmsd(aligned, native_ca)


def radius_of_gyration(coords: np.ndarray) -> float:
    c = np.asarray(coords, dtype=float)
    return float(np.sqrt(np.mean(np.sum((c - c.mean(axis=0)) ** 2, axis=1))))


# --------------------------------------------------------------------------
# Contact maps and secondary structure
# --------------------------------------------------------------------------
def contact_map(coords: np.ndarray, threshold: float = 8.0,
                min_sep: int = 3) -> Set[Tuple[int, int]]:
    c = np.asarray(coords, dtype=float)
    n = len(c)
    out = set()
    for i in range(n):
        for j in range(i + min_sep, n):
            if np.linalg.norm(c[i] - c[j]) < threshold:
                out.add((i, j))
    return out


def contact_metrics(pred: Set, native: Set) -> Tuple[float, float, float]:
    if not pred and not native:
        return 1.0, 1.0, 1.0
    inter = pred & native
    p = len(inter) / len(pred) if pred else 0.0
    r = len(inter) / len(native) if native else 0.0
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f1


def dssp_hbonds(coords: Dict[str, np.ndarray], min_sep: int = 2,
                cutoff: float = -0.5) -> List[Tuple[int, int, float]]:
    """All DSSP H-bonds below `cutoff` kcal/mol. Used for SS assignment only.

    (The energy function uses a matched variant; see energy_terms.hbond_terms.)
    """
    N, C, O = coords["N"], coords["C"], coords["O"]
    H = amide_h_positions(N, C, O)
    n = len(N)
    bonds = []
    for i in range(n):
        if not np.all(np.isfinite(H[i])):
            continue
        for j in range(n):
            if abs(i - j) < min_sep:
                continue
            r_ON = np.linalg.norm(N[i] - O[j])
            r_CH = np.linalg.norm(C[j] - H[i])
            r_OH = np.linalg.norm(H[i] - O[j])
            r_CN = np.linalg.norm(N[i] - C[j])
            if min(r_ON, r_CH, r_OH, r_CN) < 0.5:
                continue
            e = 0.084 * 332.0 * (1.0 / r_ON + 1.0 / r_CH - 1.0 / r_OH - 1.0 / r_CN)
            if e < cutoff:
                bonds.append((i, j, float(e)))
    return bonds


def assign_secondary_structure(coords: Dict[str, np.ndarray]) -> str:
    """Simplified DSSP: H (helix), E (strand), C (coil)."""
    n = len(coords["CA"])
    bonds = dssp_hbonds(coords, min_sep=2)
    ss = ["C"] * n
    partners: Dict[int, List[int]] = {}
    for i, j, _ in bonds:
        partners.setdefault(i, []).append(j)
        partners.setdefault(j, []).append(i)
    for i in range(n):
        if any(abs(j - i) in (3, 4) for j in partners.get(i, [])):
            ss[i] = "H"
    for i, j, _ in bonds:
        if abs(i - j) >= 5 and ss[i] != "H" and ss[j] != "H":
            ss[i] = "E"
            ss[j] = "E"
    return "".join(ss)


def ss_agreement(pred: str, native: str) -> float:
    n = min(len(pred), len(native))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if pred[i] == native[i]) / n


# --------------------------------------------------------------------------
# PDB I/O
# --------------------------------------------------------------------------
def parse_pdb(path: str, chain_id: Optional[str] = None):
    """Parse a PDB file -> (sequence, N, CA, C) arrays in Angstroms.

    Every call is recorded in the PDB access log so validation can prove the
    optimizer never touches native structures.
    """
    if not _HAVE_BIOPYTHON:
        raise RuntimeError("Biopython is required to parse PDB files.")
    _PDB_ACCESS_LOG.append(os.path.abspath(path))
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", path)
    seq, N, CA, C = [], [], [], []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                name = residue.resname.strip().upper()
                if name not in THREE_TO_ONE:
                    continue
                if not all(a in residue for a in ("N", "CA", "C")):
                    continue
                seq.append(THREE_TO_ONE[name])
                N.append(tuple(residue["N"].coord))
                CA.append(tuple(residue["CA"].coord))
                C.append(tuple(residue["C"].coord))
            break
        break
    if not CA:
        raise ValueError(f"No standard N/CA/C residues found in {path}")
    return ("".join(seq), np.array(N, float), np.array(CA, float),
            np.array(C, float))


def native_coords_from_pdb(path: str, chain_id: Optional[str] = None):
    """Full native coordinate dict (N, CA, C, CB, O) plus sequence and torsions."""
    seq, N, CA, C = parse_pdb(path, chain_id=chain_id)
    CB = np.array([place_cb(N[i], CA[i], C[i]) for i in range(len(CA))])
    phi, psi = extract_torsions(N, CA, C)
    O = np.array([_place_atom(N[i], CA[i], C[i], BOND_C_O, ANGLE_CA_C_O,
                              psi[i] + math.pi) for i in range(len(CA))])
    return seq, {"N": N, "CA": CA, "C": C, "CB": CB, "O": O}, phi, psi


def write_pdb(path: str, sequence: str, coords: Dict[str, np.ndarray],
              remark: str = "VQE predicted structure") -> None:
    with open(path, "w") as fh:
        fh.write(f"REMARK  {remark}\n")
        serial = 1
        for i, aa in enumerate(sequence):
            res = ONE_TO_THREE.get(aa, "GLY")
            for name, el in (("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")):
                if name not in coords:
                    continue
                x, y, z = (float(v) for v in coords[name][i])
                nm = (" " + name) if len(name) < 4 else name
                fh.write(
                    f"ATOM  {serial:>5d} {nm:<4s} {res:>3s} A{i + 1:>4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n")
                serial += 1
        fh.write("TER\nEND\n")