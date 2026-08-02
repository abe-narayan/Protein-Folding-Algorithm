"""Backbone geometry, superposition, PDB IO, and secondary-structure assignment.

2026-07-28 accuracy pass:

* **NMR ensembles are read as ensembles.** `parse_pdb` previously took the first chain of
  the first model and stopped. Most of `dataset.CANDIDATE_PDB_IDS` -- 1UAO, 1LE0, 1LE1,
  2EVQ, 1L2Y -- are solution NMR ensembles, so RMSD against model 1 is an arbitrary draw
  from the ensemble's own spread. `parse_pdb_ensemble` and `ca_rmsd_to_ensemble` report
  the min and mean over deposited models, which is the standard measure. This changes the
  reported number without changing any prediction, and is labelled as such wherever it
  surfaces.
* `contact_map` and `dssp_hbonds` are vectorised (they were O(n^2) Python loops calling
  `np.linalg.norm` on 3-vectors, four of them per pair in `dssp_hbonds`).
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


BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
BOND_C_O = 1.231
ANGLE_N_CA_C = math.radians(111.0)
ANGLE_CA_C_N = math.radians(116.2)
ANGLE_C_N_CA = math.radians(121.7)
ANGLE_CA_C_O = math.radians(120.8)
OMEGA_TRANS = math.pi

#: Shortest N...O separation that can still be a hydrogen bond, A. Below this the donor and
#: acceptor heavy atoms are interpenetrating. This lives here, not in `energy_terms`, because
#: the DSSP form has two independent copies -- `dssp_hbonds` below and
#: `energy_terms.hbond_terms` -- and they must not drift apart. `energy_terms` imports this
#: module, so it reads these as `geo.HB_MIN_ON` / `geo.HB_E_FLOOR`.
HB_MIN_ON = 2.6
#: Floor on one bond's DSSP energy. LOAD-BEARING, not redundant with HB_MIN_ON: at
#: dON = 2.6 the deepest energy this form can still reach is -5.62, in an anti-linear
#: geometry with the acceptor carbonyl C jammed 1.37 A from the donor N. -4.0 is the
#: deepest value the form assigns to a *correctly oriented* bond at the shortest
#: crystallographic N...O separation (-4.132), so the clamp costs nothing physical and
#: removes the residual plateau just inside the cutoff. Do not delete this as belt-and-braces.
HB_E_FLOOR = -4.0

DEFAULT_PHI = math.radians(-60.0)
DEFAULT_PSI = math.radians(-45.0)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items()}


_PDB_ACCESS_LOG: List[str] = []


def reset_pdb_log() -> None:
    _PDB_ACCESS_LOG.clear()


def get_pdb_log() -> List[str]:
    return list(_PDB_ACCESS_LOG)


# ==========================================================================
# NeRF backbone construction
# ==========================================================================
def _place_atom(a, b, c, length: float, angle: float, torsion: float):
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
    """Ideal-geometry backbone from torsions. Angstroms.

    Note that phi[0] is never read: the chain is grown from residue 0's frame, so only
    psi[0..n-2] and phi[1..n-1] affect CA positions. psi[n-1] affects only the terminal
    carbonyl oxygen.
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
    N = np.asarray(N, dtype=float)
    C = np.asarray(C, dtype=float)
    O = np.asarray(O, dtype=float)
    n_res = len(N)
    H = np.full((n_res, 3), np.nan)
    if n_res < 2:
        return H
    d = C[:-1] - O[:-1]
    nd = np.linalg.norm(d, axis=1)
    ok = nd > 1e-6
    H[1:][ok] = N[1:][ok] + d[ok] / nd[ok][:, None]
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
    """phi/psi from coordinates. phi[0] and psi[n-1] are genuinely undefined (they need
    the previous C and next N respectively) and are filled with DEFAULT_*; encoders
    should exclude them rather than match against the placeholder -- see
    `representations.TorsionStateRepresentation._nearest_state`."""
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


# ==========================================================================
# Superposition / metrics
# ==========================================================================
def kabsch_superpose(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
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
    """Superpose allowing a uniform scale, using the OPTIMAL least-squares scale.

    The previous version scaled by the ratio of mean bond lengths, which is not the
    scale that minimises RMSD, so it reported a worse number than the transform it
    claimed to compute. This uses the Umeyama scale, trace(R H) / trace(Pc^T Pc).

    Now that `representations.TetrahedralLatticeRepresentation.decode` returns Angstroms
    there is no legitimate reason to allow a free scale at all -- a similarity transform
    flatters the lattice and makes its RMSD incomparable with the torsion arm's. This is
    kept only for auditing historical numbers.
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
    var = float((Pc ** 2).sum())
    scale = float((S[:2].sum() + d * S[2]) / var) if var > 1e-12 else 1.0
    return kabsch_superpose(P * scale, Q), scale


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    if A.shape != B.shape:
        raise ValueError(f"shape mismatch: {A.shape} vs {B.shape}")
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def ca_rmsd(pred_ca: np.ndarray, native_ca: np.ndarray,
            allow_scale: bool = False) -> float:
    if allow_scale:
        aligned, _ = kabsch_superpose_with_scale(pred_ca, native_ca)
    else:
        aligned = kabsch_superpose(pred_ca, native_ca)
    return rmsd(aligned, native_ca)


def ca_rmsd_to_ensemble(pred_ca: np.ndarray,
                        ensemble_ca: Sequence[np.ndarray]) -> Dict[str, float]:
    """CA-RMSD against every deposited model of an NMR ensemble.

    For a solution structure the deposited models are all equally valid, so the min is
    the right headline and the spread across models is the floor below which a
    prediction cannot be meaningfully resolved. Reporting only model 1 -- which is what
    this repo did -- picks one arbitrarily.
    """
    vals = [ca_rmsd(pred_ca[:len(m)], np.asarray(m, float)[:len(pred_ca)])
            for m in ensemble_ca]
    arr = np.asarray(vals, dtype=float)
    return {
        "min": float(arr.min()), "mean": float(arr.mean()),
        "max": float(arr.max()), "model1": float(arr[0]),
        "best_model": int(np.argmin(arr)), "n_models": int(arr.size),
    }


def ensemble_spread(ensemble_ca: Sequence[np.ndarray]) -> float:
    """Mean pairwise CA-RMSD among deposited models -- the experimental resolution
    floor. A prediction closer than this is not distinguishable from the ensemble."""
    n = len(ensemble_ca)
    if n < 2:
        return 0.0
    vals = [ca_rmsd(np.asarray(ensemble_ca[i], float),
                    np.asarray(ensemble_ca[j], float))
            for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(vals))


def radius_of_gyration(coords: np.ndarray) -> float:
    c = np.asarray(coords, dtype=float)
    return float(np.sqrt(np.mean(np.sum((c - c.mean(axis=0)) ** 2, axis=1))))


# ==========================================================================
# Contacts and secondary structure
# ==========================================================================
def contact_map(coords: np.ndarray, threshold: float = 8.0,
                min_sep: int = 3) -> Set[Tuple[int, int]]:
    """Vectorised: was an O(n^2) Python loop with a norm call per pair."""
    c = np.asarray(coords, dtype=float)
    n = len(c)
    if n < min_sep + 1:
        return set()
    d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
    ii, jj = np.triu_indices(n, min_sep)
    hit = d[ii, jj] < threshold
    return set(zip(ii[hit].tolist(), jj[hit].tolist()))


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
    """DSSP electrostatic H-bonds as (donor_i, acceptor_j, energy). Vectorised."""
    N, C, O = coords["N"], coords["C"], coords["O"]
    H = amide_h_positions(N, C, O)
    n = len(N)
    valid = np.isfinite(H).all(axis=1)

    dON = np.linalg.norm(N[:, None, :] - O[None, :, :], axis=2)
    dCH = np.linalg.norm(C[None, :, :] - H[:, None, :], axis=2)
    dOH = np.linalg.norm(H[:, None, :] - O[None, :, :], axis=2)
    dCN = np.linalg.norm(N[:, None, :] - C[None, :, :], axis=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        E = 0.084 * 332.0 * (1.0 / dON + 1.0 / dCH - 1.0 / dOH - 1.0 / dCN)
    E = np.maximum(E, HB_E_FLOOR)

    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    ok = valid[:, None] & (sep >= min_sep)
    ok &= (dON > HB_MIN_ON) & (dCH > 0.5) & (dOH > 0.5) & (dCN > 0.5)
    ok &= np.isfinite(E) & (E < cutoff)
    di, aj = np.where(ok)
    return [(int(i), int(j), float(E[i, j])) for i, j in zip(di, aj)]


def assign_secondary_structure(coords: Dict[str, np.ndarray]) -> str:
    """Simplified DSSP: H (helix), E (strand), C (coil).

    Helix now requires *consecutive* n-turns (i->i+4 or i->i+3 at both i and i+1), which
    is what DSSP actually demands. A single i->i+4 bond was previously enough, so this
    over-called H; the effect was confined to the `ss_agreement` metric, not the search.
    """
    n = len(coords["CA"])
    bonds = dssp_hbonds(coords, min_sep=2)
    ss = ["C"] * n
    nturn = np.zeros(n, dtype=bool)
    for i, j, _ in bonds:
        if j - i in (3, 4):
            nturn[i] = True
        if i - j in (3, 4):
            nturn[j] = True
    for i in range(n - 1):
        if nturn[i] and nturn[i + 1]:
            span = 5 if i + 5 <= n else n - i
            for k in range(i, min(n, i + span)):
                ss[k] = "H"
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


# ==========================================================================
# PDB IO
# ==========================================================================
def _residues_of(chain):
    out = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        name = residue.resname.strip().upper()
        if name not in THREE_TO_ONE:
            continue
        if not all(a in residue for a in ("N", "CA", "C")):
            continue
        out.append((THREE_TO_ONE[name],
                    tuple(residue["N"].coord),
                    tuple(residue["CA"].coord),
                    tuple(residue["C"].coord)))
    return out


def parse_pdb_ensemble(path: str, chain_id: Optional[str] = None):
    """Parse every model in a PDB -> list of (sequence, N, CA, C).

    Most short-peptide targets in `dataset.CANDIDATE_PDB_IDS` are solution NMR
    ensembles, where each deposited model is equally valid. Models whose sequence
    differs from the first are dropped rather than silently mixed.

    Every call is recorded in the PDB access log so validation can prove the optimizer
    never touches native structures.
    """
    if not _HAVE_BIOPYTHON:
        raise RuntimeError("Biopython is required to parse PDB files.")
    _PDB_ACCESS_LOG.append(os.path.abspath(path))
    structure = PDBParser(QUIET=True).get_structure("protein", path)

    models = []
    ref_seq = None
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.id != chain_id:
                continue
            res = _residues_of(chain)
            if not res:
                continue
            seq = "".join(r[0] for r in res)
            if ref_seq is None:
                ref_seq = seq
            elif seq != ref_seq:
                break
            models.append((seq,
                           np.array([r[1] for r in res], float),
                           np.array([r[2] for r in res], float),
                           np.array([r[3] for r in res], float)))
            break
    if not models:
        raise ValueError(f"No standard N/CA/C residues found in {path}")
    return models


def parse_pdb(path: str, chain_id: Optional[str] = None, model_index: int = 0):
    """One model (default the first) -> (sequence, N, CA, C)."""
    models = parse_pdb_ensemble(path, chain_id=chain_id)
    return models[min(model_index, len(models) - 1)]


def _coords_from_backbone(seq, N, CA, C):
    CB = np.array([place_cb(N[i], CA[i], C[i]) for i in range(len(CA))])
    phi, psi = extract_torsions(N, CA, C)
    O = np.array([_place_atom(N[i], CA[i], C[i], BOND_C_O, ANGLE_CA_C_O,
                              psi[i] + math.pi) for i in range(len(CA))])
    return seq, {"N": N, "CA": CA, "C": C, "CB": CB, "O": O}, phi, psi


def native_coords_from_pdb(path: str, chain_id: Optional[str] = None,
                           model_index: int = 0):
    """Full native coordinate dict (N, CA, C, CB, O) plus sequence and torsions."""
    return _coords_from_backbone(*parse_pdb(path, chain_id=chain_id,
                                            model_index=model_index))


def native_ensemble_from_pdb(path: str, chain_id: Optional[str] = None):
    """Every deposited model as (sequence, coords, phi, psi)."""
    return [_coords_from_backbone(*m)
            for m in parse_pdb_ensemble(path, chain_id=chain_id)]


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
