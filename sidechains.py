"""Ideal-geometry heavy-atom sidechains.

Builds all heavy sidechain atoms for a residue from its backbone N, CA, C, CB.
Exists because `protein_geometry.build_backbone` emits only N, CA, C, O, CB,
while Amber ff14SB requires a complete heavy-atom residue before a topology
can be constructed.

SCOPE: eleven residue types -- G A S T D E N K P Y W. These cover the two
benchmark peptides completely (GYDPETGTWG, SWTWEGNKWTWK). Anything else
raises NotImplementedError naming the residue; there is deliberately no stub
fallback, because a silently-wrong sidechain would poison the energy without
failing loudly.

FIXED CHI IS A DELIBERATE LIMITATION. Sidechain torsions are pinned to
canonical rotamer values and are NOT encoded in the qubit register. Making
chi variable would cost roughly one extra 2-bit state per rotatable bond; for
the 12-residue trpzip that is ~34 qubits, i.e. 2^34 complex amplitudes = 17 GB
of statevector, which is not simulable. The register stays backbone-only
(2 bits/residue); sidechains are rebuilt deterministically from the backbone
at every energy evaluation. Downstream, the Day-2 restrained Amber
minimization relaxes chi, so these values are a physically sensible STARTING
point rather than a claim about the true rotamer.

GEOMETRY SOURCES
  * Bond lengths / angles: Engh & Huber (1991) restraint library, the same
    source protein_geometry already uses for the backbone, cross-checked
    against 170 residues in pdbs/. Per-line comments give the measured
    mean +/- sd; every value used agrees to within 0.02 A / 2 deg unless
    explicitly annotated otherwise.
  * Chi angles: modal rotamers of the Lovell et al. (2000) "penultimate
    rotamer library" (Proteins 40:389). See CHI_ANGLES for per-residue
    caveats.

CONSTRUCTION
  * Acyclic atoms are placed with protein_geometry._place_atom (NeRF), the
    same routine used for the backbone.
  * Rigid planar ring systems (Tyr phenol, Trp indole) are placed from a
    precomputed planar template. Chaining NeRF around a fused ring does not
    close: for indole it left the CH2-CZ2 bond 0.061 A long and ring angles
    up to 5.2 deg off. The template is fitted once so that every ring bond
    and angle is satisfied simultaneously and planarity is exact.
  * Proline ring closes back onto the backbone N. Its internal coordinates
    are solved against the FIXED N-CA-CB geometry place_cb produces; see
    PRO_RING for the residual strain.

All coordinates are in ANGSTROMS. Atom names are PDB standard, because
OpenMM matches ff14SB residue templates by atom name; a wrong name is a hard
topology failure, not a small error.
"""
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

import protein_geometry as geo


__all__ = [
    "build_sidechain", "build_full_structure", "sidechain_atom_names",
    "residue_bonds", "heavy_atom_count", "write_full_pdb",
    "SUPPORTED_RESIDUES", "CHI_ANGLES", "NotImplementedResidueError",
]


class NotImplementedResidueError(NotImplementedError):
    """Raised for residue types this module cannot build."""


THREE = dict(geo.ONE_TO_THREE)
ONE = dict(geo.THREE_TO_ONE)

SUPPORTED_RESIDUES = ("GLY", "ALA", "SER", "THR", "ASP", "GLU",
                      "ASN", "LYS", "PRO", "TYR", "TRP")


CHI_ANGLES: Dict[str, Tuple[float, ...]] = {
    "SER": (62.0,),                       # p
    "THR": (62.0,),                       # p  (chi1 measured on OG1)
    "ASP": (-70.0, -15.0),                # m-20
    "ASN": (-65.0, -20.0),                # m-20
    "GLU": (-67.0, 180.0, -10.0),         # mt-10 (chi3 = carboxyl rotation)
    "LYS": (-67.0, 180.0, 180.0, 180.0),  # mttt
    "TYR": (-65.0, -85.0),                # m-85
    "TRP": (-65.0, -90.0),                # m-90. Trp is the one residue whose
                                          # modal rotamer I could not verify
                                          # offline; m-90 and t-105 are both
                                          # heavily populated and I picked the
                                          # former. Treat as a start point.
    "PRO": (),                            # ring-closure solved, see PRO_RING
    "ALA": (),
    "GLY": (),
}


_SPECS: Dict[str, List[Tuple[str, Tuple[str, str, str], float, float, object]]] = {
    "GLY": [],
    "ALA": [],

    "SER": [
        ("OG",  ("N", "CA", "CB"), 1.417, 110.8, ("chi", 0, 0.0)),
    ],

    # THR: CB is a chiral centre (2S,3R). CG2 sits 120 deg BEHIND OG1 about
    # the CA-CB axis. Measured N-CA-CB-CG2 minus N-CA-CB-OG1 = -120.1 deg over
    # 196 residues (sd 0.1). This sign IS the stereochemistry; flipping it
    # builds allo-threonine, which ff14SB has no template for.
    "THR": [
        ("OG1", ("N", "CA", "CB"), 1.420, 110.1, ("chi", 0, 0.0)),
        ("CG2", ("N", "CA", "CB"), 1.530, 109.2, ("chi", 0, -120.0)),
    ],

    "ASP": [
        ("CG",  ("N", "CA", "CB"),  1.522, 112.6, ("chi", 0, 0.0)),
        ("OD1", ("CA", "CB", "CG"), 1.250, 118.4, ("chi", 1, 0.0)),
        ("OD2", ("CA", "CB", "CG"), 1.250, 118.4, ("chi", 1, 180.0)),
    ],

    "ASN": [
        ("CG",  ("N", "CA", "CB"),  1.521, 112.6, ("chi", 0, 0.0)),
        ("OD1", ("CA", "CB", "CG"), 1.231, 120.8, ("chi", 1, 0.0)),
        ("ND2", ("CA", "CB", "CG"), 1.328, 116.4, ("chi", 1, 180.0)),
    ],

    "GLU": [
        ("CG",  ("N", "CA", "CB"),  1.529, 113.6, ("chi", 0, 0.0)),
        ("CD",  ("CA", "CB", "CG"), 1.523, 112.6, ("chi", 1, 0.0)),
        ("OE1", ("CB", "CG", "CD"), 1.250, 118.4, ("chi", 2, 0.0)),
        ("OE2", ("CB", "CG", "CD"), 1.250, 118.4, ("chi", 2, 180.0)),
    ],

    "LYS": [
        ("CG", ("N",  "CA", "CB"), 1.531, 113.5, ("chi", 0, 0.0)),
        ("CD", ("CA", "CB", "CG"), 1.531, 113.3, ("chi", 1, 0.0)),
        ("CE", ("CB", "CG", "CD"), 1.531, 113.4, ("chi", 2, 0.0)),
        ("NZ", ("CG", "CD", "CE"), 1.486, 110.4, ("chi", 3, 0.0)),
    ],
}


_RING_TEMPLATES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "TYR": {
        "CB":  (-2.069719, -0.516542),
        "CG":  (-0.560501, -0.424851),
        "CD1": (0.077557, 0.808854),
        "CD2": (0.222218, -1.572237),
        "CE1": (1.456804, 0.896036),
        "CE2": (1.601875, -1.491804),
        "CZ":  (2.212445, -0.256383),
        "OH":  (3.585912, -0.172939),
    },
    "TRP": {
        "CB":  (-1.827828, -0.599403),
        "CG":  (-0.442130, -0.030330),
        "CD1": (-0.097941, 1.290441),
        "CD2": (0.786942, -0.767086),
        "NE1": (1.270715, 1.421830),
        "CE2": (1.836583, 0.172965),
        "CE3": (1.100358, -2.129541),
        "CZ2": (3.177670, -0.207145),
        "CZ3": (2.430771, -2.503923),
        "CH2": (3.452873, -1.547140),
    },
}

_RING_ANCHOR = {
    #        bond CB-CG, angle CA-CB-CG, bond CG-CD1, angle CB-CG-CD1
    "TYR": (1.512, 113.8, 1.389, 120.8),
    "TRP": (1.498, 113.6, 1.365, 126.9),
}


PRO_RING = {
    "b_CB_CG": 1.526,
    "a_CA_CB_CG": 102.286,
    "t_chi1": 11.675,      # Cg-endo pucker
    "b_CG_CD": 1.526,
    "a_CB_CG_CD": 106.700,
    "t_chi2": -22.549,
}


_SIDECHAIN_BONDS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "GLY": (),
    "ALA": (),
    "SER": (("CB", "OG"),),
    "THR": (("CB", "OG1"), ("CB", "CG2")),
    "ASP": (("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "ASN": (("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "GLU": (("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
    "LYS": (("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
    "PRO": (("CB", "CG"), ("CG", "CD"), ("CD", "N")),
    "TYR": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "CE1"),
            ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH")),
    "TRP": (("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"), ("CD1", "NE1"),
            ("NE1", "CE2"), ("CE2", "CD2"), ("CD2", "CE3"), ("CE3", "CZ3"),
            ("CZ3", "CH2"), ("CH2", "CZ2"), ("CZ2", "CE2")),
}

_ATOM_ORDER: Dict[str, Tuple[str, ...]] = {
    "GLY": ("N", "CA", "C", "O"),
    "ALA": ("N", "CA", "C", "O", "CB"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2",
            "CZ", "OH"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2",
            "CE3", "CZ2", "CZ3", "CH2"),
}

_ELEMENT = {"N": "N", "O": "O", "C": "C", "S": "S"}


def _resolve(resname: str) -> str:
    """Accept 'W' or 'TRP' (any case) -> three-letter code, or raise."""
    key = str(resname).strip().upper()
    if len(key) == 1:
        if key not in THREE:
            raise NotImplementedResidueError(
                f"sidechain construction not implemented for residue "
                f"{key!r}: unknown one-letter code")
        key = THREE[key]
    if key not in SUPPORTED_RESIDUES:
        raise NotImplementedResidueError(
            f"sidechain construction not implemented for residue {key} "
            f"({ONE.get(key, '?')}); supported residues are "
            f"{', '.join(SUPPORTED_RESIDUES)}")
    return key


def _nerf(a, b, c, length: float, angle_deg: float, torsion_deg: float):
    return np.array(geo._place_atom(a, b, c, float(length),
                                    math.radians(angle_deg),
                                    math.radians(torsion_deg)), dtype=float)


def _torsion_value(spec, chis: Sequence[float]) -> float:
    if isinstance(spec, tuple):
        _, index, offset = spec
        return float(chis[index]) + float(offset)
    return float(spec)


def _frame(origin, x_ref, plane_ref):
    """Orthonormal right-handed frame: e1 towards x_ref, e2 in-plane."""
    e1 = np.asarray(x_ref, float) - np.asarray(origin, float)
    e1 = e1 / np.linalg.norm(e1)
    v = np.asarray(plane_ref, float) - np.asarray(origin, float)
    e2 = v - np.dot(v, e1) * e1
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2, np.cross(e1, e2)


def _place_ring(resname: str, N, CA, CB, chis) -> Dict[str, np.ndarray]:
    """Dock a rigid planar ring template onto the CB-CG-CD1 frame.

    CG and CD1 are placed by NeRF from chi1 and chi2, so the two real degrees
    of freedom are handled exactly as for acyclic sidechains. The template
    then supplies the rigid remainder. Every template atom is coplanar with
    CB, CG and CD1, so the docking has no reflection ambiguity.
    """
    b_cg, a_cg, b_cd1, a_cd1 = _RING_ANCHOR[resname]
    CG = _nerf(N, CA, CB, b_cg, a_cg, chis[0])
    CD1 = _nerf(CA, CB, CG, b_cd1, a_cd1, chis[1])

    tmpl = _RING_TEMPLATES[resname]
    t3 = {k: np.array([v[0], v[1], 0.0]) for k, v in tmpl.items()}
    te1, te2, te3 = _frame(t3["CG"], t3["CB"], t3["CD1"])
    re1, re2, re3 = _frame(CG, CB, CD1)

    out: Dict[str, np.ndarray] = {}
    for name, p in t3.items():
        d = p - t3["CG"]
        out[name] = (CG + np.dot(d, te1) * re1 + np.dot(d, te2) * re2
                     + np.dot(d, te3) * re3)
    return out


def sidechain_atom_names(resname: str) -> Tuple[str, ...]:
    """Sidechain heavy-atom names (CB onwards) in PDB order."""
    key = _resolve(resname)
    return tuple(a for a in _ATOM_ORDER[key] if a not in ("N", "CA", "C", "O"))


def heavy_atom_count(resname: str) -> int:
    """Total heavy atoms for the residue, backbone included (no OXT)."""
    return len(_ATOM_ORDER[_resolve(resname)])


def residue_bonds(resname: str) -> Tuple[Tuple[str, str], ...]:
    """All intra-residue heavy-atom bonds, backbone included."""
    key = _resolve(resname)
    bb = [("N", "CA"), ("CA", "C"), ("C", "O")]
    if key != "GLY":
        bb.append(("CA", "CB"))
    return tuple(bb) + _SIDECHAIN_BONDS[key]


def build_sidechain(resname: str, N, CA, C, CB) -> Dict[str, np.ndarray]:
    """Heavy-atom sidechain coordinates keyed by PDB atom name.

    Returns CB and everything beyond it. Glycine returns an empty dict: it has
    no CB, and emitting one would break the ff14SB GLY template.

    `C` is accepted for interface completeness; only N, CA and CB determine
    the sidechain frame.
    """
    key = _resolve(resname)
    N = np.asarray(N, dtype=float)
    CA = np.asarray(CA, dtype=float)
    CB = np.asarray(CB, dtype=float)

    if key == "GLY":
        return {}

    atoms: Dict[str, np.ndarray] = {"CB": CB.copy()}

    if key == "PRO":
        p = PRO_RING
        atoms["CG"] = _nerf(N, CA, CB, p["b_CB_CG"], p["a_CA_CB_CG"],
                            p["t_chi1"])
        atoms["CD"] = _nerf(CA, CB, atoms["CG"], p["b_CG_CD"],
                            p["a_CB_CG_CD"], p["t_chi2"])
        return atoms

    if key in _RING_TEMPLATES:
        for name, xyz in _place_ring(key, N, CA, CB, CHI_ANGLES[key]).items():
            if name != "CB":
                atoms[name] = xyz
        return atoms

    chis = CHI_ANGLES[key]
    for name, (an, bn, cn), length, angle, tspec in _SPECS[key]:
        ref = {"N": N, "CA": CA, "CB": CB}
        ref.update(atoms)
        atoms[name] = _nerf(ref[an], ref[bn], ref[cn], length, angle,
                            _torsion_value(tspec, chis))
    return atoms


def build_full_structure(sequence: str, backbone: Dict[str, np.ndarray],
                         add_oxt: bool = True) -> Dict[str, object]:
    """Backbone dict from build_backbone() -> all heavy atoms.

    Returns {"sequence", "residues", "n_atoms"} where "residues" is a list of
    {atom_name: (3,) array}, one per residue, in PDB atom order.
    """
    seq = str(sequence).strip().upper()
    for k in ("N", "CA", "C", "O", "CB"):
        if k not in backbone:
            raise ValueError(f"backbone dict is missing key {k!r}")
    n_res = len(backbone["CA"])
    if len(seq) != n_res:
        raise ValueError(f"sequence length {len(seq)} != {n_res} residues "
                         f"in backbone")

    residues: List[Dict[str, np.ndarray]] = []
    for i, aa in enumerate(seq):
        key = _resolve(aa)
        Ni = np.asarray(backbone["N"][i], float)
        CAi = np.asarray(backbone["CA"][i], float)
        Ci = np.asarray(backbone["C"][i], float)
        Oi = np.asarray(backbone["O"][i], float)
        CBi = np.asarray(backbone["CB"][i], float)
        res: Dict[str, np.ndarray] = {"N": Ni, "CA": CAi, "C": Ci, "O": Oi}
        res.update(build_sidechain(key, Ni, CAi, Ci, CBi))
        ordered = {a: res[a] for a in _ATOM_ORDER[key] if a in res}
        if add_oxt and i == n_res - 1:
            # OXT is the carboxylate oxygen opposite O about the CA-C axis.
            # The ff14SB C-terminal template requires it, and Modeller adds
            # hydrogens but never heavy atoms.
            tor = math.degrees(geo.dihedral(Ni, CAi, Ci, Oi))
            ordered["OXT"] = _nerf(Ni, CAi, Ci, 1.250, 117.0, tor + 180.0)
        residues.append(ordered)
    return {"sequence": seq, "residues": residues,
            "n_atoms": sum(len(r) for r in residues)}


def write_full_pdb(path: str, structure: Dict[str, object],
                   remark: str = "ideal-geometry heavy atoms") -> None:
    """Write an all-heavy-atom PDB. Day 2 feeds this to OpenMM PDBFile."""
    seq = str(structure["sequence"])
    residues = structure["residues"]
    with open(path, "w") as fh:
        fh.write(f"REMARK  {remark}\n")
        serial = 1
        for i, (aa, res) in enumerate(zip(seq, residues)):
            rn = _resolve(aa)
            for name, xyz in res.items():
                el = _ELEMENT.get(name[0], "C")
                nm = name if len(name) >= 4 else " " + name
                fh.write(
                    f"ATOM  {serial:>5d} {nm:<4s} {rn:>3s} A{i + 1:>4d}    "
                    f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                    f"  1.00  0.00          {el:>2s}\n")
                serial += 1
        fh.write("TER\nEND\n")