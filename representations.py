"""Bitstring -> structure encodings.

Two changes in the 2026-07-28 accuracy pass, both aimed at RMSD:

1. **The tetrahedral lattice now returns Angstroms.** It previously returned raw lattice
   units (bond norm sqrt(3) ~ 1.732) and handed them to an energy model parameterised in
   Angstroms, a 2.194x scale error. See `TetrahedralLatticeRepresentation.decode`.
2. **Torsion libraries are per-residue-class.** A single global list of (phi, psi) pairs
   was applied to every residue, so glycine was offered nothing that isoleucine was not
   and proline was allowed phi = +60. Libraries are now selected per residue from
   {GLY, PRO, PRE_PRO, GENERAL}. This lowers the representation's RMSD ceiling at
   identical qubit count and lets `energy_terms` carry a much smaller torsion weight
   without losing amino-acid specificity -- the specificity moves into the encoding,
   where evaluating it is free.

Both are properties of residue *classes* and of the lattice geometry. Neither is tuned
to any particular sequence.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import protein_geometry as geo
import sidechains as sc


#: One-letter codes whose chi1 is encoded: the aromatics whose ring this repo can build.
#: Chi1 rotates the ring about CA-CB, which is the degree of freedom that decides whether
#: a stacked aromatic pair is geometrically reachable. It costs one qubit per aromatic
#: residue -- two for a peptide with a single aromatic pair -- against ten to move a
#: 10-mer from 4 to 8 backbone states for 0.04 A of ceiling.
CHI1_ENCODED = tuple(sorted(geo.THREE_TO_ONE[r] for r in sc.AROMATIC_RING_RESIDUES))


# ==========================================================================
# Torsion state libraries
# ==========================================================================
# Convention, relied on by callers that build reference structures: index 0 is the
# helical state of its class and index 1 is the extended/beta state, in every library.
# `bitstring_from_states([0] * n)` is therefore "as helical as this sequence can be" and
# `[1] * n` is "as extended as it can be" regardless of composition.

#: Non-Gly, non-Pro, not immediately preceding a Pro.
#: State 3 is the bridge region rather than alpha-L: alpha-L is rare for residues with a
#: C-beta and common for glycine, so it belongs in the GLY library, and the bridge region
#: is what non-Gly residues actually use in turns.
GENERAL_4: List[Tuple[float, float]] = [
    (-63.0, -42.0),    # 0  alpha-R   right-handed helix
    (-120.0, 130.0),   # 1  beta      extended strand
    (-75.0, 150.0),    # 2  PPII      polyproline-II / extended coil
    (-90.0, 0.0),      # 3  bridge    turn / helix C-cap
]

GENERAL_8: List[Tuple[float, float]] = [
    (-63.0, -42.0),    # 0  alpha-R
    (-120.0, 130.0),   # 1  beta
    (-75.0, 150.0),    # 2  PPII
    (-140.0, 160.0),   # 3  extended beta
    (-60.0, -30.0),    # 4  helix N-cap / turn type I
    (-90.0, 0.0),      # 5  bridge
    (-100.0, 120.0),   # 6  beta-bridge
    (60.0, 45.0),      # 7  alpha-L   (allowed but rare with a C-beta)
]

#: Glycine. No C-beta, so the left-handed and mirror regions are genuinely populated --
#: for Gly they are not exotic, they are where a large fraction of Gly residues sit. The
#: previous shared library offered Gly exactly one of them.
GLY_4: List[Tuple[float, float]] = [
    (-63.0, -42.0),    # 0  alpha-R
    (-120.0, 130.0),   # 1  beta
    (60.0, 45.0),      # 2  alpha-L    heavily populated for Gly
    (90.0, -175.0),    # 3  epsilon    Gly-specific mirror-extended
]

GLY_8: List[Tuple[float, float]] = [
    (-63.0, -42.0),    # 0  alpha-R
    (-120.0, 130.0),   # 1  beta
    (60.0, 45.0),      # 2  alpha-L
    (90.0, -175.0),    # 3  epsilon
    (-90.0, 0.0),      # 4  bridge
    (75.0, -10.0),     # 5  alpha-L C-cap / type I' i+2
    (-80.0, 165.0),    # 6  PPII
    (105.0, 170.0),    # 7  mirror beta
]

#: Proline. phi is locked near -63 by the pyrrolidine ring; the ring cannot open to
#: +60, which the shared library allowed. The two real clusters are alpha-R-like and
#: PPII, so the four states resolve those two more finely instead of offering
#: geometrically impossible ones.
PRO_4: List[Tuple[float, float]] = [
    (-63.0, -35.0),    # 0  alpha-R-like
    (-63.0, 150.0),    # 1  PPII (the dominant Pro state)
    (-70.0, 120.0),    # 2  beta-ish, within Pro's phi range
    (-63.0, -10.0),    # 3  bridge-ish
]

PRO_8: List[Tuple[float, float]] = [
    (-63.0, -35.0),    # 0
    (-63.0, 150.0),    # 1
    (-70.0, 120.0),    # 2
    (-63.0, -10.0),    # 3
    (-55.0, 145.0),    # 4
    (-75.0, 160.0),    # 5
    (-60.0, -50.0),    # 6
    (-70.0, 175.0),    # 7
]

#: The residue immediately N-terminal to a proline. The following ring sterically
#: depletes alpha-R and enhances beta/PPII, and opens the zeta region that is otherwise
#: unused. Modelling this is why the library has to be sequence-context aware rather
#: than merely residue-type aware.
PRE_PRO_4: List[Tuple[float, float]] = [
    (-70.0, -35.0),    # 0  alpha-R    allowed but depleted
    (-130.0, 135.0),   # 1  beta       enhanced
    (-75.0, 155.0),    # 2  PPII       enhanced
    (-140.0, 80.0),    # 3  zeta       pre-Pro specific
]

PRE_PRO_8: List[Tuple[float, float]] = [
    (-70.0, -35.0),    # 0
    (-130.0, 135.0),   # 1
    (-75.0, 155.0),    # 2
    (-140.0, 80.0),    # 3
    (-100.0, 125.0),   # 4
    (-155.0, 160.0),   # 5
    (-85.0, 105.0),    # 6
    (-60.0, 140.0),    # 7
]

#: The exact pre-2026-07-28 flat libraries, preserved verbatim.
#:
#: `run_8state_seed0.py` and `amber_obc2.py` are pinned to a persisted set of
#: bitstring -> energy pairs (`IMPLEMENTATION_LOG.md`, golden CSVs) that
#: `partest/golden_check.py` verifies to max absolute error 0.0. Changing the torsion
#: library changes the coordinates a bitstring decodes to, and therefore every energy in
#: that set. Improving the libraries and keeping those runs reproducible are both
#: legitimate, so both exist: pass `legacy_library=True` to reproduce the golden data,
#: and leave it off for anything new. Do not "clean this up" by deleting it -- that
#: silently invalidates a 4-hour reference run.
LEGACY_4: List[Tuple[float, float]] = [
    (-63.0, -42.0),    # 0  alpha-R
    (-120.0, 130.0),   # 1  beta
    (-75.0, 150.0),    # 2  PPII
    (60.0, 45.0),      # 3  alpha-L
]

LEGACY_8: List[Tuple[float, float]] = [
    (-63.0, -42.0),    # 0  alpha-R
    (-120.0, 130.0),   # 1  beta
    (-75.0, 150.0),    # 2  PPII
    (-140.0, 160.0),   # 3  extended beta
    (-60.0, -30.0),    # 4  helix N-cap / turn type I
    (-90.0, 0.0),      # 5  bridge / turn
    (60.0, 45.0),      # 6  alpha-L
    (90.0, -10.0),     # 7  glycine turn i+2 (type I'/II')
]

LEGACY_LIBRARIES = {4: LEGACY_4, 8: LEGACY_8}

CLASS_GENERAL = "GENERAL"
CLASS_GLY = "GLY"
CLASS_PRO = "PRO"
CLASS_PRE_PRO = "PRE_PRO"
CLASS_LEGACY = "LEGACY"

STATE_LIBRARIES: Dict[int, Dict[str, List[Tuple[float, float]]]] = {
    4: {CLASS_GENERAL: GENERAL_4, CLASS_GLY: GLY_4,
        CLASS_PRO: PRO_4, CLASS_PRE_PRO: PRE_PRO_4},
    8: {CLASS_GENERAL: GENERAL_8, CLASS_GLY: GLY_8,
        CLASS_PRO: PRO_8, CLASS_PRE_PRO: PRE_PRO_8},
}

#: Kept for backward compatibility with code and tests that referenced the old flat
#: library. New code should go through `residue_classes` / `STATE_LIBRARIES`.
STATES_4 = GENERAL_4
STATES_8 = GENERAL_8


def residue_classes(sequence: Optional[str], n_residues: int) -> List[str]:
    """Per-residue library class. Falls back to GENERAL everywhere without a sequence."""
    if not sequence:
        return [CLASS_GENERAL] * n_residues
    seq = sequence.strip().upper()
    if len(seq) != n_residues:
        raise ValueError(f"sequence length {len(seq)} != n_residues {n_residues}")
    out = []
    for i, aa in enumerate(seq):
        nxt = seq[i + 1] if i + 1 < len(seq) else ""
        if aa == "G":
            out.append(CLASS_GLY)
        elif aa == "P":
            out.append(CLASS_PRO)
        elif nxt == "P":
            out.append(CLASS_PRE_PRO)
        else:
            out.append(CLASS_GENERAL)
    return out


def _bits_needed(k: int) -> int:
    b = 1
    while (1 << b) < k:
        b += 1
    return b


def _ang_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


# ==========================================================================
class TorsionStateRepresentation:
    """Per-residue (phi, psi) drawn from a discrete, residue-class-specific library."""

    name = "torsion"
    is_lattice = False

    def __init__(self, n_residues: int, n_states: int = 4,
                 sequence: Optional[str] = None,
                 legacy_library: bool = False,
                 chi_bits: bool = True):
        if n_states not in STATE_LIBRARIES:
            raise ValueError(f"n_states must be one of {sorted(STATE_LIBRARIES)}")
        self.n_residues = int(n_residues)
        self.n_states = int(n_states)
        self.sequence = sequence.strip().upper() if sequence else None
        self.legacy_library = bool(legacy_library)
        if self.legacy_library:
            # Reproduce the persisted golden runs exactly -- see LEGACY_4 / LEGACY_8.
            # `legacy_library` means the whole *encoding* is the old one, so chi1 bits are
            # off too: adding them would change n_bits and the bitstring -> structure map.
            self.classes = [CLASS_LEGACY] * self.n_residues
            self.libraries = [LEGACY_LIBRARIES[n_states]] * self.n_residues
            self.chi_bits = False
        else:
            self.classes = residue_classes(self.sequence, self.n_residues)
            libs = STATE_LIBRARIES[n_states]
            self.libraries = [libs[c] for c in self.classes]
            self.chi_bits = bool(chi_bits)

        self.bits_per_residue = _bits_needed(self.n_states)
        self.n_backbone_bits = self.bits_per_residue * self.n_residues
        # One bit per encoded aromatic, appended after the backbone bits so the backbone
        # slot layout is untouched -- search moves and the ceiling search address slots by
        # `i * bits_per_residue` and must keep working unchanged.
        self.chi_residues: Tuple[int, ...] = tuple(
            i for i, aa in enumerate(self.sequence or "") if aa in CHI1_ENCODED
        ) if self.chi_bits else tuple()
        self.n_chi_bits = len(self.chi_residues)
        self.n_bits = self.n_backbone_bits + self.n_chi_bits
        self.n_qubits = self.n_bits
        self.offsets = [i * self.bits_per_residue for i in range(self.n_residues)]
        self._rows = np.arange(self.n_residues)
        self._phi = np.array([[math.radians(p) for p, _ in lib]
                              for lib in self.libraries])
        self._psi = np.array([[math.radians(q) for _, q in lib]
                              for lib in self.libraries])
        # (residue index, 3-letter code, ring atom names) for the encoded aromatics,
        # resolved once instead of per structure.
        self._ring_spec = tuple(
            (i, geo.ONE_TO_THREE[self.sequence[i]],
             sc.ring_atom_names(geo.ONE_TO_THREE[self.sequence[i]]))
            for i in self.chi_residues)
        self._rotamers = tuple(
            sc.CHI1_ROTAMERS[geo.ONE_TO_THREE[self.sequence[i]]]
            for i in self.chi_residues)

    # -- unchecked internals ------------------------------------------------
    # These skip `_check` so a single validated entry point can do all the work without
    # re-validating. `hamiltonian.components` previously validated the bitstring four
    # times and decoded it twice per energy evaluation, on a path that runs 20,000 times
    # per search arm.
    def _states(self, bitstring: str) -> List[int]:
        b = self.bits_per_residue
        return [int(bitstring[o:o + b], 2) % self.n_states for o in self.offsets]

    def _angles(self, idx: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        return (self._phi[self._rows, idx].copy(),
                self._psi[self._rows, idx].copy())

    def _chi(self, bitstring: str) -> Dict[int, float]:
        off = self.n_backbone_bits
        return {i: float(rot[int(bitstring[off + k]) % len(rot)])
                for k, ((i, _, _), rot) in
                enumerate(zip(self._ring_spec, self._rotamers))}

    def _rings(self, coords: Dict[str, np.ndarray],
               chi: Dict[int, float]) -> Dict[int, Dict[str, np.ndarray]]:
        out: Dict[int, Dict[str, np.ndarray]] = {}
        for i, key, names in self._ring_spec:
            if not names:
                continue
            atoms = sc.build_sidechain(key, coords["N"][i], coords["CA"][i],
                                       coords["C"][i], coords["CB"][i],
                                       chi1=chi.get(i))
            out[i] = {n: atoms[n] for n in names if n in atoms}
        return out

    # -- decode -------------------------------------------------------------
    def state_indices(self, bitstring: str) -> List[int]:
        self._check(bitstring)
        return self._states(bitstring)

    def decode(self, bitstring: str) -> Tuple[np.ndarray, np.ndarray]:
        """bitstring -> (phi, psi) arrays in radians.

        Unlike the previous version this does NOT overwrite phi[0] and psi[n-1] with
        constants. `geo.build_backbone` never reads phi[0] for any CA position, but
        `energy_terms.rama_penalty` *was* being evaluated on it, so residue 0's torsion
        penalty was computed from a torsion the structure did not have. psi[n-1] does
        affect the terminal carbonyl, so it is likewise taken from the library.
        """
        self._check(bitstring)
        return self._angles(self._states(bitstring))

    def build_coords(self, bitstring: str) -> Dict[str, np.ndarray]:
        return geo.build_backbone(*self.decode(bitstring))

    def build_all(self, bitstring: str):
        """``(phi, psi, coords, rings)`` in one pass. The Hamiltonian's entry point.

        Validates once, decodes once, builds the backbone once, and builds the aromatic
        rings from that same backbone. `rings` is None when chi1 is not encoded, which is
        the signal for `energy_terms` to fall back to CB proxies.
        """
        self._check(bitstring)
        phi, psi = self._angles(self._states(bitstring))
        coords = geo.build_backbone(phi, psi)
        rings = self._rings(coords, self._chi(bitstring)) if self._ring_spec else None
        return phi, psi, coords, (rings or None)

    # -- chi1 ---------------------------------------------------------------
    def chi_indices(self, bitstring: str) -> List[int]:
        """Rotamer index (0 or 1) per entry of `chi_residues`."""
        self._check(bitstring)
        off = self.n_backbone_bits
        return [int(bitstring[off + k]) for k in range(self.n_chi_bits)]

    def chi1_degrees(self, bitstring: str) -> Dict[int, float]:
        """Residue index -> chi1 in degrees, for encoded residues only.

        Empty when chi encoding is off, in which case every consumer falls back to
        `sidechains.CHI_ANGLES` and builds what it always did.
        """
        if not self.n_chi_bits:
            return {}
        self._check(bitstring)
        return self._chi(bitstring)

    def build_rings(self, bitstring: str,
                    coords: Optional[Dict[str, np.ndarray]] = None
                    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Aromatic ring atoms per residue, honouring the encoded chi1.

        Only the residues in `chi_residues` are built -- typically two out of ten -- so
        this costs a couple of NeRF placements per structure, not a full sidechain build.
        `energy_terms` uses the returned atoms for a real pi-stacking geometry (ring
        centroid and normal) and for ring excluded volume, neither of which existed before.

        Returns {} when chi encoding is off, and callers then fall back to the CB proxy.
        Prefer `build_all` on the hot path -- it avoids rebuilding the backbone.
        """
        if not self.n_chi_bits:
            return {}
        self._check(bitstring)
        if coords is None:
            coords = geo.build_backbone(*self._angles(self._states(bitstring)))
        return self._rings(coords, self._chi(bitstring))

    # -- encode -------------------------------------------------------------
    def _nearest_state(self, i: int, phi_rad: Optional[float],
                       psi_rad: Optional[float]) -> int:
        """Closest library state for residue `i`.

        `phi_rad`/`psi_rad` may be None for a terminus, where the torsion is genuinely
        undefined (phi[0] needs the previous residue's C, psi[n-1] the next residue's N).
        Scoring on a placeholder there would bias the snapped native.
        """
        pd = None if phi_rad is None else math.degrees(phi_rad)
        qd = None if psi_rad is None else math.degrees(psi_rad)
        best, best_d = 0, float("inf")
        for k, (p, q) in enumerate(self.libraries[i]):
            d = 0.0
            if pd is not None:
                d += _ang_diff_deg(pd, p) ** 2
            if qd is not None:
                d += _ang_diff_deg(qd, q) ** 2
            if d < best_d:
                best_d, best = d, k
        return best

    def bitstring_from_states(self, states: Sequence[int],
                              chi_states: Optional[Sequence[int]] = None) -> str:
        """Backbone states (+ optional chi rotamer indices) -> bitstring.

        `chi_states=None` sets every chi bit to 0, which is rotamer index 0 -- i.e. exactly
        the angle in `sidechains.CHI_ANGLES`. So `bitstring_from_states([0] * n)` still
        means "helical, default rotamers" as it always did.
        """
        w = self.bits_per_residue
        if len(states) != self.n_residues:
            raise ValueError(
                f"got {len(states)} states for {self.n_residues} residues")
        bits = "".join(format(int(s) % self.n_states, f"0{w}b") for s in states)
        if not self.n_chi_bits:
            return bits
        if chi_states is None:
            chi = [0] * self.n_chi_bits
        else:
            chi = list(chi_states)
            # Silently truncating would emit a short bitstring that every downstream
            # `_check` then rejects with a confusing message about n_bits.
            if len(chi) != self.n_chi_bits:
                raise ValueError(
                    f"got {len(chi)} chi states for {self.n_chi_bits} chi bits "
                    f"(residues {self.chi_residues})")
        return bits + "".join(str(int(c) % 2) for c in chi)

    def angles_to_bits(self, phi, psi,
                       chi_states: Optional[Sequence[int]] = None) -> str:
        n = self.n_residues
        return self.bitstring_from_states(
            [self._nearest_state(i,
                                 None if i == 0 else phi[i],
                                 None if i == n - 1 else psi[i])
             for i in range(n)],
            chi_states=chi_states)

    def native_bitstring(self, native_phi, native_psi, **_) -> str:
        """Native backbone, default chi rotamers.

        Chi1 is deliberately not snapped from the native structure: `protein_geometry`
        reads only N, CA and C, so a native chi1 is not available, and chi1 does not affect
        CA positions -- so the representation ceiling, which is a pure CA-RMSD quantity, is
        unaffected either way.
        """
        return self.angles_to_bits(native_phi, native_psi)

    # -- sampling -----------------------------------------------------------
    def random_bitstring(self, rng: np.random.Generator) -> str:
        return self.bitstring_from_states(
            rng.integers(0, self.n_states, size=self.n_residues),
            chi_states=(rng.integers(0, 2, size=self.n_chi_bits)
                        if self.n_chi_bits else None))

    def _check(self, bitstring: str) -> None:
        if len(bitstring) != self.n_bits:
            raise ValueError(
                f"bitstring length {len(bitstring)} != n_bits {self.n_bits}")

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_states": self.n_states,
            "bits_per_residue": self.bits_per_residue,
            "n_qubits": self.n_qubits,
            "n_backbone_bits": self.n_backbone_bits,
            "n_chi_bits": self.n_chi_bits,
            "chi_residues": list(self.chi_residues),
            "config_space": (float(self.n_states) ** self.n_residues
                             * float(2 ** self.n_chi_bits)),
            "residue_classes": list(self.classes),
            "sequence_aware": self.sequence is not None and not self.legacy_library,
            "legacy_library": self.legacy_library,
            "expresses_alpha_helix": True,
            "expresses_beta_strand": True,
            "expresses_turns": True,
            "chiral": True,
            "realistic_bond_geometry": True,
        }


# ==========================================================================
LATTICE_DIRECTIONS = {
    (0, 0): (1.0, 1.0, 1.0),
    (0, 1): (1.0, -1.0, -1.0),
    (1, 0): (-1.0, 1.0, -1.0),
    (1, 1): (-1.0, -1.0, 1.0),
}

#: Virtual CA-CA bond length in a real polypeptide.
CA_CA_DISTANCE = 3.80
#: Lattice bond vectors have norm sqrt(3), so this puts adjacent CA atoms at 3.80 A.
LATTICE_SCALE = CA_CA_DISTANCE / math.sqrt(3.0)      # 2.19393...


class TetrahedralLatticeRepresentation:
    """CA-only tetrahedral lattice trace, 2 bits per bond."""

    name = "lattice"
    is_lattice = True

    def __init__(self, n_residues: int, sequence: Optional[str] = None):
        self.n_residues = int(n_residues)
        self.sequence = sequence.strip().upper() if sequence else None
        self.n_bonds = self.n_residues - 1
        self.n_bits = 2 * self.n_bonds
        self.n_qubits = self.n_bits
        self.bits_per_residue = 2  # per bond, actually

    def _check(self, bitstring: str) -> None:
        if len(bitstring) != self.n_bits:
            raise ValueError(
                f"bitstring length {len(bitstring)} != n_bits {self.n_bits}")

    def bond_directions(self, bitstring: str) -> List[Tuple[int, int]]:
        self._check(bitstring)
        return [(int(bitstring[2 * i]), int(bitstring[2 * i + 1]))
                for i in range(self.n_bonds)]

    def decode(self, bitstring: str) -> np.ndarray:
        """bitstring -> (n_res, 3) CA coordinates in ANGSTROMS.

        Scaled by `LATTICE_SCALE` so adjacent CA atoms sit at 3.80 A. This scaling was
        missing, and it was the largest single defect in the model: the raw lattice bond
        norm is sqrt(3) ~ 1.732, so every threshold in `energy_terms` -- all of which are
        in Angstroms -- was wrong by 2.194x on this representation. The visible
        consequence was in the steric term, which charged 20.80 for a |i-j| = 2 turn
        (d = 2.00 "A") against 0.45 for a straight continuation (d = 3.46), a 46x penalty
        for turning that came entirely from the unit mismatch and drove every search to
        the extended chain. Term-by-term accounting of the resulting all-straight optimum
        reproduces the previously published lattice minimum of 9.847041.

        With correct scaling, adjacent CA is exactly 3.80 and |i-j| >= 2 is at least
        4.39, so the steric term correctly reads zero except for true backtracking --
        which `energy_terms.backtracking_term` handles explicitly.
        """
        coords = [(0.0, 0.0, 0.0)]
        x = y = z = 0.0
        for bits in self.bond_directions(bitstring):
            dx, dy, dz = LATTICE_DIRECTIONS[bits]
            x, y, z = x + dx, y + dy, z + dz
            coords.append((x, y, z))
        return np.array(coords, dtype=float) * LATTICE_SCALE

    def build_coords(self, bitstring: str) -> Dict[str, np.ndarray]:
        """CA only.

        The previous version also returned `"CB": ca.copy()`, which made every CB
        distance a duplicate of a CA distance and caused the steric term to charge the
        same pair twice at two different thresholds. `energy_terms.energy_components`
        already falls back to CA when CB is absent, which is the honest behaviour for a
        CA-only model.
        """
        return {"CA": self.decode(bitstring)}

    def build_all(self, bitstring: str):
        """``(phi, psi, coords, rings)``, matching the torsion representation's interface.

        A CA-only trace has no torsions and no sidechains, so phi/psi/rings are all None --
        which is what makes `energy_terms` report `torsion`, both `hbond_*` terms and both
        cooperativity terms as exactly zero on this representation.
        """
        return None, None, {"CA": self.decode(bitstring)}, None

    def native_bitstring(self, native_phi=None, native_psi=None,
                         native_ca: Optional[np.ndarray] = None,
                         rng: Optional[np.random.Generator] = None,
                         iterations: int = 20000) -> str:
        """Best lattice trace for a native CA chain, by annealed search on CA-RMSD."""
        if native_ca is None:
            raise ValueError("lattice native_bitstring requires native_ca")
        if rng is None:
            rng = np.random.default_rng(0)
        native_ca = np.asarray(native_ca, dtype=float)

        def score(bits_list) -> float:
            bs = "".join(format(d, "02b") for d in bits_list)
            # allow_scale is no longer needed now that decode() returns Angstroms, but
            # is kept off deliberately: a free similarity scale flatters the lattice and
            # makes its RMSD incomparable with the torsion arm's.
            return geo.ca_rmsd(self.decode(bs), native_ca, allow_scale=False)

        cur = list(rng.integers(0, 4, size=self.n_bonds))
        cur_s = score(cur)
        best, best_s = list(cur), cur_s
        T0, T1 = 2.0, 1e-3
        for k in range(iterations):
            frac = k / max(1, iterations - 1)
            T = T0 * (1 - frac) + T1 * frac
            cand = list(cur)
            cand[int(rng.integers(0, self.n_bonds))] = int(rng.integers(0, 4))
            cs = score(cand)
            if cs < cur_s or rng.random() < math.exp(-(cs - cur_s) / max(T, 1e-9)):
                cur, cur_s = cand, cs
                if cs < best_s:
                    best, best_s = list(cand), cs
        return "".join(format(d, "02b") for d in best)

    def random_bitstring(self, rng: np.random.Generator) -> str:
        return "".join(format(int(d), "02b")
                       for d in rng.integers(0, 4, size=self.n_bonds))

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_states": 4,
            "bits_per_residue": 2,
            "n_qubits": self.n_qubits,
            "config_space": 4.0 ** self.n_bonds,
            "ca_ca_distance": CA_CA_DISTANCE,
            "expresses_alpha_helix": False,
            "expresses_beta_strand": "approximately",
            "expresses_turns": "coarsely",
            "chiral": False,
            "realistic_bond_geometry": False,
        }


def make_representation(kind: str, n_residues: int, n_states: int = 4,
                        sequence: Optional[str] = None,
                        legacy_library: bool = False,
                        chi_bits: bool = True):
    """Build a representation.

    `sequence` is optional for backward compatibility but should always be supplied for
    the torsion representation: without it every residue falls back to the GENERAL
    library, glycine/proline/pre-proline lose their class-specific states, and no chi1
    bits are allocated.

    `legacy_library=True` restores the pre-2026-07-28 encoding entirely (flat library, no
    chi1 bits) for reproducing the persisted golden runs. See `LEGACY_4`.
    """
    kind = kind.lower()
    if kind == "torsion":
        return TorsionStateRepresentation(n_residues, n_states=n_states,
                                          sequence=sequence,
                                          legacy_library=legacy_library,
                                          chi_bits=chi_bits)
    if kind == "lattice":
        return TetrahedralLatticeRepresentation(n_residues, sequence=sequence)
    raise ValueError(f"unknown representation {kind!r}")
