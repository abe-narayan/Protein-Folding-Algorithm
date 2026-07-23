"""Discrete conformational representations for a GLOBAL VQE.

Two representations are provided:

  TorsionStateRepresentation   (default)  -- per-residue (phi, psi) state
  TetrahedralLatticeRepresentation        -- the original 4-direction lattice

Both expose the same interface so they can be swapped in ablations:

    n_qubits            int
    n_bits              int  (== n_qubits; H is diagonal)
    decode(bitstring)   -> geometry payload
    build_coords(bits)  -> dict of atom arrays (or CA-only for the lattice)
    random_bitstring(rng)
    native_bitstring(...)  -> best representable approximation to a native
    ceiling_rmsd(...)      -> best achievable CA-RMSD under this representation

QUBIT SCALING (this is the central design constraint):

    Torsion, K states:  ceil(log2 K) * N qubits
        K=4  -> 2N   qubits   (DEFAULT)
        K=8  -> 3N   qubits
    Tetrahedral lattice: 2(N-1) qubits

For a genuine full-system VQE the entire register is simulated at once, so
statevector memory is 2^n_qubits complex amplitudes. At 16 bytes each:

    20 qubits -> 16 MB      (N=10 torsion-4)
    24 qubits -> 268 MB     (N=12 torsion-4)
    28 qubits -> 4.3 GB     (N=14 torsion-4)
    30 qubits -> 17 GB      (N=15 torsion-4)  <-- practical wall
    40 qubits -> 17 TB      (N=20 torsion-4)  <-- NOT SIMULABLE

There is no way around this while keeping one global circuit. That is an
honest statement of the method's limit, not a defect of the implementation.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import protein_geometry as geo


# --------------------------------------------------------------------------
# Torsion state libraries
#
# Each state is a (phi, psi) pair sitting in a populated Ramachandran basin.
# Unlike a uniform torsion grid, every state is physically realizable, so no
# amplitude is wasted on forbidden regions.
#
# The 4-state library is the DEFAULT because it costs 2 bits/residue -- the
# same qubit count as the original tetrahedral lattice -- while being able to
# express alpha helices, beta strands, polyproline-II, and left-handed
# (glycine) turns with correct L-amino-acid chirality. The lattice can express
# none of these.
# --------------------------------------------------------------------------
STATES_4: List[Tuple[float, float]] = [
    (-63.0, -42.0),    # 0  alpha-R  : right-handed helix
    (-120.0, 130.0),   # 1  beta     : extended strand
    (-75.0, 150.0),    # 2  PPII     : polyproline-II / extended coil
    (60.0, 45.0),      # 3  alpha-L  : left-handed, glycine turns
]

STATES_8: List[Tuple[float, float]] = [
    (-63.0, -42.0),    # 0  alpha-R
    (-120.0, 130.0),   # 1  beta
    (-75.0, 150.0),    # 2  PPII
    (-140.0, 160.0),   # 3  extended beta
    (-60.0, -30.0),    # 4  helix N-cap / turn type I
    (-90.0, 0.0),      # 5  bridge / turn
    (60.0, 45.0),      # 6  alpha-L
    (90.0, -10.0),     # 7  glycine turn i+2 (type I'/II')
]

STATE_LIBRARIES = {4: STATES_4, 8: STATES_8}


def _bits_needed(k: int) -> int:
    b = 1
    while (1 << b) < k:
        b += 1
    return b


def _ang_diff_deg(a: float, b: float) -> float:
    return ((a - b + 180.0) % 360.0) - 180.0


class TorsionStateRepresentation:
    """Per-residue discrete (phi, psi) state. Default representation."""

    name = "torsion"
    is_lattice = False

    def __init__(self, n_residues: int, n_states: int = 4):
        if n_states not in STATE_LIBRARIES:
            raise ValueError(f"n_states must be one of {sorted(STATE_LIBRARIES)}")
        self.n_residues = int(n_residues)
        self.n_states = int(n_states)
        self.states = list(STATE_LIBRARIES[n_states])
        self.bits_per_residue = _bits_needed(self.n_states)
        self.n_bits = self.bits_per_residue * self.n_residues
        self.n_qubits = self.n_bits
        self.offsets = [i * self.bits_per_residue for i in range(self.n_residues)]
        self._phi = np.array([math.radians(p) for p, _ in self.states])
        self._psi = np.array([math.radians(q) for _, q in self.states])

    # -- decode -------------------------------------------------------------
    def state_indices(self, bitstring: str) -> List[int]:
        self._check(bitstring)
        b = self.bits_per_residue
        return [int(bitstring[o:o + b], 2) % self.n_states for o in self.offsets]

    def decode(self, bitstring: str) -> Tuple[np.ndarray, np.ndarray]:
        """bitstring -> (phi, psi) arrays in radians."""
        idx = self.state_indices(bitstring)
        phi = self._phi[idx].copy()
        psi = self._psi[idx].copy()
        # Terminal torsions are undefined; pin them to canonical values so
        # that the same bitstring always yields the same geometry.
        phi[0] = geo.DEFAULT_PHI
        psi[self.n_residues - 1] = geo.DEFAULT_PSI
        return phi, psi

    def build_coords(self, bitstring: str) -> Dict[str, np.ndarray]:
        phi, psi = self.decode(bitstring)
        return geo.build_backbone(phi, psi)

    # -- encode -------------------------------------------------------------
    def _nearest_state(self, phi_rad: float, psi_rad: float) -> int:
        pd, qd = math.degrees(phi_rad), math.degrees(psi_rad)
        best, best_d = 0, float("inf")
        for k, (p, q) in enumerate(self.states):
            d = _ang_diff_deg(pd, p) ** 2 + _ang_diff_deg(qd, q) ** 2
            if d < best_d:
                best_d, best = d, k
        return best

    def bitstring_from_states(self, states: Sequence[int]) -> str:
        w = self.bits_per_residue
        return "".join(format(int(s) % self.n_states, f"0{w}b") for s in states)

    def angles_to_bits(self, phi, psi) -> str:
        return self.bitstring_from_states(
            [self._nearest_state(phi[i], psi[i]) for i in range(self.n_residues)])

    def native_bitstring(self, native_phi, native_psi, **_) -> str:
        """Best representable approximation to a native structure.

        EVALUATION ONLY. Never used to seed or bias the VQE search.
        """
        return self.angles_to_bits(native_phi, native_psi)

    # -- sampling -----------------------------------------------------------
    def random_bitstring(self, rng: np.random.Generator) -> str:
        return self.bitstring_from_states(
            rng.integers(0, self.n_states, size=self.n_residues))

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
            "config_space": float(self.n_states) ** self.n_residues,
            "expresses_alpha_helix": True,
            "expresses_beta_strand": True,
            "expresses_turns": True,
            "chiral": True,
            "realistic_bond_geometry": True,
        }


# --------------------------------------------------------------------------
# Original tetrahedral lattice — RETAINED for ablation
# --------------------------------------------------------------------------
LATTICE_DIRECTIONS = {
    (0, 0): (1.0, 1.0, 1.0),
    (0, 1): (1.0, -1.0, -1.0),
    (1, 0): (-1.0, 1.0, -1.0),
    (1, 1): (-1.0, -1.0, 1.0),
}


class TetrahedralLatticeRepresentation:
    """The original 4-direction tetrahedral lattice.

    Retained VERBATIM in behaviour so that the "did the representation change
    help?" ablation is a fair test of your original design.

    Known limitations (documented, not hidden):
      * Bond angles are restricted to ~70.5/109.5/180 deg. Real CA-CA-CA
        pseudo-angles cluster near 90 (helix) and 120-130 (strand), so an
        alpha helix is NOT representable.
      * The lattice is achiral: mirror-image folds have identical energy, so
        right-handed helix preference cannot be encoded.
      * Coordinates are in dimensionless lattice units; RMSD requires an
        isotropic scale fit (handled by kabsch_superpose_with_scale).
      * Chain reversal (backtracking) is geometrically allowed.
    """

    name = "lattice"
    is_lattice = True

    def __init__(self, n_residues: int):
        self.n_residues = int(n_residues)
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
        """bitstring -> (n_res, 3) CA coordinates in LATTICE UNITS."""
        coords = [(0.0, 0.0, 0.0)]
        x = y = z = 0.0
        for bits in self.bond_directions(bitstring):
            dx, dy, dz = LATTICE_DIRECTIONS[bits]
            x, y, z = x + dx, y + dy, z + dz
            coords.append((x, y, z))
        return np.array(coords, dtype=float)

    def build_coords(self, bitstring: str) -> Dict[str, np.ndarray]:
        """CA-only. CB is aliased to CA so shared energy terms remain callable.

        No N/C/O exist on the lattice, so hydrogen-bond terms are structurally
        unavailable for this representation -- the Hamiltonian detects this
        and omits them. That asymmetry is reported, not silently absorbed.
        """
        ca = self.decode(bitstring)
        return {"CA": ca, "CB": ca.copy()}

    def native_bitstring(self, native_phi=None, native_psi=None,
                         native_ca: Optional[np.ndarray] = None,
                         rng: Optional[np.random.Generator] = None,
                         iterations: int = 20000) -> str:
        """Best lattice approximation to a native CA trace, by annealing.

        EVALUATION ONLY (representation ceiling). This function reads native
        coordinates and must never be called from the optimization path.
        Simulated annealing over bond directions, scoring by scaled Kabsch
        RMSD. This generalizes `real_structure_to_bitstring` from the
        original implementation.
        """
        if native_ca is None:
            raise ValueError("lattice native_bitstring requires native_ca")
        if rng is None:
            rng = np.random.default_rng(0)
        native_ca = np.asarray(native_ca, dtype=float)

        def score(bits_list) -> float:
            bs = "".join(format(d, "02b") for d in bits_list)
            return geo.ca_rmsd(self.decode(bs), native_ca, allow_scale=True)

        cur = list(rng.integers(0, 4, size=self.n_bonds))
        cur_s = score(cur)
        best, best_s = list(cur), cur_s
        T0, T1 = 2.0, 1e-3
        for k in range(iterations):
            T = T0 * (1 - k / max(1, iterations - 1)) + T1 * (k / max(1, iterations - 1))
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
            "expresses_alpha_helix": False,
            "expresses_beta_strand": "approximately",
            "expresses_turns": "coarsely",
            "chiral": False,
            "realistic_bond_geometry": False,
        }


def make_representation(kind: str, n_residues: int,
                        n_states: int = 4):
    kind = kind.lower()
    if kind == "torsion":
        return TorsionStateRepresentation(n_residues, n_states=n_states)
    if kind == "lattice":
        return TetrahedralLatticeRepresentation(n_residues)
    raise ValueError(f"unknown representation {kind!r}")