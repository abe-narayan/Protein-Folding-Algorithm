
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import protein_geometry as geo



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

    def state_indices(self, bitstring: str) -> List[int]:
        self._check(bitstring)
        b = self.bits_per_residue
        return [int(bitstring[o:o + b], 2) % self.n_states for o in self.offsets]

    def decode(self, bitstring: str) -> Tuple[np.ndarray, np.ndarray]:
        """bitstring -> (phi, psi) arrays in radians."""
        idx = self.state_indices(bitstring)
        phi = self._phi[idx].copy()
        psi = self._psi[idx].copy()

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



LATTICE_DIRECTIONS = {
    (0, 0): (1.0, 1.0, 1.0),
    (0, 1): (1.0, -1.0, -1.0),
    (1, 0): (-1.0, 1.0, -1.0),
    (1, 1): (-1.0, -1.0, 1.0),
}


class TetrahedralLatticeRepresentation:


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

        ca = self.decode(bitstring)
        return {"CA": ca, "CB": ca.copy()}

    def native_bitstring(self, native_phi=None, native_psi=None,
                         native_ca: Optional[np.ndarray] = None,
                         rng: Optional[np.random.Generator] = None,
                         iterations: int = 20000) -> str:

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