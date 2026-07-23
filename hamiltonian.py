"""The diagonal folding Hamiltonian.

    H = sum_x E(x) |x><x|

E(x) is the weighted sum of the terms in energy_terms.py, evaluated on the
structure that representation `rep` decodes from bitstring x.

HONESTY NOTE (repeated in the audit): H is DIAGONAL in the computational
basis. Consequently the VQE expectation value <psi|H|psi> = sum_x p(x) E(x)
is exactly an expectation over a classical probability distribution. The
parameterized circuit is a device for producing that distribution; no
off-diagonal quantum dynamics is involved. Everything in this project is
about whether a variational quantum *distribution* can be optimized well,
not about quantum speedup.

LEAKAGE GUARANTEE: this class never touches native coordinates. Its inputs
are a sequence string, a representation, and weights. There is deliberately
no code path by which a PDB file can influence E(x).
"""
from typing import Dict, Optional

import numpy as np

import energy_terms as et


class FoldingHamiltonian:

    def __init__(self, sequence: str, representation,
                 weights: Optional[Dict[str, float]] = None,
                 use_corrected_mj: bool = True,
                 backtracking_penalty: float = 5.0,
                 cache_limit: int = 500_000):
        self.sequence = sequence.strip().upper()
        self.rep = representation
        if len(self.sequence) != self.rep.n_residues:
            raise ValueError(
                f"sequence length {len(self.sequence)} != representation "
                f"n_residues {self.rep.n_residues}")
        self.weights = dict(et.DEFAULT_WEIGHTS if weights is None else weights)
        self.use_corrected_mj = bool(use_corrected_mj)
        self.backtracking_penalty = float(backtracking_penalty)
        self._cache: Dict[str, float] = {}
        self._cache_limit = int(cache_limit)
        self.n_energy_evaluations = 0   # cache MISSES only == real work

    # -- properties ---------------------------------------------------------
    @property
    def n_qubits(self) -> int:
        return self.rep.n_qubits

    @property
    def n_bits(self) -> int:
        return self.rep.n_bits

    # -- core ---------------------------------------------------------------
    def components(self, bitstring: str) -> Dict[str, float]:
        """Per-term breakdown for one bitstring. Not cached."""
        if getattr(self.rep, "is_lattice", False):
            coords = self.rep.build_coords(bitstring)
            phi = psi = None
        else:
            phi, psi = self.rep.decode(bitstring)
            coords = self.rep.build_coords(bitstring)
        comp = et.energy_components(self.sequence, coords, phi, psi,
                                    use_corrected_mj=self.use_corrected_mj)
        comp["backtracking"] = (self.backtracking_penalty
                                * et.backtracking_term(self.rep, bitstring))
        return comp

    def energy(self, bitstring: str) -> float:
        """E(x). Cached; cache hits do not count as energy evaluations."""
        hit = self._cache.get(bitstring)
        if hit is not None:
            return hit
        comp = self.components(bitstring)
        e = et.total_from_components(comp, self.weights) + comp["backtracking"]
        if len(self._cache) < self._cache_limit:
            self._cache[bitstring] = e
        self.n_energy_evaluations += 1
        return e

    def energy_from_coords(self, coords: Dict[str, np.ndarray],
                           phi=None, psi=None) -> float:
        """Energy of an arbitrary structure under the SAME Hamiltonian.

        Used to score the native structure for the energy-gap metric. This is
        an EVALUATION path, not an optimization path.
        """
        comp = et.energy_components(self.sequence, coords, phi, psi,
                                    use_corrected_mj=self.use_corrected_mj)
        return et.total_from_components(comp, self.weights)

    def reset_counters(self) -> None:
        self.n_energy_evaluations = 0

    def cache_size(self) -> int:
        return len(self._cache)