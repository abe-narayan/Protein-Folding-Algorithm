
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
        self.n_energy_evaluations = 0  

    @property
    def n_qubits(self) -> int:
        return self.rep.n_qubits

    @property
    def n_bits(self) -> int:
        return self.rep.n_bits

    def components(self, bitstring: str) -> Dict[str, float]:
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
        comp = et.energy_components(self.sequence, coords, phi, psi,
                                    use_corrected_mj=self.use_corrected_mj)
        return et.total_from_components(comp, self.weights)

    def reset_counters(self) -> None:
        self.n_energy_evaluations = 0

    def cache_size(self) -> int:
        return len(self._cache)