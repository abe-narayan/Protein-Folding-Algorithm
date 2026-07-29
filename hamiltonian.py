"""Knowledge-based Hamiltonian over a discrete structure encoding."""
import itertools
from typing import Dict, Optional

import numpy as np

import energy_terms as et
import protein_geometry as geo
import sidechains as sc
from budget import BudgetedEnergyModel


#: Cap on the chi1 rotamer combinations scanned when scoring a *native* structure. The
#: native's chi1 is not available from the PDB (`protein_geometry` reads N, CA and C only),
#: so the fair comparison gives the native the same rotamer freedom the encoding gives a
#: prediction rather than pinning it to one rotamer -- otherwise the native is handicapped
#: by a degree of freedom the prediction gets to optimise. 2**6 = 64 evaluations is a
#: generous ceiling; beyond it the scan falls back to rotamer 0.
MAX_NATIVE_CHI_SCAN_BITS = 6


class FoldingHamiltonian(BudgetedEnergyModel):

    def __init__(self, sequence: str, representation,
                 weights: Optional[Dict[str, float]] = None,
                 use_corrected_mj: bool = True,
                 backtracking_penalty: float = 5.0,
                 cache_limit: int = 500_000,
                 eval_budget: Optional[int] = None):
        self.sequence = sequence.strip().upper()
        self.rep = representation
        if len(self.sequence) != self.rep.n_residues:
            raise ValueError(
                f"sequence length {len(self.sequence)} != representation "
                f"n_residues {self.rep.n_residues}")
        self.weights = dict(et.DEFAULT_WEIGHTS if weights is None else weights)
        self.use_corrected_mj = bool(use_corrected_mj)
        self.backtracking_penalty = float(backtracking_penalty)
        self._init_budget(cache_limit, eval_budget)

    @property
    def n_qubits(self) -> int:
        return self.rep.n_qubits

    @property
    def n_bits(self) -> int:
        return self.rep.n_bits

    # -- aromatic rings -----------------------------------------------------
    def _ring_residues(self):
        """Residue indices whose aromatic ring can be built, with their 3-letter code."""
        return [(i, geo.ONE_TO_THREE[aa]) for i, aa in enumerate(self.sequence)
                if geo.ONE_TO_THREE[aa] in sc.AROMATIC_RING_RESIDUES]

    def _rings_from_coords(self, coords: Dict[str, np.ndarray],
                           chi1: Optional[Dict[int, float]] = None
                           ) -> Dict[int, Dict[str, np.ndarray]]:
        """Aromatic ring atoms from a full backbone, at the given chi1.

        Used for structures that did not come from a bitstring -- natives, mainly -- so
        their aromatic term is computed with the *same* functional form as a prediction's.
        Without this the native would silently fall back to the CB proxy while predictions
        used real ring geometry, and the two energies would not be comparable at all.
        """
        if not all(k in coords for k in ("N", "CA", "C", "CB")):
            return {}
        chi1 = chi1 or {}
        out: Dict[int, Dict[str, np.ndarray]] = {}
        for i, key in self._ring_residues():
            names = sc.ring_atom_names(key)
            if not names:
                continue
            atoms = sc.build_sidechain(key, coords["N"][i], coords["CA"][i],
                                       coords["C"][i], coords["CB"][i],
                                       chi1=chi1.get(i))
            out[i] = {n: atoms[n] for n in names if n in atoms}
        return out

    # -- energy -------------------------------------------------------------
    def components(self, bitstring: str) -> Dict[str, float]:
        # One pass through the representation: validate once, decode once, build the
        # backbone once, and build the aromatic rings from that same backbone. Calling
        # decode() and build_coords() separately validated the bitstring four times and
        # decoded it twice, on a path that runs 20,000 times per search arm.
        # `rings` is None unless chi1 is encoded, which is the CB-proxy fallback signal.
        phi, psi, coords, rings = self.rep.build_all(bitstring)
        comp = et.energy_components(self.sequence, coords, phi, psi,
                                    use_corrected_mj=self.use_corrected_mj,
                                    rings=rings)
        comp["backtracking"] = (self.backtracking_penalty
                                * et.backtracking_term(self.rep, bitstring))
        return comp

    def energy(self, bitstring: str) -> float:
        hit = self._cache.get(bitstring)
        if hit is not None:
            return hit                  # cache hits are free -- revisiting costs nothing
        self._charge()                  # raises BudgetExhausted before doing the work
        comp = self.components(bitstring)
        e = et.total_from_components(comp, self.weights) + comp["backtracking"]
        if len(self._cache) < self._cache_limit:
            self._cache[bitstring] = e
        return e

    def energy_from_coords(self, coords: Dict[str, np.ndarray],
                           phi=None, psi=None,
                           chi1: Optional[Dict[int, float]] = None,
                           scan_chi: bool = True) -> float:
        """Energy of an arbitrary structure, typically a native.

        `scan_chi=True` minimises over the encoded chi1 rotamers, giving this structure the
        same sidechain freedom a bitstring-encoded prediction has. Pass an explicit `chi1`,
        or `scan_chi=False`, to pin the rotamers instead.
        """
        def evaluate(chi):
            comp = et.energy_components(
                self.sequence, coords, phi, psi,
                use_corrected_mj=self.use_corrected_mj,
                rings=self._rings_from_coords(coords, chi) or None)
            return et.total_from_components(comp, self.weights)

        if chi1 is not None or not scan_chi:
            return evaluate(chi1)

        residues = [i for i, _ in self._ring_residues()]
        if not residues or len(residues) > MAX_NATIVE_CHI_SCAN_BITS:
            return evaluate(None)
        rot = {i: sc.CHI1_ROTAMERS[geo.ONE_TO_THREE[self.sequence[i]]]
               for i in residues}
        best = float("inf")
        for combo in itertools.product(*(range(len(rot[i])) for i in residues)):
            best = min(best, evaluate({i: rot[i][k]
                                       for i, k in zip(residues, combo)}))
        return best
