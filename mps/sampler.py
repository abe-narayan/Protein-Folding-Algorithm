"""Exact-to-cutoff matrix-product-state sampler for the global CVaR-VQE ansatz.

The ansatz is the same one built by ``vqe.build_global_circuit``: per layer, an RY
rotation on every qubit followed by a linear CNOT chain and (optionally) a closing
ring CNOT. :class:`MPSSampler` is a drop-in replacement for the lightning
``probs -> rng.choice`` sampling step, letting the driver reach qubit counts a dense
statevector cannot (the 30-qubit STATES_8 chignolin run).

Bit convention matches PennyLane ``qml.probs(wires=range(n))``: qubit 0 is the MOST
significant bit, so ``int(sample_string, 2)`` equals the PennyLane basis index.
"""
import numpy as np
import quimb.tensor as qtn

__all__ = ["MPSSampler"]


class MPSSampler:
    """Builds the ansatz as a quimb ``CircuitMPS`` and samples basis states from it.

    Parameters
    ----------
    n_qubits, layers, ring
        Ansatz shape (must match the driver's circuit).
    max_bond, cutoff
        MPS truncation controls. ``cutoff`` discards singular values below the
        threshold; ``max_bond`` caps the bond dimension.
    """

    def __init__(self, n_qubits: int, layers: int, ring: bool = True,
                 max_bond: int = 8192, cutoff: float = 1e-12):
        self.n = int(n_qubits)
        self.layers = int(layers)
        self.ring = bool(ring)
        self.max_bond = max_bond
        self.cutoff = float(cutoff)

    def _circuit(self, params) -> "qtn.CircuitMPS":
        p = np.asarray(params, dtype=float).reshape(self.layers, self.n)
        circ = qtn.CircuitMPS(N=self.n, max_bond=self.max_bond, cutoff=self.cutoff)
        for l in range(self.layers):
            for q in range(self.n):
                circ.apply_gate('RY', float(p[l][q]), q)
            for q in range(self.n - 1):
                circ.apply_gate('CNOT', q, q + 1)
            if self.ring and self.n > 2:
                circ.apply_gate('CNOT', self.n - 1, 0)
        return circ

    def draw_indices(self, params, shots: int, rng: np.random.Generator) -> np.ndarray:
        """Length-``shots`` int64 basis indices (duplicates intact), seeded from ``rng``."""
        circ = self._circuit(params)
        seed = int(rng.integers(0, 2 ** 63 - 1))
        it = circ.sample(int(shots), seed=seed)
        return np.fromiter((int(s, 2) for s in it), dtype=np.int64, count=int(shots))

    def probs(self, params) -> np.ndarray:
        """Full 2^n probability vector (validation / small-n only)."""
        d = np.asarray(self._circuit(params).psi.to_dense()).ravel()
        return np.abs(d) ** 2

    def bond_dim(self, params) -> int:
        return int(self._circuit(params).psi.max_bond())
