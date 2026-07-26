"""MPS bitstring sampler for the global CVaR-VQE ansatz (RY layer + linear CNOT
chain + ring CNOT), built as an exact-to-cutoff quimb CircuitMPS. Drop-in
replacement for the lightning probs->rng.choice sampling step: draw_indices()
returns a length-`shots` int64 array (basis-state indices, duplicates intact),
seeded deterministically from the per-eval Generator so runs are reproducible.

Bit convention matches PennyLane qml.probs(wires=range(n)): qubit 0 is the MOST
significant bit, so int(sample_string, 2) == the PennyLane basis index.
"""
import numpy as np
import quimb.tensor as qtn

__all__ = ["MPSSampler"]


class MPSSampler:
    def __init__(self, n_qubits, layers, ring=True, max_bond=8192, cutoff=1e-12):
        self.n = int(n_qubits)
        self.layers = int(layers)
        self.ring = bool(ring)
        self.max_bond = max_bond
        self.cutoff = float(cutoff)

    def _circuit(self, params):
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

    def draw_indices(self, params, shots, rng):
        """length-`shots` int64 basis indices, duplicates intact, seeded from rng."""
        circ = self._circuit(params)
        seed = int(rng.integers(0, 2 ** 63 - 1))
        it = circ.sample(int(shots), seed=seed)
        return np.fromiter((int(s, 2) for s in it), dtype=np.int64, count=int(shots))

    def probs(self, params):
        """Full 2^n probability vector (validation / small-n only)."""
        d = np.asarray(self._circuit(params).psi.to_dense()).ravel()
        return np.abs(d) ** 2

    def bond_dim(self, params):
        return int(self._circuit(params).psi.max_bond())
