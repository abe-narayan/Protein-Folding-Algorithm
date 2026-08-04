"""Exact direct sampler for the project's one-layer RY-CNOT VQE ansatz.

For one layer, the RY gates prepare independent Bernoulli variables and the CNOT
chain/ring only applies an invertible XOR transformation to computational-basis states.
Sampling therefore costs O(shots * qubits) and does not require constructing the full
2**n statevector or probability vector.  This changes simulator runtime only; it samples
exactly the same ideal circuit distribution.
"""
from __future__ import annotations

import numpy as np


class OneLayerDirectSampler:
    """Sample the exact one-layer circuit distribution without a statevector."""

    backend_name = "direct-one-layer"

    def __init__(self, n_qubits: int, layers: int = 1, ring: bool = True):
        self.n_qubits = int(n_qubits)
        self.layers = int(layers)
        self.ring = bool(ring)
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if self.layers != 1:
            raise ValueError("OneLayerDirectSampler is exact only for layers=1")
        self._weights = 1 << np.arange(
            self.n_qubits - 1, -1, -1, dtype=np.int64)

    def _apply_cnot_network(self, bits: np.ndarray) -> np.ndarray:
        out = np.array(bits, dtype=np.uint8, copy=True)
        for q in range(self.n_qubits - 1):
            out[:, q + 1] ^= out[:, q]
        if self.ring and self.n_qubits > 2:
            out[:, 0] ^= out[:, -1]
        return out

    def draw_indices(self, params, shots: int,
                     rng: np.random.Generator) -> np.ndarray:
        angles = np.asarray(params, dtype=float).reshape(self.n_qubits)
        probability_one = np.sin(angles / 2.0) ** 2
        initial = (rng.random((int(shots), self.n_qubits))
                   < probability_one).astype(np.uint8)
        measured = self._apply_cnot_network(initial)
        return measured.astype(np.int64) @ self._weights

    def probs(self, params) -> np.ndarray:
        """Return an exact dense probability vector for small validation systems."""
        if self.n_qubits > 20:
            raise MemoryError("dense validation probabilities are limited to 20 qubits")
        angles = np.asarray(params, dtype=float).reshape(self.n_qubits)
        p1 = np.sin(angles / 2.0) ** 2
        indices = np.arange(1 << self.n_qubits, dtype=np.int64)
        shifts = np.arange(self.n_qubits - 1, -1, -1, dtype=np.int64)
        initial = ((indices[:, None] >> shifts) & 1).astype(np.uint8)
        probabilities = np.prod(np.where(initial == 1, p1, 1.0 - p1), axis=1)
        measured = self._apply_cnot_network(initial)
        output_indices = measured.astype(np.int64) @ self._weights
        output = np.empty_like(probabilities)
        output[output_indices] = probabilities
        return output
