"""OPTIONAL external sequence prior. OFF BY DEFAULT.

The default pipeline does not import or use anything here. This module exists
only so that the ablation "does an ESM contact prior change the picture?" can
be run and reported honestly as a SEPARATE, clearly-labelled variant.

WHY IT IS OFF BY DEFAULT
------------------------
A previous design made an ESM-2 contact prior a core energy term. The audit
of that design concluded that the prior was responsible for essentially all
of the measured accuracy, which makes the result an ESM result rather than a
folding-energy or VQE result. Requirement 7 of this project forbids a
pretrained model from directly providing the target structure. Keeping the
prior available but non-default resolves both concerns.

WHAT IT IS NOT
--------------
This never sees native coordinates. It is sequence-only. But a contact map
predicted by a model trained on the PDB is, for a peptide as well-studied as
chignolin, very close to a lookup of the answer. Any result using it must be
reported with that caveat stated.
"""
import math
from typing import Dict, Optional

import numpy as np


class SequencePrior:
    """Contact-probability prior. mode is 'esm' or 'heuristic'."""

    def __init__(self, prefer_esm: bool = True,
                 esm_name: str = "esm2_t6_8M_UR50D"):
        self.mode = "heuristic"
        self._esm = None
        self._batch_converter = None
        self._esm_name = esm_name
        if prefer_esm:
            self._try_load_esm(esm_name)

    def _try_load_esm(self, esm_name: str) -> None:
        try:
            import io
            import contextlib
            import warnings
            warnings.filterwarnings("ignore")
            with contextlib.redirect_stderr(io.StringIO()):
                import torch
                import esm
            model, alphabet = getattr(esm.pretrained, esm_name)()
            model.eval()
            self._esm = (model, alphabet, torch)
            self._batch_converter = alphabet.get_batch_converter()
            self.mode = "esm"
        except Exception as exc:
            print(f"  [prior] ESM unavailable ({type(exc).__name__}); "
                  "falling back to heuristic prior.")
            self._esm = None
            self.mode = "heuristic"

    def contact_probabilities(self, sequence: str) -> np.ndarray:
        if self.mode == "esm":
            try:
                return self._esm_contacts(sequence)
            except Exception:
                self.mode = "heuristic"
        return self._heuristic_contacts(sequence)

    def _esm_contacts(self, sequence: str) -> np.ndarray:
        model, alphabet, torch = self._esm
        _, _, tokens = self._batch_converter([("seq", sequence)])
        with torch.no_grad():
            out = model(tokens, return_contacts=True)
        # .tolist() avoids the torch<->numpy C bridge, which breaks under
        # some torch/numpy version combinations.
        return np.array(out["contacts"][0].detach().cpu().tolist())

    def _heuristic_contacts(self, sequence: str) -> np.ndarray:
        """Transparent hydrophobicity-based prior. Not a neural network."""
        from energy_terms import KD
        n = len(sequence)
        kd = np.array([KD.get(a, 0.0) for a in sequence])
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 3, n):
                hp = (max(kd[i], 0.0) / 4.5) * (max(kd[j], 0.0) / 4.5)
                C[i, j] = C[j, i] = min(1.0, 0.4 * hp)
        return C

    def info(self) -> Dict[str, str]:
        return {
            "mode": self.mode,
            "description": (f"ESM-2 ({self._esm_name}) attention contacts"
                            if self.mode == "esm"
                            else "heuristic hydrophobicity contact prior"),
        }


def contact_violation_energy(contact_prob: np.ndarray, CB: np.ndarray,
                             confidence: float = 0.30, min_sep: int = 4,
                             target: float = 5.5, sigma: float = 2.0) -> float:
    """Penalty for confident predicted contacts left unrealized.

        E = sum_{p_ij >= conf, |i-j| >= min_sep}
                p_ij * (1 - exp(-(d_CB - 5.5)^2 / (2 sigma^2)))

    ABLATION-ONLY TERM. Added to the Hamiltonian only when --prior is set.
    """
    n = len(CB)
    di, dj = np.triu_indices(n, 1)
    sep = dj - di
    d = np.linalg.norm(CB[di] - CB[dj], axis=1)
    p = contact_prob[di, dj]
    conf = (p >= confidence) & (sep >= min_sep)
    if not np.any(conf):
        return 0.0
    satisfaction = np.exp(-((d - target) ** 2) / (2.0 * sigma ** 2))
    return float(np.sum((p * (1.0 - satisfaction))[conf]))