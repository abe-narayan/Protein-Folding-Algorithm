"""Matrix-product-state backend for the global CVaR-VQE.

A coherent home for what used to live as one-off ``scratch/`` scripts:

* :class:`~mps.sampler.MPSSampler` — exact-to-cutoff quimb MPS sampler for the
  RY + CNOT-chain + ring ansatz.
* :func:`~mps.driver.run_global_cvar_vqe` — the CVaR-VQE driver with a pluggable
  sampler (lightning or MPS) plus the collapse audit.
"""
from mps.sampler import MPSSampler
from mps.driver import run_global_cvar_vqe

__all__ = ["MPSSampler", "run_global_cvar_vqe"]

