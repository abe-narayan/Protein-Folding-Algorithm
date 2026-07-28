import math
import time
from typing import Dict

import numpy as np

from budget import BudgetExhausted


def random_search(hamiltonian, n_samples: int = 20000, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    rep = hamiltonian.rep
    hamiltonian.reset_counters()
    t0 = time.time()
    best_b, best_e = None, float("inf")
    terminated_by = "n_samples"
    n_done = 0
    for _ in range(n_samples):
        n_done += 1
        b = rep.random_bitstring(rng)
        try:
            e = hamiltonian.energy(b)
        except BudgetExhausted:
            terminated_by = "budget"     # normal termination: return best so far
            break
        if e < best_e:
            best_e, best_b = e, b
    return {
        "method": "random_search",
        "best_bitstring": best_b,
        "best_energy": float(best_e),
        "n_energy_evaluations": hamiltonian.n_energy_evaluations,
        "n_objective_evals": n_done,
        "terminated_by": terminated_by,
        "runtime": time.time() - t0,
        "seed": seed,
    }


def simulated_annealing(hamiltonian, n_steps: int = 20000, t_start: float = 4.0,
                        t_end: float = 1e-3, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    rep = hamiltonian.rep
    hamiltonian.reset_counters()
    t0 = time.time()

    is_lattice = getattr(rep, "is_lattice", False)
    n_slots = rep.n_bonds if is_lattice else rep.n_residues
    width = 2 if is_lattice else rep.bits_per_residue
    n_choices = 4 if is_lattice else rep.n_states

    current = rep.random_bitstring(rng)
    cur_e = hamiltonian.energy(current)
    best, best_e = current, cur_e
    terminated_by = "n_steps"
    n_done = 0

    for k in range(n_steps):
        n_done += 1
        frac = k / max(1, n_steps - 1)
        temp = t_start * (1 - frac) + t_end * frac
        slot = int(rng.integers(0, n_slots))
        off = slot * width
        bits = list(current)
        bits[off:off + width] = list(
            format(int(rng.integers(0, n_choices)), f"0{width}b"))
        cand = "".join(bits)
        try:
            ce = hamiltonian.energy(cand)
        except BudgetExhausted:
            terminated_by = "budget"     # normal termination: return best so far
            break
        if ce < cur_e or rng.random() < math.exp(-(ce - cur_e) / max(temp, 1e-9)):
            current, cur_e = cand, ce
            if ce < best_e:
                best_e, best = ce, cand

    return {
        "method": "simulated_annealing",
        "best_bitstring": best,
        "best_energy": float(best_e),
        "n_energy_evaluations": hamiltonian.n_energy_evaluations,
        "n_steps": n_steps,
        "n_objective_evals": n_done,
        "terminated_by": terminated_by,
        "runtime": time.time() - t0,
        "seed": seed,
    }


def exhaustive_search(hamiltonian, max_bits: int = 22) -> Dict:

    n = hamiltonian.n_bits
    if n > max_bits:
        raise ValueError(
            f"exhaustive_search refuses n_bits={n} (> {max_bits}); "
            f"that is {2**n:.3g} structures")
    hamiltonian.reset_counters()
    # Explicitly exempt from the shared budget: this is the ground-truth reference the
    # arms are measured against, not a competing arm.
    saved_budget = getattr(hamiltonian, "eval_budget", None)
    if hasattr(hamiltonian, "eval_budget"):
        hamiltonian.eval_budget = None
    t0 = time.time()
    fmt = f"0{n}b"
    best_b, best_e = None, float("inf")
    try:
        for idx in range(1 << n):
            b = format(idx, fmt)
            e = hamiltonian.energy(b)
            if e < best_e:
                best_e, best_b = e, b
    finally:
        if hasattr(hamiltonian, "eval_budget"):
            hamiltonian.eval_budget = saved_budget
    return {
        "method": "exhaustive",
        "best_bitstring": best_b,
        "best_energy": float(best_e),
        "n_structures": 1 << n,
        "n_energy_evaluations": hamiltonian.n_energy_evaluations,
        "runtime": time.time() - t0,
    }