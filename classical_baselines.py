
import math
import time
from typing import Dict, Optional

import numpy as np


def random_search(hamiltonian, n_samples: int = 20000, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    rep = hamiltonian.rep
    hamiltonian.reset_counters()
    t0 = time.time()
    best_b, best_e = None, float("inf")
    for _ in range(n_samples):
        b = rep.random_bitstring(rng)
        e = hamiltonian.energy(b)
        if e < best_e:
            best_e, best_b = e, b
    return {
        "method": "random_search",
        "best_bitstring": best_b,
        "best_energy": float(best_e),
        "n_energy_evaluations": hamiltonian.n_energy_evaluations,
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

    for k in range(n_steps):
        frac = k / max(1, n_steps - 1)
        temp = t_start * (1 - frac) + t_end * frac
        slot = int(rng.integers(0, n_slots))
        off = slot * width
        bits = list(current)
        bits[off:off + width] = list(
            format(int(rng.integers(0, n_choices)), f"0{width}b"))
        cand = "".join(bits)
        ce = hamiltonian.energy(cand)
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
    t0 = time.time()
    fmt = f"0{n}b"
    best_b, best_e = None, float("inf")
    for idx in range(1 << n):
        b = format(idx, fmt)
        e = hamiltonian.energy(b)
        if e < best_e:
            best_e, best_b = e, b
    return {
        "method": "exhaustive",
        "best_bitstring": best_b,
        "best_energy": float(best_e),
        "n_structures": 1 << n,
        "n_energy_evaluations": hamiltonian.n_energy_evaluations,
        "runtime": time.time() - t0,
    }