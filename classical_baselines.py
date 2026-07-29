"""Classical search arms: random search, simulated annealing, exhaustive reference."""
import math
import time
from typing import Dict, Optional

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
                        t_end: float = 1e-3, seed: int = 0,
                        anneal_on: str = "budget") -> Dict:
    """Metropolis annealing over single-slot mutations.

    ``anneal_on`` controls what the temperature schedule is a function of:

    ``"budget"`` (default)
        Fraction of the shared evaluation budget consumed. This is the fix for a real
        defect: ``experiments.run_one`` sets ``n_steps = 20 * eval_budget`` deliberately,
        so that the *budget* is what stops the run. But the schedule used to be a
        function of the step index, so with n_steps = 400,000 and the budget halting the
        run near step ~33,000 the temperature only ever fell to

            4.0 * (1 - 0.083) + 0.001 * 0.083 = 3.67

        i.e. 92% of t_start. The arm never converged -- it was a random walk with a
        best-seen memory, which is why it barely edged out random search (-2.282 against
        -2.153) despite annealing being the stronger algorithm. Scheduling on the
        resource that actually terminates the run makes the arm anneal as intended.

    ``"steps"``
        Legacy behaviour, schedule on step index. Only correct when ``n_steps`` is what
        terminates the run.
    """
    rng = np.random.default_rng(seed)
    rep = hamiltonian.rep
    hamiltonian.reset_counters()
    t0 = time.time()

    is_lattice = getattr(rep, "is_lattice", False)
    n_slots = rep.n_bonds if is_lattice else rep.n_residues
    width = 2 if is_lattice else rep.bits_per_residue
    n_choices = 4 if is_lattice else rep.n_states

    budget = getattr(hamiltonian, "eval_budget", None)
    use_budget = (anneal_on == "budget") and bool(budget)

    current = rep.random_bitstring(rng)
    try:
        cur_e = hamiltonian.energy(current)
    except BudgetExhausted:
        # The budget was already spent before this arm started. Returning a structure
        # with an infinite energy is more honest than raising out of a search driver.
        return {
            "method": "simulated_annealing", "best_bitstring": current,
            "best_energy": float("inf"), "n_energy_evaluations": 0,
            "n_steps": n_steps, "n_objective_evals": 0,
            "terminated_by": "budget", "runtime": time.time() - t0, "seed": seed,
        }
    best, best_e = current, cur_e
    terminated_by = "n_steps"
    n_done = 0
    temp = float(t_start)       # defined even if n_steps == 0, since it is reported

    for k in range(n_steps):
        n_done += 1
        if use_budget:
            frac = min(1.0, hamiltonian.n_energy_evaluations / float(budget))
        else:
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
        "anneal_on": "budget" if use_budget else "steps",
        "final_temperature": float(temp),
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
