"""Classical search arms: random, annealing, distribution search, exhaustive reference."""
import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from budget import BudgetExhausted


def _mutable_fields(rep) -> List[Tuple[int, int, int]]:
    """Return every independently encoded field as ``(offset, width, choices)``.

    Aromatic chi1 bits are appended after the backbone fields.  The former annealer only
    addressed ``n_residues * bits_per_residue`` and therefore froze those bits at their
    random initial values.  Keeping field discovery here gives every classical method the
    same search space as VQE and ``rep.random_bitstring``.
    """
    if getattr(rep, "is_lattice", False):
        return [(2 * i, 2, 4) for i in range(rep.n_bonds)]
    fields = [(i * rep.bits_per_residue, rep.bits_per_residue, rep.n_states)
              for i in range(rep.n_residues)]
    chi0 = getattr(rep, "n_backbone_bits",
                   rep.n_residues * rep.bits_per_residue)
    fields.extend((chi0 + i, 1, 2)
                  for i in range(getattr(rep, "n_chi_bits", 0)))
    return fields


def _trace_point(trace: List[Dict], hamiltonian, best_b: Optional[str],
                 best_e: float, started: float, method_step: int,
                 restart: int = 0, cvar: Optional[float] = None) -> None:
    trace.append({
        "n_energy_evaluations": int(hamiltonian.n_energy_evaluations),
        "best_energy": float(best_e),
        "best_bitstring": best_b,
        "elapsed_s": float(time.time() - started),
        "method_step": int(method_step),
        "restart": int(restart),
        "cvar": cvar,
    })


def random_search(hamiltonian, n_samples: int = 20000, seed: int = 0,
                  trace_interval: int = 250, batch_size: int = 64,
                  energy_batch: Optional[Callable] = None) -> Dict:
    rng = np.random.default_rng(seed)
    rep = hamiltonian.rep
    hamiltonian.reset_counters()
    t0 = time.time()
    best_b, best_e = None, float("inf")
    trace: List[Dict] = []
    next_trace = max(1, int(trace_interval))
    terminated_by = "n_samples"
    n_done = 0
    while n_done < n_samples:
        m = min(int(batch_size), n_samples - n_done) if energy_batch else 1
        bitstrings = [rep.random_bitstring(rng) for _ in range(m)]
        emap = energy_batch(bitstrings) if energy_batch is not None else None
        for b in bitstrings:
            n_done += 1
            try:
                e = (emap[b] if emap is not None and b in emap
                     else hamiltonian.energy(b))
            except BudgetExhausted:
                terminated_by = "budget"     # normal termination: return best so far
                break
            if e < best_e:
                best_e, best_b = e, b
            if hamiltonian.n_energy_evaluations >= next_trace:
                _trace_point(trace, hamiltonian, best_b, best_e, t0, n_done)
                next_trace += max(1, int(trace_interval))
        if terminated_by == "budget":
            break
    if not trace or trace[-1]["n_energy_evaluations"] != hamiltonian.n_energy_evaluations:
        _trace_point(trace, hamiltonian, best_b, best_e, t0, n_done)
    return {
        "method": "random_search",
        "best_bitstring": best_b,
        "best_energy": float(best_e),
        "n_energy_evaluations": hamiltonian.n_energy_evaluations,
        "n_objective_evals": n_done,
        "terminated_by": terminated_by,
        "runtime": time.time() - t0,
        "trace": trace,
        "seed": seed,
    }


def simulated_annealing(hamiltonian, n_steps: int = 20000, t_start: float = 4.0,
                        t_end: float = 1e-3, seed: int = 0,
                        anneal_on: str = "budget", restarts: int = 1,
                        trace_interval: int = 250) -> Dict:
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
    rep = hamiltonian.rep
    hamiltonian.reset_counters()
    t0 = time.time()
    fields = _mutable_fields(rep)
    if not fields:
        raise ValueError("representation has no mutable fields")
    restarts = max(1, int(restarts))

    budget = getattr(hamiltonian, "eval_budget", None)
    use_budget = (anneal_on == "budget") and bool(budget)

    best, best_e = None, float("inf")
    final, final_e = None, float("inf")
    terminated_by = "n_steps"
    n_done = 0
    temp = float(t_start)
    trace: List[Dict] = []
    next_trace = max(1, int(trace_interval))
    seeds = ([seed] if restarts == 1 else
             [int(s.generate_state(1)[0])
              for s in np.random.SeedSequence(seed).spawn(restarts)])
    completed = 0

    for restart, rseed in enumerate(seeds):
        if budget:
            cumulative_limit = int(round(budget * (restart + 1) / restarts))
            hamiltonian.reserve(budget - cumulative_limit)
        restart_start = hamiltonian.n_energy_evaluations
        restart_span = (max(1, cumulative_limit - restart_start)
                        if budget else max(1, n_steps // restarts))
        rng = np.random.default_rng(rseed)
        current = rep.random_bitstring(rng)
        try:
            cur_e = hamiltonian.energy(current)
        except BudgetExhausted:
            break
        if cur_e < best_e:
            best, best_e = current, cur_e

        steps_this_restart = max(1, int(math.ceil(n_steps / restarts)))
        for k in range(steps_this_restart):
            n_done += 1
            if use_budget:
                frac = min(1.0, (hamiltonian.n_energy_evaluations - restart_start)
                           / float(restart_span))
            else:
                frac = k / max(1, steps_this_restart - 1)
            temp = t_start * (1 - frac) + t_end * frac
            off, width, n_choices = fields[int(rng.integers(0, len(fields)))]
            bits = list(current)
            # Preserve the historical proposal stream: the sampled value may equal the
            # current one, in which case the cache hit is free. The only search-space
            # change is that appended chi1 fields are now eligible too.
            new = int(rng.integers(0, n_choices))
            bits[off:off + width] = list(format(new, f"0{width}b"))
            cand = "".join(bits)
            try:
                ce = hamiltonian.energy(cand)
            except BudgetExhausted:
                terminated_by = "budget"
                break
            if ce < cur_e or rng.random() < math.exp(-(ce - cur_e) / max(temp, 1e-9)):
                current, cur_e = cand, ce
                if ce < best_e:
                    best_e, best = ce, cand
            if hamiltonian.n_energy_evaluations >= next_trace:
                _trace_point(trace, hamiltonian, best, best_e, t0, n_done, restart)
                next_trace += max(1, int(trace_interval))
        final, final_e = current, cur_e
        completed += 1
        if terminated_by == "budget" and restart + 1 == restarts:
            break

    hamiltonian.release()
    if not trace or trace[-1]["n_energy_evaluations"] != hamiltonian.n_energy_evaluations:
        _trace_point(trace, hamiltonian, best, best_e, t0, n_done,
                     max(0, completed - 1))

    return {
        "method": "simulated_annealing",
        "best_bitstring": best,
        "best_energy": float(best_e),
        "final_bitstring": final,
        "final_energy": float(final_e),
        "n_energy_evaluations": hamiltonian.n_energy_evaluations,
        "n_steps": n_steps,
        "n_objective_evals": n_done,
        "anneal_on": "budget" if use_budget else "steps",
        "final_temperature": float(temp),
        "restarts_completed": completed,
        "terminated_by": terminated_by,
        "runtime": time.time() - t0,
        "trace": trace,
        "seed": seed,
    }


def cross_entropy_search(hamiltonian, n_samples: int = 200000,
                         batch_size: int = 64, elite_frac: float = 0.20,
                         learning_rate: float = 0.50, min_prob: float = 0.02,
                         seed: int = 0, trace_interval: int = 250,
                         energy_batch: Optional[Callable] = None) -> Dict:
    """Classical factorized distribution optimizer under the shared energy budget.

    This is the classical control closest to CVaR-VQE: both draw a batch of global
    bitstrings, keep a low-energy tail, and update a probability distribution.  A VQE
    advantage over random search or a local walk alone is not enough to distinguish the
    circuit from ordinary classical probability adaptation.
    """
    if not (0.0 < elite_frac <= 1.0):
        raise ValueError("elite_frac must be in (0, 1]")
    if not (0.0 < learning_rate <= 1.0):
        raise ValueError("learning_rate must be in (0, 1]")
    rng = np.random.default_rng(seed)
    rep = hamiltonian.rep
    hamiltonian.reset_counters()
    t0 = time.time()
    probs = np.full(rep.n_bits, 0.5, dtype=float)
    best_b, best_e = None, float("inf")
    trace: List[Dict] = []
    next_trace = max(1, int(trace_interval))
    n_drawn = n_batches = 0
    terminated_by = "n_samples"

    while n_drawn < n_samples:
        m = min(int(batch_size), n_samples - n_drawn)
        samples = rng.random((m, rep.n_bits)) < probs
        bitstrings = ["".join("1" if x else "0" for x in row) for row in samples]
        emap = energy_batch(bitstrings) if energy_batch is not None else None
        energies, kept_rows = [], []
        for row, bs in zip(samples, bitstrings):
            try:
                e = (emap[bs] if emap is not None and bs in emap
                     else hamiltonian.energy(bs))
            except BudgetExhausted:
                terminated_by = "budget"
                break
            energies.append(e)
            kept_rows.append(row)
            n_drawn += 1
            if e < best_e:
                best_b, best_e = bs, e
            if hamiltonian.n_energy_evaluations >= next_trace:
                _trace_point(trace, hamiltonian, best_b, best_e, t0, n_batches)
                next_trace += max(1, int(trace_interval))
        if not energies:
            break
        n_batches += 1
        earr = np.asarray(energies, dtype=float)
        finite = np.isfinite(earr)
        if np.any(finite):
            order = np.where(finite)[0][np.argsort(earr[finite], kind="stable")]
            k = max(1, int(math.ceil(elite_frac * len(order))))
            elite = np.asarray(kept_rows, dtype=float)[order[:k]]
            target = elite.mean(axis=0)
            probs = (1.0 - learning_rate) * probs + learning_rate * target
            probs = np.clip(probs, min_prob, 1.0 - min_prob)
        if terminated_by == "budget":
            break

    if not trace or trace[-1]["n_energy_evaluations"] != hamiltonian.n_energy_evaluations:
        _trace_point(trace, hamiltonian, best_b, best_e, t0, n_batches)
    entropy = -np.sum(probs * np.log2(probs) + (1 - probs) * np.log2(1 - probs))
    return {
        "method": "cross_entropy", "best_bitstring": best_b,
        "best_energy": float(best_e),
        "n_energy_evaluations": hamiltonian.n_energy_evaluations,
        "n_objective_evals": n_batches, "n_samples": n_drawn,
        "terminated_by": terminated_by, "runtime": time.time() - t0,
        "distribution_entropy_bits": float(entropy),
        "distribution_top1_prob": float(np.prod(np.maximum(probs, 1.0 - probs))),
        "trace": trace, "seed": seed,
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
