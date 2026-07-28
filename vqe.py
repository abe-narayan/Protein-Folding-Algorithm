import math
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from budget import BudgetExhausted, resolve_maxiter, check_optimizer_budget


def cvar_from_samples(energies: Sequence[float], alpha: float) -> float:

    e = np.sort(np.asarray(energies, dtype=float))
    if e.size == 0:
        raise ValueError("cvar_from_samples received no samples")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    keep = max(1, int(math.ceil(alpha * e.size)))
    return float(e[:keep].mean())


def cvar_from_distribution(energies: np.ndarray, probs: np.ndarray,
                           alpha: float) -> float:

    energies = np.asarray(energies, dtype=float)
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, None)
    tot = probs.sum()
    if tot <= 0:
        return float(np.min(energies))
    probs = probs / tot
    order = np.argsort(energies)
    acc = esum = 0.0
    for k in order:
        p = probs[k]
        if acc + p < alpha:
            esum += p * energies[k]
            acc += p
        else:
            esum += (alpha - acc) * energies[k]
            acc = alpha
            break
    return esum / acc if acc > 0 else float(energies[order[0]])



def build_global_circuit(n_qubits: int, layers: int, ring: bool = True,
                         device: str = "lightning.qubit") -> Callable:
    """One circuit over ALL n_qubits. Returns probs over the full register."""
    try:
        dev = qml.device(device, wires=n_qubits)
    except Exception:
        dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        p = np.reshape(np.asarray(params, dtype=float), (layers, n_qubits))
        for l in range(layers):
            for q in range(n_qubits):
                qml.RY(float(p[l][q]), wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
            if ring and n_qubits > 2:
                qml.CNOT(wires=[n_qubits - 1, 0])
        return qml.probs(wires=range(n_qubits))

    return circuit


def n_parameters(n_qubits: int, layers: int) -> int:
    return layers * n_qubits


class BestSeenTracker:
    """Records the lowest-energy bitstring seen anywhere during a run."""

    def __init__(self):
        self.best_energy = float("inf")
        self.best_bitstring: Optional[str] = None
        self.n_lookups = 0

    def offer(self, bitstring: str, energy: float) -> None:
        self.n_lookups += 1
        if energy < self.best_energy:
            self.best_energy = energy
            self.best_bitstring = bitstring


def _run_single(hamiltonian, circuit, n_qubits: int, layers: int,
                alpha: float, shots: int, maxiter: int, seed: int,
                optimizer: str, tracker: BestSeenTracker,
                init_scale: float, verbose: bool) -> Dict:
    fmt = f"0{n_qubits}b"
    n_par = n_parameters(n_qubits, layers)

    init_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC0FFEE]))

    params0 = (math.pi / 2.0) + init_rng.normal(0.0, init_scale, size=n_par)

    history: List[float] = []
    eval_counter = {"n": 0}

    def objective(params: np.ndarray) -> float:
        k = eval_counter["n"]
        eval_counter["n"] = k + 1
        rng = np.random.default_rng(np.random.SeedSequence([seed, k]))

        probs = np.asarray(circuit(params), dtype=float)
        probs = np.clip(probs, 0.0, None)
        s = probs.sum()
        if s <= 0:
            probs = np.full_like(probs, 1.0 / probs.size)
        else:
            probs = probs / s

        idx = rng.choice(probs.size, size=shots, p=probs)


        uniq, inverse = np.unique(idx, return_inverse=True)
        uniq_energies = np.empty(uniq.size, dtype=float)
        for m, u in enumerate(uniq):
            bs = format(int(u), fmt)
            e = hamiltonian.energy(bs)
            uniq_energies[m] = e
            tracker.offer(bs, e)
        sample_energies = uniq_energies[inverse]   # length == shots

        val = cvar_from_samples(sample_energies, alpha)
        history.append(val)
        if verbose and (k % 25 == 0):
            print(f"      eval {k:4d} | CVaR {val:9.3f} | "
                  f"best seen {tracker.best_energy:9.3f}")
        return val

    # Best parameters seen by the objective itself, so an interrupted run still has an
    # answer. Without this, exhausting the budget mid-COBYLA would leave no `res.x`.
    best = {"f": float("inf"), "x": np.array(params0, dtype=float)}
    _raw_objective = objective

    def objective(params):                          # noqa: F811 -- wraps the above
        val = _raw_objective(params)
        if val < best["f"]:
            best["f"], best["x"] = val, np.array(params, dtype=float)
        return val

    t0 = time.time()
    terminated_by = "optimizer"
    try:
        if optimizer.upper() == "COBYLA":
            res = minimize(objective, params0, method="COBYLA",
                           options={"maxiter": maxiter, "rhobeg": 0.4})
        elif optimizer.upper() == "SPSA":
            res = _spsa(objective, params0, maxiter,
                        np.random.default_rng(np.random.SeedSequence([seed, 7])))
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}; use COBYLA or SPSA")
        final_params = np.asarray(res.x if hasattr(res, "x") else res, dtype=float)
        final_objective = (float(res.fun) if hasattr(res, "fun")
                           else float(objective(final_params)))
    except BudgetExhausted:
        # A normal termination condition, not a failure: the arm spent its share of the
        # shared evaluation budget. Return the best point the objective actually saw.
        terminated_by = "budget"
        final_params, final_objective = best["x"], best["f"]
    runtime = time.time() - t0

    return {
        "final_params": final_params,
        "final_objective": final_objective,
        "history": history,
        "n_objective_evals": eval_counter["n"],
        "terminated_by": terminated_by,
        "runtime": runtime,
    }


class _SPSAResult:
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun


def _spsa(objective, x0, n_iter, rng, a=0.25, c=0.15):

    x = np.array(x0, dtype=float)
    A = max(1, n_iter // 10)
    best_x, best_f = x.copy(), objective(x)
    for k in range(n_iter):
        ak = a / ((k + 1 + A) ** 0.602)
        ck = c / ((k + 1) ** 0.101)
        d = rng.choice([-1.0, 1.0], size=x.size)
        fp = objective(x + ck * d)
        fm = objective(x - ck * d)
        x = x - ak * (fp - fm) / (2.0 * ck) * d
        fx = objective(x)
        if fx < best_f:
            best_f, best_x = fx, x.copy()
    return _SPSAResult(best_x, best_f)


def run_global_cvar_vqe(hamiltonian, layers: int = 4, alpha: float = 0.15,
                        shots: int = 2048, maxiter: Optional[int] = None,
                        restarts: int = 4, seed: int = 0,
                        optimizer: str = "COBYLA", ring: bool = True,
                        final_shots: int = 8192, init_scale: float = 0.6,
                        device: str = "lightning.qubit",
                        readout_reserve_frac: float = 0.01,
                        verbose: bool = False) -> Dict:
    """Global CVaR-VQE over the entire protein configuration register.

    Returns a dict distinguishing best-seen from the final VQE solution.

    Cost is governed by ``hamiltonian.eval_budget`` -- the number of *unique* structures
    whose energy had to be computed -- not by ``maxiter``. Every search method is handed
    the same budget through the Hamiltonian it shares, so whichever finds the lowest
    energy inside that budget wins and no method can buy a better answer with a larger
    optimizer allowance. ``maxiter=None`` resolves deliberately high so the shared budget
    is what binds.
    """
    n_qubits = hamiltonian.n_qubits
    if n_qubits > 30:
        raise MemoryError(
            f"n_qubits={n_qubits} requires ~{2**n_qubits * 16 / 1e9:.0f} GB "
            "for a statevector. A genuine full-system VQE is not simulable "
            "at this size. Reduce protein length or state count.")

    n_par = n_parameters(n_qubits, layers)
    maxiter = resolve_maxiter(maxiter, n_par)
    check_optimizer_budget(maxiter, n_par, optimizer,
                           getattr(hamiltonian, "eval_budget", None))

    circuit = build_global_circuit(n_qubits, layers, ring=ring, device=device)
    tracker = BestSeenTracker()

    hamiltonian.reset_counters()

    # Withhold a slice for the final read-out so a run can always report its answer
    # after the optimizer has spent everything. The read-out scans most-probable-first
    # and those states are nearly all cache hits after a converged search, so this needs
    # to be small -- reserving 10% measurably penalised the VQE's search relative to the
    # classical arms, which is precisely the unfairness the shared budget exists to remove.
    budget = getattr(hamiltonian, "eval_budget", None)
    readout_reserve = 0 if budget is None else max(1, int(budget * readout_reserve_frac))

    restart_seeds = [int(s.generate_state(1)[0])
                     for s in np.random.SeedSequence(seed).spawn(restarts)]

    t0 = time.time()
    runs = []
    best_run = None
    restarts_completed = 0
    for r, rseed in enumerate(restart_seeds):
        if verbose:
            print(f"    --- restart {r + 1}/{restarts} (seed {rseed}) ---")
        if budget is not None:
            # Slice the pool cumulatively: restart r may spend up to its equal share plus
            # anything earlier restarts left unspent. Without this, restart 1 consumes the
            # whole pool and a 4-restart algorithm silently becomes a 1-restart one.
            searchable = budget - readout_reserve
            cum_limit = int(round(searchable * (r + 1) / restarts))
            hamiltonian.reserve(budget - cum_limit)
            if hamiltonian.budget_remaining <= 0:
                if verbose:
                    print(f"    (budget exhausted; {restarts - r} restarts not run)")
                break
        run = _run_single(hamiltonian, circuit, n_qubits, layers, alpha,
                          shots, maxiter, rseed, optimizer, tracker,
                          init_scale, verbose)
        runs.append(run)
        restarts_completed += 1
        if best_run is None or run["final_objective"] < best_run["final_objective"]:
            best_run = run
    total_runtime = time.time() - t0

    if best_run is None:
        raise BudgetExhausted(budget or 0, hamiltonian.n_energy_evaluations)

    hamiltonian.release()               # hand the reserve back for the read-out

    fmt = f"0{n_qubits}b"
    final_probs = np.asarray(circuit(best_run["final_params"]), dtype=float)
    final_probs = np.clip(final_probs, 0.0, None)
    final_probs = final_probs / final_probs.sum()

    final_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xF1A1]))
    final_idx = final_rng.choice(final_probs.size, size=final_shots,
                                 p=final_probs)
    # Scan most-probable-first. np.unique returns basis indices in *sorted* order, so if
    # the budget truncates the read-out a plain unique() scan keeps an arbitrary
    # low-index subset rather than the states the circuit actually favours.
    uniq, counts = np.unique(final_idx, return_counts=True)
    uniq = uniq[np.argsort(-counts, kind="stable")]
    vqe_bits, vqe_energy = None, float("inf")
    for u in uniq:
        bs = format(int(u), fmt)
        try:
            e = hamiltonian.energy(bs)
        except BudgetExhausted:
            break
        if e < vqe_energy:
            vqe_energy, vqe_bits = e, bs

    modal_bits = format(int(np.argmax(final_probs)), fmt)
    try:
        modal_energy = hamiltonian.energy(modal_bits)
    except BudgetExhausted:
        modal_energy = float("nan")

    p_sorted = np.sort(final_probs)[::-1]
    top1 = float(p_sorted[0])
    top16 = float(p_sorted[:16].sum())
    nz = final_probs[final_probs > 1e-15]
    entropy = float(-np.sum(nz * np.log2(nz)))

    return {
        "vqe_bitstring": vqe_bits,
        "vqe_energy": float(vqe_energy),
        "vqe_modal_bitstring": modal_bits,
        "vqe_modal_energy": float(modal_energy),
        "best_seen_bitstring": tracker.best_bitstring,
        "best_seen_energy": float(tracker.best_energy),
        "final_objective": best_run["final_objective"],
        "history": best_run["history"],
        "distribution_top1_prob": top1,
        "distribution_top16_mass": top16,
        "distribution_entropy_bits": entropy,
        "max_entropy_bits": float(n_qubits),
        "n_qubits": n_qubits,
        "n_parameters": n_parameters(n_qubits, layers),
        "layers": layers,
        "alpha": alpha,
        "shots_per_eval": shots,
        "final_shots": final_shots,
        "restarts": restarts,
        "optimizer": optimizer,
        "n_objective_evals_total": sum(r["n_objective_evals"] for r in runs),
        "n_objective_evals_best_run": best_run["n_objective_evals"],
        "n_energy_evaluations": hamiltonian.n_energy_evaluations,
        "n_unique_structures_cached": hamiltonian.cache_size(),
        "maxiter_resolved": maxiter,
        "maxiter_over_n_params": maxiter / max(1, n_par),
        "eval_budget": budget,
        "budget_exhausted": bool(getattr(hamiltonian, "budget_exhausted", False)),
        "terminated_by": ("budget" if any(r["terminated_by"] == "budget" for r in runs)
                          else "optimizer"),
        "restarts_completed": restarts_completed,
        "runtime": total_runtime,
        "seed": seed,
    }