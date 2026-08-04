import math
import time
import warnings
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
                init_scale: float, verbose: bool, sampler=None,
                energy_batch: Optional[Callable] = None,
                spsa_a: float = 0.25, spsa_c: float = 0.15,
                trace_interval: int = 250) -> Dict:
    fmt = f"0{n_qubits}b"
    n_par = n_parameters(n_qubits, layers)

    init_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC0FFEE]))

    params0 = (math.pi / 2.0) + init_rng.normal(0.0, init_scale, size=n_par)

    history: List[float] = []
    trace: List[Dict] = []
    interval = max(1, int(trace_interval))
    spent_at_start = int(hamiltonian.n_energy_evaluations)
    next_trace = {"n": ((spent_at_start // interval) + 1) * interval}
    eval_counter = {"n": 0}
    # Which bitstring-sample stream the next objective call draws from. `None` means "a
    # fresh stream per call", which is what a deterministic optimizer like COBYLA gets.
    # SPSA pins it so both arms of its finite difference share one stream -- see `_spsa`.
    sample_tag = {"v": None}

    def objective(params: np.ndarray) -> float:
        k = eval_counter["n"]
        eval_counter["n"] = k + 1
        tag = k if sample_tag["v"] is None else sample_tag["v"]
        rng = np.random.default_rng(np.random.SeedSequence([seed, tag]))

        if sampler is None:
            probs = np.asarray(circuit(params), dtype=float)
            probs = np.clip(probs, 0.0, None)
            s = probs.sum()
            if s <= 0:
                probs = np.full_like(probs, 1.0 / probs.size)
            else:
                probs = probs / s
            idx = rng.choice(probs.size, size=shots, p=probs)
        else:
            idx = sampler.draw_indices(params, shots, rng)

        uniq, inverse = np.unique(idx, return_inverse=True)
        bslist = [format(int(u), fmt) for u in uniq]
        emap = energy_batch(bslist) if energy_batch is not None else None
        uniq_energies = np.empty(uniq.size, dtype=float)
        for m, bs in enumerate(bslist):
            # A budget-aware batch may return only the prefix that fit. Falling back to
            # .energy() for a missing entry preserves normal BudgetExhausted termination.
            e = (emap[bs] if emap is not None and bs in emap
                 else hamiltonian.energy(bs))
            uniq_energies[m] = e
            tracker.offer(bs, e)
        sample_energies = uniq_energies[inverse]   # length == shots

        val = cvar_from_samples(sample_energies, alpha)
        history.append(val)
        if hamiltonian.n_energy_evaluations >= next_trace["n"]:
            trace.append({
                "n_energy_evaluations": int(hamiltonian.n_energy_evaluations),
                "objective_eval": int(k + 1),
                "cvar": float(val),
                "best_energy": float(tracker.best_energy),
                "best_bitstring": tracker.best_bitstring,
                "elapsed_s": float(time.time() - t0),
            })
            next_trace["n"] += interval
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

    # Fraction of THIS restart's budget slice consumed, for SPSA's step schedule. Captured
    # at entry because `n_energy_evaluations` is cumulative across restarts.
    _span = max(1.0, float(hamiltonian.budget_remaining)) \
        if getattr(hamiltonian, "eval_budget", None) else None

    def progress() -> Optional[float]:
        if _span is None:
            return None
        return min(1.0, max(0.0, 1.0 - float(hamiltonian.budget_remaining) / _span))

    t0 = time.time()
    terminated_by = "optimizer"
    try:
        if optimizer.upper() == "COBYLA":
            res = minimize(objective, params0, method="COBYLA",
                           options={"maxiter": maxiter, "rhobeg": 0.4})
        elif optimizer.upper() == "SPSA":
            res = _spsa(objective, params0, maxiter,
                        np.random.default_rng(np.random.SeedSequence([seed, 7])),
                        set_sample_tag=lambda t: sample_tag.__setitem__("v", t),
                        progress=progress, a=spsa_a, c=spsa_c)
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
        "trace": trace,
        "n_objective_evals": eval_counter["n"],
        "terminated_by": terminated_by,
        "runtime": runtime,
    }


class _SPSAResult:
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun


def _spsa(objective, x0, n_iter, rng, a=0.25, c=0.15,
          set_sample_tag=None, progress=None):
    """Simultaneous Perturbation Stochastic Approximation (Spall).

    This is the right family of optimizer for this objective, and the reason is structural
    rather than empirical. The CVaR objective is a *stochastic* estimate: it draws `shots`
    bitstrings from the circuit's distribution, so evaluating the same parameters twice
    returns different values. COBYLA is a derivative-free **trust-region** method that
    builds a linear model from n+1 points and shrinks its region when predicted and actual
    improvement disagree -- which sampling noise guarantees at every step. It requires a
    reproducible objective and does not have one here. SPSA's convergence theory is *for*
    noisy evaluations, and its cost per iteration is 2 evaluations regardless of dimension,
    against COBYLA's n+1 just to build an initial simplex (89 evaluations at 88 parameters).

    Three fixes over the previous implementation, all of which mattered:

    1. **Progress-driven schedule.** `A` and the decay exponents were calibrated against
       `n_iter`, but `experiments.run_one` deliberately sets `maxiter` far above what the
       shared evaluation budget affords, so the budget terminates the run long before `k`
       approaches `n_iter`. With `n_iter = 50 x n_params = 4000` that made `A = 400` and a
       first step of `a / 401**0.602 = 0.007` -- five times too small -- and the decay never
       completed. SPSA crawled. This is the same defect that made simulated annealing never
       cool: a schedule indexed by a horizon the run never reaches. `progress()` returns the
       fraction of the budget slice consumed, and the schedule is traversed against that.
    2. **Common random numbers.** Both arms of the finite difference are now evaluated
       against the *same* bitstring sample stream, so the sampling noise largely cancels in
       `fp - fm`. With independent draws the difference carries twice the noise variance of
       a single evaluation -- the worst case for a finite-difference gradient.
    3. **Two evaluations per iteration, not three (plus one).** The old version spent a
       third call at the updated `x` purely for best-tracking. The midpoint `(fp + fm) / 2`
       is an unbiased estimate of `f(x)` with *lower* variance than a fresh evaluation, so
       that call was pure waste: 1 + 3n evaluations became 2n, a 1.5x gain in iterations at
       equal cost.

    `a` and `c` are Spall's gain constants and are now the main tuning knobs; `--hparams`
    sweeps the optimizer choice, and they should be swept with it rather than guessed.
    """
    x = np.array(x0, dtype=float)
    A = max(1, n_iter // 10)
    best_x, best_f = x.copy(), float("inf")
    for k in range(n_iter):
        # Effective index along the intended schedule. Budget-driven when a budget exists,
        # so the decay completes exactly as the slice runs out; otherwise the raw index.
        p = None if progress is None else progress()
        kk = k if p is None else p * n_iter
        ak = a / ((kk + 1.0 + A) ** 0.602)
        ck = c / ((kk + 1.0) ** 0.101)

        d = rng.choice([-1.0, 1.0], size=x.size)
        if set_sample_tag is not None:
            set_sample_tag(k)               # common random numbers across the +/- pair
        fp = objective(x + ck * d)
        fm = objective(x - ck * d)

        f_mid = 0.5 * (fp + fm)
        if f_mid < best_f:
            best_f, best_x = f_mid, x.copy()

        x = x - ak * (fp - fm) / (2.0 * ck) * d
    if set_sample_tag is not None:
        set_sample_tag(None)                # restore per-call streams
    return _SPSAResult(best_x, best_f)


#: Fewest SPSA iterations at which the optimiser has been observed to do useful work on
#: this objective. Below this the run returns something close to its initialisation.
#: Measured: the shipped configuration (shots 2048, budget 20000, restarts 4) affords
#: about 33, and returned a bit-identical answer across 3 seeds and 5 hyperparameter
#: settings -- the signature of an optimiser that never moved.
MIN_USEFUL_SPSA_ITERS = 100


def _warn_if_underoptimised(hamiltonian, shots: int, restarts: int,
                            optimizer: str, verbose: bool) -> Optional[str]:
    """Warn when the shared budget cannot afford enough optimiser steps.

    `hamiltonian.energy` charges per UNIQUE bitstring, so while the circuit distribution is
    still broad, one objective call costs ~`shots` of the budget. The affordable number of
    SPSA iterations is therefore about `budget / (shots * restarts * 2)` early on -- it
    improves as the distribution concentrates and samples start hitting the cache, but the
    early phase is exactly when the optimiser needs to move.

    This is a warning rather than an error because a small run (validation uses shots=256,
    maxiter=120) is legitimately allowed to be under-optimised; what is not acceptable is
    for a headline comparison to be under-optimised without saying so.
    """
    budget = getattr(hamiltonian, "eval_budget", None)
    if not budget or optimizer.upper() != "SPSA":
        return None
    affordable = budget / max(1, shots * restarts * 2)
    if affordable >= MIN_USEFUL_SPSA_ITERS:
        return None
    msg = (f"VQE is budget-starved: shots={shots} x restarts={restarts} against a shared "
           f"budget of {budget} affords about {affordable:.0f} SPSA iterations early on, "
           f"below the {MIN_USEFUL_SPSA_ITERS} at which this objective has been seen to "
           f"make progress. The result will be close to the initialisation and will look "
           f"deceptively stable across seeds. Reduce shots (try "
           f"{max(32, int(budget / (MIN_USEFUL_SPSA_ITERS * restarts * 2)))}), reduce "
           f"restarts, or raise the budget.")
    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    if verbose:
        print(f"    !! {msg}")
    return msg


def run_global_cvar_vqe(hamiltonian, layers: int = 4, alpha: float = 0.15,
                        shots: int = 2048, maxiter: Optional[int] = None,
                        restarts: int = 4, seed: int = 0,
                        optimizer: str = "SPSA", ring: bool = True,
                        final_shots: int = 8192, init_scale: float = 1.0,
                        device: str = "lightning.qubit",
                        readout_reserve_frac: float = 0.01,
                        sampler=None, energy_batch: Optional[Callable] = None,
                        spsa_a: float = 0.25, spsa_c: float = 0.15,
                        trace_interval: int = 250,
                        verbose: bool = False) -> Dict:
    """Global CVaR-VQE over the entire protein configuration register.

    Returns a dict distinguishing best-seen from the final VQE solution.

    Cost is governed by ``hamiltonian.eval_budget`` -- the number of *unique* structures
    whose energy had to be computed -- not by ``maxiter``. Every search method is handed
    the same budget through the Hamiltonian it shares, so whichever finds the lowest
    energy inside that budget wins and no method can buy a better answer with a larger
    optimizer allowance. ``maxiter=None`` resolves deliberately high so the shared budget
    is what binds.

    ``optimizer`` defaults to SPSA because this objective is a *stochastic* estimate: it
    samples ``shots`` bitstrings per call, so the same parameters return different values.
    COBYLA is a derivative-free trust-region method that requires a reproducible objective
    and shrinks its region whenever predicted and actual improvement disagree -- which
    sampling noise causes at every step. See ``_spsa`` for the full argument and for the
    three defects fixed in the SPSA implementation. COBYLA remains selectable so the choice
    is measurable; ``experiments.experiment_vqe_hyperparameters`` runs both.

    Note ``maxiter`` means different things to the two: function evaluations for COBYLA,
    iterations (two evaluations each) for SPSA. ``budget.check_optimizer_budget`` guards
    both, with an absolute floor for SPSA since its per-iteration cost does not grow with
    the parameter count.

    ``init_scale`` and the shots/budget ratio are LOAD-BEARING TOGETHER, and this is the
    single most consequential thing to know about this function. Measured on 1UAO at 400
    SPSA iterations, changing nothing but the initial spread:

        init_scale 0.25   ->  E -11.018   CA-RMSD 5.27 A   (the alpha-helix)
        init_scale 1.00   ->  E -11.350   CA-RMSD 1.96 A   (the global minimum's backbone)

    Neither variable does anything on its own, which is why this went undiagnosed:

    * Wide init with too few iterations never concentrates -- the distribution is still
      near-uniform (entropy 17-20 bits of a possible 22) when the run ends, i.e. random
      sampling with extra steps.
    * Enough iterations from a narrow init converges reliably into the SAME basin every
      time. At ``init_scale=0.25`` every parameter starts within 0.25 rad of pi/2, so all
      restarts and all seeds begin at effectively one point and reach one answer. That is
      why 5 hyperparameter settings (alpha 0.05/0.15/0.35, layers 4/8, restarts 4/8) and
      3 seeds all returned a bit-identical -11.018, and why 12x more optimizer iterations
      changed nothing.

    The shots/budget interaction is the other half. ``hamiltonian.energy`` charges the
    budget per UNIQUE bitstring and serves repeats from cache for free, so while the
    distribution is flat, ``shots=2048`` draws ~2048 distinct structures and burns ~2048
    of the shared budget on a SINGLE objective call. At the default 20000 that is ~10
    calls before caching starts paying off -- around 67 objective calls in the best
    restart, roughly 33 SPSA iterations to fit 88 parameters. The failure is silent: the
    run reports full budget usage, respects cost matching, and returns a plausible
    structure. ``_warn_if_underoptimised`` below exists so it is never silent again.

    ``sampler`` selects an alternative exact/approximate sampler without forking the
    optimizer logic. The AMBER study uses the distribution-exact direct path for the
    one-layer RY-CNOT ansatz and MPS for supported deeper cases.
    ``energy_batch`` may evaluate the independent unique structures in one objective call
    concurrently; the Hamiltonian remains the sole owner of the shared hard budget.
    """
    _warn_if_underoptimised(hamiltonian, shots, restarts, optimizer, verbose)

    n_qubits = hamiltonian.n_qubits
    if sampler is None and n_qubits > 30:
        raise MemoryError(
            f"n_qubits={n_qubits} requires ~{2**n_qubits * 16 / 1e9:.0f} GB "
            "for a statevector. A genuine full-system VQE is not simulable "
            "at this size. Reduce protein length or state count.")

    n_par = n_parameters(n_qubits, layers)
    maxiter = resolve_maxiter(maxiter, n_par)
    check_optimizer_budget(maxiter, n_par, optimizer,
                           getattr(hamiltonian, "eval_budget", None))

    circuit = (None if sampler is not None else
               build_global_circuit(n_qubits, layers, ring=ring, device=device))
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
                          init_scale, verbose, sampler=sampler,
                          energy_batch=energy_batch, spsa_a=spsa_a,
                          spsa_c=spsa_c, trace_interval=trace_interval)
        if (not run["trace"] or
                run["trace"][-1]["n_energy_evaluations"] !=
                hamiltonian.n_energy_evaluations):
            run["trace"].append({
                "n_energy_evaluations": int(hamiltonian.n_energy_evaluations),
                "objective_eval": int(run["n_objective_evals"]),
                "cvar": (float(run["history"][-1]) if run["history"] else float("nan")),
                "best_energy": float(tracker.best_energy),
                "best_bitstring": tracker.best_bitstring,
                "elapsed_s": float(run["runtime"]),
            })
        for point in run["trace"]:
            point["restart"] = r
        runs.append(run)
        restarts_completed += 1
        if best_run is None or run["final_objective"] < best_run["final_objective"]:
            best_run = run
    total_runtime = time.time() - t0

    if best_run is None:
        raise BudgetExhausted(budget or 0, hamiltonian.n_energy_evaluations)

    hamiltonian.release()               # hand the reserve back for the read-out

    fmt = f"0{n_qubits}b"
    final_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xF1A1]))
    if sampler is None:
        final_probs = np.asarray(circuit(best_run["final_params"]), dtype=float)
        final_probs = np.clip(final_probs, 0.0, None)
        final_probs = final_probs / final_probs.sum()
        final_idx = final_rng.choice(final_probs.size, size=final_shots,
                                     p=final_probs)
        modal_bits = format(int(np.argmax(final_probs)), fmt)
        p_sorted = np.sort(final_probs)[::-1]
        top1 = float(p_sorted[0])
        top16 = float(p_sorted[:16].sum())
        nz = final_probs[final_probs > 1e-15]
        entropy = float(-np.sum(nz * np.log2(nz)))
    else:
        final_idx = sampler.draw_indices(best_run["final_params"], final_shots,
                                         final_rng)
        vals, counts = np.unique(final_idx, return_counts=True)
        freq = counts.astype(float) / float(final_shots)
        modal_bits = format(int(vals[np.argmax(counts)]), fmt)
        top1 = float(freq.max())
        top16 = float(np.sort(freq)[::-1][:16].sum())
        entropy = float(-np.sum(freq * np.log2(freq)))
    # Scan most-probable-first. np.unique returns basis indices in *sorted* order, so if
    # the budget truncates the read-out a plain unique() scan keeps an arbitrary
    # low-index subset rather than the states the circuit actually favours.
    uniq, counts = np.unique(final_idx, return_counts=True)
    uniq = uniq[np.argsort(-counts, kind="stable")]
    readout_bits = [format(int(u), fmt) for u in uniq]
    readout_energies = (energy_batch(readout_bits)
                        if energy_batch is not None else None)
    vqe_bits, vqe_energy = None, float("inf")
    for bs in readout_bits:
        try:
            e = (readout_energies[bs]
                 if readout_energies is not None and bs in readout_energies
                 else hamiltonian.energy(bs))
        except BudgetExhausted:
            break
        if e < vqe_energy:
            vqe_energy, vqe_bits = e, bs

    try:
        modal_energy = hamiltonian.energy(modal_bits)
    except BudgetExhausted:
        modal_energy = float("nan")

    objective_trace = []
    for restart, run in enumerate(runs):
        for point in run["trace"]:
            q = dict(point)
            q["restart"] = restart
            objective_trace.append(q)
    return {
        "vqe_bitstring": vqe_bits,
        "vqe_energy": float(vqe_energy),
        "vqe_modal_bitstring": modal_bits,
        "vqe_modal_energy": float(modal_energy),
        "best_seen_bitstring": tracker.best_bitstring,
        "best_seen_energy": float(tracker.best_energy),
        "final_objective": best_run["final_objective"],
        "history": best_run["history"],
        "objective_trace": objective_trace,
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
        "spsa_a": float(spsa_a),
        "spsa_c": float(spsa_c),
        "backend": (getattr(sampler, "backend_name", "sampler")
                    if sampler is not None else "statevector"),
        "n_objective_evals_total": sum(r["n_objective_evals"] for r in runs),
        "n_objective_evals_best_run": best_run["n_objective_evals"],
        "n_spsa_iterations_total": (sum(r["n_objective_evals"] for r in runs) // 2
                                     if optimizer.upper() == "SPSA" else 0),
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
