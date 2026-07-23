"""Global full-system CVaR-VQE.

WHAT THIS IS
------------
ONE parameterized quantum circuit acting on ALL n_qubits of the problem. The
whole protein configuration is optimized jointly. There is no block
decomposition, no coordinate descent, and no enumeration of the configuration
space anywhere in the search path.

ANSATZ
------
Hardware-efficient, `layers` repetitions of:

    RY(theta_{l,q}) on every qubit q
    CNOT chain q -> q+1 for q = 0..n-2
    CNOT ring closure n-1 -> 0        (if n > 2 and ring=True)

Parameter count: layers * n_qubits.

Entanglement: the CNOT chain+ring generates genuine multi-qubit entanglement
in the state |psi(theta)>. However -- see the audit -- because H is diagonal
and all gates are real, the resulting *measurement distribution* is one that
a classical model could also represent. Entanglement is present in the state;
it is not being used as a computational resource in a way that is known to
help. This is stated plainly rather than dressed up.

OBJECTIVE
---------
CVaR_alpha over sampled energies, preserving sample multiplicities:

    draw S bitstrings x_1..x_S ~ p_theta(x)
    sort their energies ascending
    CVaR = mean of the lowest ceil(alpha * S) energies

Multiplicities are preserved: if a bitstring is drawn 40 times it contributes
40 entries to the sorted list. This is the correct sample CVaR estimator, and
it is the behaviour of the original implementation, retained deliberately.

RNG DISCIPLINE
--------------
The original implementation recreated `np.random.default_rng(seed)` inside
every objective call, so all evaluations shared one fixed random tape. That
is fixed here with a *counter-based* stream: evaluation k uses
`default_rng(SeedSequence([seed, k]))`. This gives (a) statistically
independent samples across evaluations, and (b) exact reproducibility for a
given seed, and (c) common random numbers within a single objective call so
that finite-difference gradients are not swamped by sampling noise.

FINAL ANSWER EXTRACTION
-----------------------
Two quantities are reported SEPARATELY and must never be conflated:

  best_seen        lowest-energy bitstring encountered at ANY point during
                   optimization, including early random evaluations. This is
                   an *anytime* result, not a VQE result. The original
                   implementation reported this as "the VQE answer", which
                   overstated what the optimizer achieved.

  vqe_solution     obtained from the FINAL optimized circuit only: draw
                   `final_shots` samples from p_theta*(x) and take the
                   lowest-energy sample. Also reported: `vqe_modal`, the
                   single most probable bitstring under p_theta*.

If best_seen is much better than vqe_solution, the VQE did not converge to a
useful distribution and the run should be reported as such.
"""
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from scipy.optimize import minimize


# ==========================================================================
# CVaR
# ==========================================================================
def cvar_from_samples(energies: Sequence[float], alpha: float) -> float:
    """Sample CVaR: mean of the lowest ceil(alpha*S) energies.

    `energies` MUST contain one entry per DRAWN SAMPLE, including repeats.
    Passing a de-duplicated list silently destroys the probability weighting
    and is a correctness bug (it was one in a previous design).
    """
    e = np.sort(np.asarray(energies, dtype=float))
    if e.size == 0:
        raise ValueError("cvar_from_samples received no samples")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    keep = max(1, int(math.ceil(alpha * e.size)))
    return float(e[:keep].mean())


def cvar_from_distribution(energies: np.ndarray, probs: np.ndarray,
                           alpha: float) -> float:
    """Exact CVaR against a full probability vector.

    VALIDATION ONLY -- requires enumerating all 2^n energies, which is
    exactly what the VQE search path must not do. Used by validation.py to
    confirm that cvar_from_samples converges to the right value.
    """
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


# ==========================================================================
# Ansatz / circuit
# ==========================================================================
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


# ==========================================================================
# Tracker
# ==========================================================================
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


# ==========================================================================
# Single VQE run
# ==========================================================================
def _run_single(hamiltonian, circuit, n_qubits: int, layers: int,
                alpha: float, shots: int, maxiter: int, seed: int,
                optimizer: str, tracker: BestSeenTracker,
                init_scale: float, verbose: bool) -> Dict:
    fmt = f"0{n_qubits}b"
    n_par = n_parameters(n_qubits, layers)

    init_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC0FFEE]))
    # Centre the RY angles at pi/2 (equal superposition) rather than 0, and
    # perturb around that. Initialising near theta=0 starts the circuit at
    # |00...0>, a near-deterministic state from which a CVaR objective has
    # little incentive -- and COBYLA no gradient signal -- to spread out.
    params0 = (math.pi / 2.0) + init_rng.normal(0.0, init_scale, size=n_par)

    history: List[float] = []
    eval_counter = {"n": 0}

    def objective(params: np.ndarray) -> float:
        k = eval_counter["n"]
        eval_counter["n"] = k + 1
        # Counter-based stream: independent across evaluations, reproducible.
        rng = np.random.default_rng(np.random.SeedSequence([seed, k]))

        probs = np.asarray(circuit(params), dtype=float)
        probs = np.clip(probs, 0.0, None)
        s = probs.sum()
        if s <= 0:
            probs = np.full_like(probs, 1.0 / probs.size)
        else:
            probs = probs / s

        idx = rng.choice(probs.size, size=shots, p=probs)

        # Evaluate energy once per UNIQUE index, then expand back to the full
        # sample list. This preserves multiplicities in the CVaR while not
        # paying the energy cost more than once per distinct structure.
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

    t0 = time.time()
    if optimizer.upper() == "COBYLA":
        res = minimize(objective, params0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.4})
    elif optimizer.upper() == "SPSA":
        res = _spsa(objective, params0, maxiter,
                    np.random.default_rng(np.random.SeedSequence([seed, 7])))
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}; use COBYLA or SPSA")
    runtime = time.time() - t0

    final_params = np.asarray(res.x if hasattr(res, "x") else res, dtype=float)
    return {
        "final_params": final_params,
        "final_objective": float(res.fun) if hasattr(res, "fun")
                           else float(objective(final_params)),
        "history": history,
        "n_objective_evals": eval_counter["n"],
        "runtime": runtime,
    }


class _SPSAResult:
    def __init__(self, x, fun):
        self.x = x
        self.fun = fun


def _spsa(objective, x0, n_iter, rng, a=0.25, c=0.15):
    """Simultaneous Perturbation Stochastic Approximation.

    Two objective evaluations per iteration regardless of dimension, which
    is the right choice for noisy shot-based objectives at high parameter
    count. COBYLA is the default because it is deterministic given the RNG
    discipline above and converges faster at these sizes.
    """
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


# ==========================================================================
# Public entry point
# ==========================================================================
def run_global_cvar_vqe(hamiltonian, layers: int = 4, alpha: float = 0.15,
                        shots: int = 2048, maxiter: int = 300,
                        restarts: int = 4, seed: int = 0,
                        optimizer: str = "COBYLA", ring: bool = True,
                        final_shots: int = 8192, init_scale: float = 0.6,
                        device: str = "lightning.qubit",
                        verbose: bool = False) -> Dict:
    """Global CVaR-VQE over the entire protein configuration register.

    Returns a dict distinguishing best-seen from the final VQE solution.
    """
    n_qubits = hamiltonian.n_qubits
    if n_qubits > 30:
        raise MemoryError(
            f"n_qubits={n_qubits} requires ~{2**n_qubits * 16 / 1e9:.0f} GB "
            "for a statevector. A genuine full-system VQE is not simulable "
            "at this size. Reduce protein length or state count.")



    n_par = n_parameters(n_qubits, layers)
    if optimizer.upper() == "COBYLA" and maxiter < n_par + 2:
        raise ValueError(
            f"COBYLA needs maxiter >= n_params + 2 = {n_par + 2}; got {maxiter}. "
            f"With n_qubits={n_qubits} and layers={layers} there are {n_par} "
            "parameters. Increase maxiter or reduce layers.")

    circuit = build_global_circuit(n_qubits, layers, ring=ring, device=device)
    tracker = BestSeenTracker()


    hamiltonian.reset_counters()

    restart_seeds = [int(s.generate_state(1)[0])
                     for s in np.random.SeedSequence(seed).spawn(restarts)]

    t0 = time.time()
    runs = []
    best_run = None
    for r, rseed in enumerate(restart_seeds):
        if verbose:
            print(f"    --- restart {r + 1}/{restarts} (seed {rseed}) ---")
        run = _run_single(hamiltonian, circuit, n_qubits, layers, alpha,
                          shots, maxiter, rseed, optimizer, tracker,
                          init_scale, verbose)
        runs.append(run)
        if best_run is None or run["final_objective"] < best_run["final_objective"]:
            best_run = run
    total_runtime = time.time() - t0

    # ---- FINAL VQE SOLUTION: sample the final optimized circuit only ----
    fmt = f"0{n_qubits}b"
    final_probs = np.asarray(circuit(best_run["final_params"]), dtype=float)
    final_probs = np.clip(final_probs, 0.0, None)
    final_probs = final_probs / final_probs.sum()

    final_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xF1A1]))
    final_idx = final_rng.choice(final_probs.size, size=final_shots,
                                 p=final_probs)
    uniq = np.unique(final_idx)
    vqe_bits, vqe_energy = None, float("inf")
    for u in uniq:
        bs = format(int(u), fmt)
        e = hamiltonian.energy(bs)
        if e < vqe_energy:
            vqe_energy, vqe_bits = e, bs

    modal_bits = format(int(np.argmax(final_probs)), fmt)
    modal_energy = hamiltonian.energy(modal_bits)

    # Concentration diagnostics: how peaked did the circuit actually get?
    p_sorted = np.sort(final_probs)[::-1]
    top1 = float(p_sorted[0])
    top16 = float(p_sorted[:16].sum())
    nz = final_probs[final_probs > 1e-15]
    entropy = float(-np.sum(nz * np.log2(nz)))

    return {
        # --- final VQE answer (from the optimized circuit) ---
        "vqe_bitstring": vqe_bits,
        "vqe_energy": float(vqe_energy),
        "vqe_modal_bitstring": modal_bits,
        "vqe_modal_energy": float(modal_energy),
        # --- anytime best (NOT the VQE answer) ---
        "best_seen_bitstring": tracker.best_bitstring,
        "best_seen_energy": float(tracker.best_energy),
        # --- diagnostics ---
        "final_objective": best_run["final_objective"],
        "history": best_run["history"],
        "distribution_top1_prob": top1,
        "distribution_top16_mass": top16,
        "distribution_entropy_bits": entropy,
        "max_entropy_bits": float(n_qubits),
        # --- budget accounting ---
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
        "runtime": total_runtime,
        "seed": seed,
    }