
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from scipy.optimize import minimize


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
        "runtime": total_runtime,
        "seed": seed,
    }