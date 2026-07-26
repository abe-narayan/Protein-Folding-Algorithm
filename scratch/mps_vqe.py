"""Faithful copy of vqe._run_single / run_global_cvar_vqe with a pluggable sampler.
sampler=None -> lightning path, TEXTUALLY identical ops to vqe.py (byte-identical).
sampler=MPSSampler -> draw indices from the exact MPS. Everything downstream
(unique -> energy -> uniq_energies[inverse] -> cvar) is unchanged, so multiplicities
and CVaR are preserved. This scratch module mirrors the exact vqe.py diff to paste."""
import math, time
import numpy as np
from scipy.optimize import minimize
from vqe import (cvar_from_samples, BestSeenTracker, n_parameters,
                 build_global_circuit, _spsa)


def _run_single(hamiltonian, circuit, sampler, n_qubits, layers, alpha, shots,
                maxiter, seed, optimizer, tracker, init_scale, verbose):
    fmt = f"0{n_qubits}b"
    n_par = n_parameters(n_qubits, layers)
    init_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC0FFEE]))
    params0 = (math.pi / 2.0) + init_rng.normal(0.0, init_scale, size=n_par)
    history = []
    lowtail_collapse = []          # AUDIT: per-eval count of low-tail structures < -450
    eval_counter = {"n": 0}

    def objective(params):
        k = eval_counter["n"]; eval_counter["n"] = k + 1
        rng = np.random.default_rng(np.random.SeedSequence([seed, k]))
        if sampler is None:                       # ---- lightning: unchanged ----
            probs = np.asarray(circuit(params), dtype=float)
            probs = np.clip(probs, 0.0, None)
            s = probs.sum()
            if s <= 0:
                probs = np.full_like(probs, 1.0 / probs.size)
            else:
                probs = probs / s
            idx = rng.choice(probs.size, size=shots, p=probs)
        else:                                     # ---- MPS: draw from exact MPS ----
            idx = sampler.draw_indices(params, shots, rng)
        uniq, inverse = np.unique(idx, return_inverse=True)
        uniq_energies = np.empty(uniq.size, dtype=float)
        for m, u in enumerate(uniq):
            bs = format(int(u), fmt)
            e = hamiltonian.energy(bs)
            uniq_energies[m] = e
            tracker.offer(bs, e)
        sample_energies = uniq_energies[inverse]
        val = cvar_from_samples(sample_energies, alpha)
        keep = max(1, int(np.ceil(alpha * sample_energies.size)))   # AUDIT (pure obs)
        lowtail_collapse.append(int((np.sort(sample_energies)[:keep] < -450.0).sum()))
        history.append(val)
        if verbose and (k % 25 == 0):
            print(f"      eval {k:4d} | CVaR {val:9.3f} | best seen {tracker.best_energy:9.3f}")
        return val

    t0 = time.time()
    if optimizer.upper() == "COBYLA":
        res = minimize(objective, params0, method="COBYLA",
                       options={"maxiter": maxiter, "rhobeg": 0.4})
    elif optimizer.upper() == "SPSA":
        res = _spsa(objective, params0, maxiter,
                    np.random.default_rng(np.random.SeedSequence([seed, 7])))
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    runtime = time.time() - t0
    final_params = np.asarray(res.x if hasattr(res, "x") else res, dtype=float)
    return {"final_params": final_params,
            "final_objective": float(res.fun) if hasattr(res, "fun") else float(objective(final_params)),
            "history": history, "lowtail_collapse": lowtail_collapse,
            "n_objective_evals": eval_counter["n"], "runtime": runtime}


def run_global_cvar_vqe(hamiltonian, layers=4, alpha=0.15, shots=2048, maxiter=300,
                        restarts=4, seed=0, optimizer="COBYLA", ring=True,
                        final_shots=8192, init_scale=0.6, device="lightning.qubit",
                        sampler=None, verbose=False):
    n_qubits = hamiltonian.n_qubits
    n_par = n_parameters(n_qubits, layers)
    if optimizer.upper() == "COBYLA" and maxiter < n_par + 2:
        raise ValueError(f"COBYLA needs maxiter >= {n_par + 2}; got {maxiter}")
    circuit = None if sampler is not None else build_global_circuit(
        n_qubits, layers, ring=ring, device=device)
    tracker = BestSeenTracker()
    hamiltonian.reset_counters()
    restart_seeds = [int(s.generate_state(1)[0])
                     for s in np.random.SeedSequence(seed).spawn(restarts)]
    t0 = time.time(); runs = []; best_run = None
    for r, rseed in enumerate(restart_seeds):
        if verbose:
            print(f"    --- restart {r + 1}/{restarts} (seed {rseed}) ---")
        run = _run_single(hamiltonian, circuit, sampler, n_qubits, layers, alpha,
                          shots, maxiter, rseed, optimizer, tracker, init_scale, verbose)
        runs.append(run)
        if best_run is None or run["final_objective"] < best_run["final_objective"]:
            best_run = run
    total_runtime = time.time() - t0

    fmt = f"0{n_qubits}b"
    final_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xF1A1]))
    if sampler is None:                            # ---- lightning: unchanged ----
        final_probs = np.asarray(circuit(best_run["final_params"]), dtype=float)
        final_probs = np.clip(final_probs, 0.0, None); final_probs = final_probs / final_probs.sum()
        final_idx = final_rng.choice(final_probs.size, size=final_shots, p=final_probs)
        modal_bits = format(int(np.argmax(final_probs)), fmt)
        p_sorted = np.sort(final_probs)[::-1]
        top1 = float(p_sorted[0]); top16 = float(p_sorted[:16].sum())
        nz = final_probs[final_probs > 1e-15]; entropy = float(-np.sum(nz * np.log2(nz)))
    else:                                          # ---- MPS: empirical final stats ----
        final_idx = sampler.draw_indices(best_run["final_params"], final_shots, final_rng)
        counts = np.bincount(final_idx); freq = counts[counts > 0] / final_shots
        modal_bits = format(int(np.argmax(counts)), fmt)
        top1 = float(freq.max()); top16 = float(np.sort(freq)[::-1][:16].sum())
        entropy = float(-np.sum(freq * np.log2(freq)))
    uniq = np.unique(final_idx)
    vqe_bits, vqe_energy = None, float("inf")
    final_energies = []                         # AUDIT: energies of the final selected sample
    for u in uniq:
        bs = format(int(u), fmt); e = hamiltonian.energy(bs)
        final_energies.append(e)
        if e < vqe_energy:
            vqe_energy, vqe_bits = e, bs
    modal_energy = hamiltonian.energy(modal_bits)
    final_energies = np.array(final_energies)
    audit = {
        "n_evals_lowtail_collapse_best_run": int(sum(1 for x in best_run["lowtail_collapse"] if x > 0)),
        "n_evals_best_run": len(best_run["lowtail_collapse"]),
        "n_evals_lowtail_collapse_all": int(sum(1 for r in runs for x in r["lowtail_collapse"] if x > 0)),
        "final_selected_energy": float(vqe_energy),
        "final_selected_is_collapse": bool(vqe_energy < -450.0),
        "final_sample_n_collapse": int((final_energies < -450.0).sum()),
        "final_sample_n_unique": int(final_energies.size),
        "final_sample_lowest5": [float(x) for x in np.sort(final_energies)[:5]],
    }
    return {"vqe_bitstring": vqe_bits, "vqe_energy": float(vqe_energy),
            "vqe_modal_bitstring": modal_bits, "vqe_modal_energy": float(modal_energy),
            "best_seen_bitstring": tracker.best_bitstring, "best_seen_energy": float(tracker.best_energy),
            "final_objective": best_run["final_objective"], "history": best_run["history"],
            "distribution_top1_prob": top1, "distribution_top16_mass": top16,
            "distribution_entropy_bits": entropy, "max_entropy_bits": float(n_qubits),
            "n_qubits": n_qubits, "n_parameters": n_par, "layers": layers, "alpha": alpha,
            "shots_per_eval": shots, "final_shots": final_shots, "restarts": restarts,
            "optimizer": optimizer, "n_objective_evals_total": sum(r["n_objective_evals"] for r in runs),
            "n_objective_evals_best_run": best_run["n_objective_evals"],
            "n_energy_evaluations": hamiltonian.n_energy_evaluations,
            "n_unique_structures_cached": hamiltonian.cache_size(),
            "audit": audit, "runtime": total_runtime, "seed": seed}
