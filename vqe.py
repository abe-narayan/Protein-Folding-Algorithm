import numpy as np
import pennylane as qml
from scipy.optimize import minimize

from encoding import bits_to_coords
from hamiltonian import path_energy


def create_ansatz(params, n_qubits, layers=3):
    idx = 0
    for _ in range(layers):
        for i in range(n_qubits):
            qml.RY(params[idx], wires=i)
            idx += 1
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])




def _build_probs_circuit(n_qubits, layers):
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        create_ansatz(params, n_qubits, layers=layers)
        return qml.probs(wires=range(n_qubits))

    return circuit


def _sample_energies(probs, sequence, n_qubits, shots, rng):
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()  
    indices = rng.choice(len(probs), size=shots, p=probs)
    fmt = f"0{n_qubits}b"
    return np.array([path_energy(format(idx, fmt), sequence) for idx in indices])


def _cvar(energies, alpha):
    energies_sorted = np.sort(energies)
    keep = max(1, int(alpha * len(energies_sorted)))
    return float(energies_sorted[:keep].mean())


def calculate_cvar(params, sequence, alpha=0.1, repetitions=1000, seed=None, layers=3):
    """Exact circuit + finite-shot CVaR estimate for one parameter vector."""
    n_qubits = 2 * (len(sequence) - 1)
    circuit = _build_probs_circuit(n_qubits, layers)
    probs = circuit(params)
    rng = np.random.default_rng(seed)
    energies = _sample_energies(probs, sequence, n_qubits, repetitions, rng)
    return _cvar(energies, alpha)




def _spawn_seeds(seed, n):
    if seed is None:
        return [None] * n
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]


def _robust_estimate(circuit, sequence, n_qubits, params, shots, alpha, seed, n_repeats=3):
    """Average CVaR over several independent shot draws, for a fair,
    low-variance comparison across restarts (instead of one noisy read)."""
    probs = circuit(params)
    repeat_seeds = _spawn_seeds(seed, n_repeats)
    scores = []
    for rseed in repeat_seeds:
        rng = np.random.default_rng(rseed)
        energies = _sample_energies(probs, sequence, n_qubits, shots, rng)
        scores.append(_cvar(energies, alpha))
    return float(np.mean(scores))





def _single_vqe_run(sequence, alpha, repetitions, optimization_steps, seed, layers):
    n_qubits = 2 * (len(sequence) - 1)
    n_params = layers * n_qubits

    init_seed, low_seed, mid_seed, high_seed, eval_seed = _spawn_seeds(seed, 5)


    rng_init = np.random.default_rng(init_seed)
    params0 = rng_init.normal(loc=0.0, scale=0.1, size=n_params)

    circuit = _build_probs_circuit(n_qubits, layers)

    low_shots = max(100, repetitions // 4)
    mid_shots = max(200, repetitions // 2)
    high_shots = repetitions
    third = max(1, optimization_steps // 3)

    history = []
    call_count = 0
    best_high_energy = None
    best_high_params = None

    def objective(params):
        nonlocal call_count, best_high_energy, best_high_params
        call_count += 1

        if call_count <= third:
            shots, stage_seed, stage_name = low_shots, low_seed, "low"
        elif call_count <= 2 * third:
            shots, stage_seed, stage_name = mid_shots, mid_seed, "mid"
        else:
            shots, stage_seed, stage_name = high_shots, high_seed, "high"

        probs = circuit(params)
        rng = np.random.default_rng(stage_seed)  # same seed within a tier -> CRN
        energies = _sample_energies(probs, sequence, n_qubits, shots, rng)
        cvar = _cvar(energies, alpha)
        history.append(cvar)
        print(f"  call {call_count:3d} [{stage_name:>4s} shots={shots:4d}] CVaR={cvar:.4f}")

        if stage_name == "high" and (best_high_energy is None or cvar < best_high_energy):
            best_high_energy = cvar
            best_high_params = params.copy()

        return cvar

    result = minimize(
        objective,
        params0,
        method="COBYLA",
        options={"maxiter": optimization_steps, "rhobeg": 0.3},
    )

    if best_high_energy is not None and best_high_energy < result.fun:
        result.x = best_high_params
        result.fun = best_high_energy


    result.fun = _robust_estimate(circuit, sequence, n_qubits, result.x, high_shots, alpha, eval_seed)

    return result, history



def run_vqe(sequence, alpha=0.1, repetitions=1000, optimization_steps=100, seed=42, layers=3, restarts=5):
    restart_seeds = _spawn_seeds(seed, restarts)

    best_result = None
    best_history = None

    for i, restart_seed in enumerate(restart_seeds):
        print(f"--- Restart {i + 1}/{restarts} ---")
        result, history = _single_vqe_run(sequence, alpha, repetitions, optimization_steps, restart_seed, layers)

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_history = history

    return best_result, best_history




def best_fold_from_params(params, sequence, repetitions=1000, seed=42, layers=3):
    n_qubits = 2 * (len(sequence) - 1)
    circuit = _build_probs_circuit(n_qubits, layers)
    probs = circuit(params)

    rng = np.random.default_rng(seed)
    probs_arr = np.asarray(probs, dtype=np.float64)
    probs_arr = probs_arr / probs_arr.sum()
    indices = rng.choice(len(probs_arr), size=repetitions, p=probs_arr)
    fmt = f"0{n_qubits}b"

    best_energy = None
    best_bitstring = None
    for idx in indices:
        bitstring = format(idx, fmt)
        energy = path_energy(bitstring, sequence)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring

    best_coords = bits_to_coords(best_bitstring)
    return best_bitstring, best_coords, best_energy


def get_best_structure(result, sequence, repetitions=1000, seed=42, layers=3):
    return best_fold_from_params(result.x, sequence, repetitions=repetitions, seed=seed, layers=layers)