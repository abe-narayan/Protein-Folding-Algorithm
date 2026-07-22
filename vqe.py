import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
import pennylane as qml
from scipy.optimize import minimize

from encoding import bits_to_coords
from hamiltonian import path_energy


class VQEStateTracker:
    def __init__(self, sequence):
        self.sequence = sequence
        self.energy_cache = {}
        self.best_energy = float('inf')
        self.best_bitstring = None
        self.best_coords = None

    def evaluate(self, bitstring):
        if bitstring in self.energy_cache:
            return self.energy_cache[bitstring]
        
        energy = path_energy(bitstring, self.sequence)
        self.energy_cache[bitstring] = energy
        
        if energy < self.best_energy:
            self.best_energy = energy
            self.best_bitstring = bitstring
            self.best_coords = bits_to_coords(bitstring)
            
        return energy


def create_ansatz(params, n_qubits, layers=4):
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


def _cvar(energies, alpha):
    energies_sorted = np.sort(energies)
    keep = max(1, int(alpha * len(energies_sorted)))
    return float(energies_sorted[:keep].mean())


def _spawn_seeds(seed, n):
    if seed is None:
        return [None] * n
    return [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]


def _single_vqe_run(sequence, alpha, shots, optimization_steps, seed, layers, circuit, n_qubits, tracker):
    n_params = layers * n_qubits
    rng_init = np.random.default_rng(seed)
    params0 = rng_init.normal(loc=0.0, scale=0.1, size=n_params)

    history = []
    call_count = 0

    def objective(params):
        nonlocal call_count
        call_count += 1
        
        rng = np.random.default_rng(seed)
        
        probs = circuit(params)
        probs = np.asarray(probs, dtype=np.float64)
        
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()  

        indices = rng.choice(len(probs), size=shots, p=probs)
        unique_indices = np.unique(indices)
        fmt = f"0{n_qubits}b"
        
        val_map = {}
        for idx in unique_indices:
            bstr = format(idx, fmt)
            val_map[idx] = tracker.evaluate(bstr)
            
        energies = np.array([val_map[idx] for idx in indices])
        cvar = _cvar(energies, alpha)
        
        history.append(cvar)
        if call_count % 10 == 0 or call_count == 1:
            print(f"  eval {call_count:3d} | CVaR: {cvar:.4f} | Global Best: {tracker.best_energy:.4f}")

        return cvar

    result = minimize(
        objective,
        params0,
        method="L-BFGS-B",
        options={"maxiter": optimization_steps, "maxfun": optimization_steps * 3, "eps": 1e-3},
    )

    return result, history


def run_vqe(sequence, alpha=0.1, repetitions=1200, optimization_steps=120, seed=42, layers=4, restarts=6):
    n_qubits = 2 * (len(sequence) - 1)
    circuit = _build_probs_circuit(n_qubits, layers)
    restart_seeds = _spawn_seeds(seed, restarts)

    tracker = VQEStateTracker(sequence)
    
    best_result = None
    best_history = None

    for i, restart_seed in enumerate(restart_seeds):
        print(f"--- Restart {i + 1}/{restarts} ---")
        result, history = _single_vqe_run(
            sequence, alpha, repetitions, optimization_steps, 
            restart_seed, layers, circuit, n_qubits, tracker
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result
            best_history = history

    best_result.tracker = tracker
    return best_result, best_history


def get_best_structure(result, sequence, repetitions=1200, seed=42, layers=4):
    tracker = result.tracker
    return tracker.best_bitstring, tracker.best_coords, tracker.best_energy