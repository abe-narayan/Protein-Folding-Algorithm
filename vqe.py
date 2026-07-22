import pennylane as qml
import numpy as np
from scipy.optimize import minimize

from encoding import bits_to_coords
from hamiltonian import path_energy


def create_ansatz(params, n_qubits):
    for i in range(n_qubits):
        qml.RY(params[i], wires=i)

    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])


def calculate_cvar(
    params,
    sequence,
    alpha=0.1,
    repetitions=1000,
    seed=None
):
    n_qubits = 2 * (len(sequence) - 1)

    dev = qml.device(
        "default.qubit",
        wires=n_qubits,
        seed=seed,
    )

    @qml.set_shots(shots=repetitions)
    @qml.qnode(dev)
    def circuit():
        create_ansatz(params, n_qubits)
        return qml.sample(wires=range(n_qubits))

    samples = circuit()
    energies = []

    for sample in samples:
        bitstring = "".join(str(int(bit)) for bit in sample)
        energy = path_energy(bitstring, sequence)
        energies.append(energy)

    energies.sort()
    number_to_keep = max(1, int(alpha * len(energies)))
    lowest_energies = energies[:number_to_keep]

    return np.mean(lowest_energies)


def run_vqe(
    sequence,
    alpha=0.1,
    repetitions=1000,
    optimization_steps=100,
    seed=None
):
    n_qubits = 2 * (len(sequence) - 1)

    if seed is not None:
        np.random.seed(seed)

    params = np.random.uniform(0, 2 * np.pi, n_qubits)
    history = []

    def objective(current_params):
        cvar = calculate_cvar(
            current_params,
            sequence,
            alpha,
            repetitions,
            seed=seed
        )
        history.append(cvar)
        print("Step:", len(history), "CVaR:", cvar)
        return cvar

    result = minimize(
        objective,
        params,
        method="COBYLA",
        options={"maxiter": optimization_steps}
    )

    return result, history


def best_fold_from_params(
    params,
    sequence,
    repetitions=1000,
    seed=None
):
    """Sample the optimized circuit and return the lowest-energy fold it produced."""
    n_qubits = 2 * (len(sequence) - 1)

    dev = qml.device(
        "default.qubit",
        wires=n_qubits,
        seed=seed,
    )

    @qml.set_shots(shots=repetitions)
    @qml.qnode(dev)
    def circuit():
        create_ansatz(params, n_qubits)
        return qml.sample(wires=range(n_qubits))

    samples = circuit()

    best_energy = None
    best_bitstring = None

    for sample in samples:
        bitstring = "".join(str(int(bit)) for bit in sample)
        energy = path_energy(bitstring, sequence)

        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring

    best_coords = bits_to_coords(best_bitstring)

    return best_bitstring, best_coords, best_energy


def get_best_structure(result, sequence, repetitions=1000, seed=None):
    return best_fold_from_params(
        result.x, sequence, repetitions=repetitions, seed=seed
    )