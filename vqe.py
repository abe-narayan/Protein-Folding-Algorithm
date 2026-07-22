import pennylane as qml
import numpy as np

from scipy.optimize import minimize

from hamiltonian import path_energy


def create_ansatz(
    params,
    n_qubits
):

    for i in range(n_qubits):

        qml.RY(
            params[i],
            wires=i
        )

    for i in range(n_qubits - 1):

        qml.CNOT(
            wires=[
                i,
                i + 1
            ]
        )


def calculate_cvar(
    params,
    sequence,
    alpha=0.1,
    repetitions=1000
):

    n_qubits = 2 * (len(sequence) - 1)

    dev = qml.device(
        "default.qubit",
        wires=n_qubits,
    )

    @qml.set_shots(shots=repetitions)
    @qml.qnode(dev)
    def circuit():

        create_ansatz(
            params,
            n_qubits
        )

        return qml.sample(
            wires=range(n_qubits)
        )


    samples = circuit()

    energies = []

    for sample in samples:

        bitstring = ""

        for bit in sample:

            bitstring += str(int(bit))

        energy = path_energy(
            bitstring,
            sequence
        )

        energies.append(
            energy
        )


    energies.sort()

    number_to_keep = max(
        1,
        int(alpha * len(energies))
    )

    lowest_energies = energies[
        :number_to_keep
    ]

    cvar = np.mean(
        lowest_energies
    )

    return cvar


def run_vqe(
    sequence,
    alpha=0.1,
    repetitions=1000,
    optimization_steps=100
):

    n_qubits = 2 * (len(sequence) - 1)

    params = np.random.uniform(
        0,
        2 * np.pi,
        n_qubits
    )

    history = []


    def objective(
        current_params
    ):

        cvar = calculate_cvar(
            current_params,
            sequence,
            alpha,
            repetitions
        )

        history.append(
            cvar
        )

        print(
            "Step:",
            len(history),
            "CVaR:",
            cvar
        )

        return cvar


    result = minimize(
        objective,
        params,
        method="COBYLA",
        options={
            "maxiter": optimization_steps
        }
    )


    return result, history

def get_best_structure(result, sequence, repetitions=1000):
    from encoding import bits_to_coords

    optimal_params = result.x
    n_qubits = 2 * (len(sequence) - 1)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.set_shots(shots=repetitions)
    @qml.qnode(dev)
    def circuit():
        create_ansatz(optimal_params, n_qubits)
        return qml.sample(wires=range(n_qubits))

    samples = circuit()

    best_energy = float("inf")
    best_bitstring = ""

    for sample in samples:
        bitstring = "".join(str(int(bit)) for bit in sample)
        energy = path_energy(bitstring, sequence)

        if energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring

    best_coords = bits_to_coords(best_bitstring)

    return best_bitstring, best_coords, best_energy