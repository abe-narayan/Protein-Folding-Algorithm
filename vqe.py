import cirq
import numpy as np

from scipy.optimize import minimize

from hamiltonian import path_energy


def create_ansatz(
    params,
    n_qubits
):

    qubits = cirq.LineQubit.range(n_qubits)

    circuit = cirq.Circuit()

    for i in range(n_qubits):

        circuit.append(
            cirq.ry(params[i])(
                qubits[i]
            )
        )

    for i in range(n_qubits - 1):

        circuit.append(
            cirq.CNOT(
                qubits[i],
                qubits[i + 1]
            )
        )

    circuit.append(
        cirq.measure(
            *qubits,
            key="result"
        )
    )

    return circuit


def calculate_cvar(
    params,
    sequence,
    alpha=0.1,
    repetitions=1000
):

    n_qubits = 2 * (len(sequence) - 1)

    circuit = create_ansatz(
        params,
        n_qubits
    )

    simulator = cirq.Simulator()

    result = simulator.run(
        circuit,
        repetitions=repetitions
    )

    samples = result.measurements["result"]

    energies = []

    for sample in samples:

        bitstring = ""

        for bit in sample:

            bitstring += str(bit)

        energy = path_energy(
            bitstring,
            sequence
        )

        energies.append(energy)

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

    def objective(current_params):

        cvar = calculate_cvar(
            current_params,
            sequence,
            alpha,
            repetitions
        )

        history.append(cvar)

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