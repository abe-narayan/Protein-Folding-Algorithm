import pennylane as qml

from hamiltonian import path_energy


def build_cost_function(
    sequence,
    overlap_penalty=30.0
):

    def cost_function(bitstring):

        return path_energy(
            bitstring,
            sequence,
            overlap_penalty
        )

    return cost_function