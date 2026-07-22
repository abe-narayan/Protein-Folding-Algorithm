import itertools
import numpy as np
import pennylane as qml
from encoding import bits_to_coords


def build_local_hamiltonian(sequence, get_interaction, overlap_penalty=10.0):

    n_residues = len(sequence)
    n_qubits = 2 * (n_residues - 1)

    energies = []

    for bits in itertools.product([0, 1], repeat=n_qubits):
        bitstring = ''.join(str(bit) for bit in bits)
        coords = bits_to_coords(bitstring)
        energy = 0.0

        for i in range(n_residues):
            for j in range(i + 1, n_residues):

                if coords[i] == coords[j]:
                    energy += overlap_penalty

        for i in range(n_residues):
            for j in range(i + 2, n_residues):

                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dz = coords[i][2] - coords[j][2]

                if dx * dx + dy * dy + dz * dz == 8:

                    energy += get_interaction(
                        sequence[i],
                        sequence[j]
                    )

        energies.append(energy)

    matrix = np.diag(energies)

    return qml.Hamiltonian(
        [1.0],
        [qml.Hermitian(matrix, wires=range(n_qubits))]
    )
