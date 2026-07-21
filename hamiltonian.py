from encoding import bits_to_coords, bits_to_directions, DIRECTIONS
import cirq as cirq
import matplotlib.pyplot as plt


INTERACTIONS = {
    ('H', 'H'): -1.0, ('H', 'P'):  0.0, ('H', '+'):  0.0, ('H', '-'):  0.0,
    ('P', 'H'):  0.0, ('P', 'P'):  0.0, ('P', '+'): -0.2, ('P', '-'): -0.2,
    ('+', 'H'):  0.0, ('+', 'P'): -0.2, ('+', '+'):  1.0, ('+', '-'): -1.0,
    ('-', 'H'):  0.0, ('-', 'P'): -0.2, ('-', '+'): -1.0, ('-', '-'):  1.0,
}


FALLBACK_INTERACTION = 0.0 # if not listed above

def get_interaction(a, b):
    if (a, b) in INTERACTIONS:
        return INTERACTIONS[(a, b)]
    return FALLBACK_INTERACTION



def path_energy(bitstring, sequence, overlap_penalty=10.0):
    coords = bits_to_coords(bitstring)
    dirs = bits_to_directions(bitstring)
    energy = 0.0
    n_residues = len(sequence)
    expected_bits = 2 * (n_residues - 1)

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
                energy += get_interaction(sequence[i], sequence[j])

    if len(bitstring) != expected_bits:
        raise ValueError(
            f"Expected {expected_bits} bits for {len(sequence)} residues, "
            f"but received {len(bitstring)}"
        )

    return energy
