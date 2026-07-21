import numpy as np
from encoding import (
    bits_to_coords,
    bits_to_directions,
    OPPOSITE
)

DIRECTIONS = {

(0,0):(1,0),
(0,1):(0,1),
(1,0):(-1,0),
(1,1):(0,-1),
}

OPPOSITE = {

(0,0):(1,0),
(1,0):(0,0),
(0,1):(1,1),
(1,1):(0,1),
}


INTERACTIONS = {
    ('H', 'H'): -1.0, ('H', 'P'):  0.0, ('H', '+'):  0.0, ('H', '-'):  0.0,
    ('P', 'H'):  0.0, ('P', 'P'):  0.0, ('P', '+'): -0.2, ('P', '-'): -0.2,
    ('+', 'H'):  0.0, ('+', 'P'): -0.2, ('+', '+'):  1.0, ('+', '-'): -1.0,
    ('-', 'H'):  0.0, ('-', 'P'): -0.2, ('-', '+'): -1.0, ('-', '-'):  1.0,
}


FALLBACK_INTERACTION = 0.0 # if not listed above

def get_interaction(a, b):
    return INTERACTIONS.get((a, b), FALLBACK_INTERACTION)



def path_energy(bitstring, overlap_penalty=10.0, reversal_penalty=5.0):
    coords = bits_to_coords(bitstring)
    dirs = bits_to_directions(bitstring)
    energy = 0.0

    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            if coords[i] == coords[j]:
                energy += overlap_penalty

    for i in range(n_residues):
        for j in range(i + 2, n_residues):
            dist = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
            if dist == 1:
                energy += get_interaction(sequence[i], sequence[j])

    for i in range(len(dirs) - 1):
        if OPPOSITE[dirs[i]] == dirs[i + 1]:
            energy += reversal_penalty

    
    return energy

energy_table = np.array([
    path_energy(format(idx, f'0{n_qubits}b')) for idx in range(2 ** n_qubits)
])
