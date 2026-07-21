import time
import numpy as np
import itertools
energy_interactions = {
    ('+', '+'): +placeholder,
    ('-', '-'): +placeholder,
    ('+','-'): -placeholder,
    ('-','+'):-placeholder,
    ('H', 'H'): -placeholder
}

DIRECTIONS = {
    (0, 0): (1, 0),   # +x
    (0, 1): (0, 1),   # +y
    (1, 0): (-1, 0),  # -x
    (1, 1): (0, -1),  # -y
}
OPPOSITE = {
    (0, 0): (1, 0),
    (1, 0): (0, 0),
    (0, 1): (1, 1),
    (1, 1): (0, 1),
}

lambda_back = 1000
def turn_calculation(aminos, turns):
    classical_energy = 0
    for i in range(0, len(turns)-1):
        if turns[i+1] == OPPOSITE[turns[i]]:
            classical_energy += lambda_back
    for j in range(0, len(turns)-2):
        if turns[i+2] == OPPOSITE[turns[i]] and turns[i+1]!=turns[i] and turns[i+2]!=OPPOSITE[turns[i]]:
            classical_energy+=energy_interactions.get((aminos[i], aminos[i+3]),0.0)
    return classical_energy

