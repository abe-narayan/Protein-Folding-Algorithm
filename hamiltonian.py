import matplotlib.pyplot as plt
from local_hamiltonian import build_local_hamiltonian

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


def path_hamiltonian(sequence, overlap_penalty=10.0):
    return build_local_hamiltonian(sequence, get_interaction, overlap_penalty)
