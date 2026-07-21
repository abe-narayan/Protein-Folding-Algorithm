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