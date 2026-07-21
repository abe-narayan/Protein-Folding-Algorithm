import matplotlib.pyplot as plt
import numpy as np
import sympy
import itertools
from math import radians, degrees
from scipy.optimize import minimize
import cirq as cirq

def ansatz(params):
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(4)
    circuit.append(cirq.ry(params[0])(qubits[0]))
    circuit.append(cirq.ry(params[1])(qubits[1]))
    circuit.append(cirq.CX(qubits[0], qubits[1]))
    circuit.append(cirq.ry(params[2])(qubits[2]))
    circuit.append(cirq.ry(params[3])(qubits[3]))
    circuit.append(cirq.measure(*qubits, key='m'))
    return circuit

ansatz_params = [0.1, 0.2, 0.3, 0.4]
ansatz_results = ansatz(ansatz_params)

sim = cirq.Simulator()
result = sim.run(ansatz_results, repetitions=1000)

hist = cirq.plot_state_histogram(result, plt.subplot(),
                                 title='Measurement Results',
                                 xlabel='State',
                                 ylabel='Counts')

plt.show()