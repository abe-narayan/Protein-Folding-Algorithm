from hamiltonian import path_energy
from encoding import bits_to_coords
import numpy

seq = ['-', 'P','H','+'] #adjustable

n_turns = len(seq)-1
n_qubits = 2*n_turns

energy_list = []
for idx in range(2 ** n_qubits):
    bitstring = format(idx, f'0{n_qubits}b')
    energy = path_energy(bitstring, seq)
    energy_list.append(energy)
energy_table = numpy.array(energy_list)

idx = energy_table.argmin()
lowest_energy = energy_table[idx]
best_bitstring = format(idx, f'0{n_qubits}b')

print(f"Lowest Energy: {lowest_energy}")
print(f"Current structure: {bits_to_coords(best_bitstring)}, in bitstring format |{best_bitstring}>")