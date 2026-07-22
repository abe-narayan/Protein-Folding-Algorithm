from local_hamiltonian import build_cost_function


sequence = "HP+-HHP+-HHP+-"


cost_function = build_cost_function(sequence)


bitstring = "0000000000000000000000000000"


energy = cost_function(bitstring)


print("Sequence:", sequence)
print("Number of amino acids:", len(sequence))
print("Number of qubits:", 2 * (len(sequence) - 1))
print("Bitstring:", bitstring)
print("Energy:", energy)