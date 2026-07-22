from vqe import run_vqe


sequence = "HP+-HHP+-HHP+-H"


result, history = run_vqe(
    sequence=sequence,
    alpha=0.1,
    repetitions=1000,
    optimization_steps=100
)


print()
print("Finished!")
print()

print("Sequence:")
print(sequence)

print()

print("Number of amino acids:")
print(len(sequence))

print()

print("Number of qubits:")
print(2 * (len(sequence) - 1))

print()

print("Best CVaR:")
print(result.fun)

print()

print("Best parameters:")
print(result.x)