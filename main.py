from vqe import run_vqe


sequence = "HP+-H--+-PPH+H"


result, history = run_vqe(
    sequence=sequence,
    alpha=0.5,
    repetitions=100,
    optimization_steps=10
)


print("Finished")
print("Best CVaR:", result.fun)