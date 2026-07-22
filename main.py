from vqe import run_vqe


sequence = "HP+-H"


result, history = run_vqe(
    sequence=sequence,
    alpha=0.5,
    repetitions=1000,
    optimization_steps=50
)


print("Finished")
print("Best CVaR:", result.fun)