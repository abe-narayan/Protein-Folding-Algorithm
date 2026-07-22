import matplotlib.pyplot as plt
from vqe import get_best_structure, run_vqe
from hamiltonian import ONE_LETTER_TO_FULL

sequence = "HP+-H--+-P+H"

result, history = run_vqe(
    sequence=sequence, alpha=0.5, repetitions=100, optimization_steps=30
)

best_bitstring, best_coords, min_energy = get_best_structure(
    result, sequence, repetitions=1000
)

print("Optimal Bitstring:", best_bitstring)
print("Lowest Energy:", min_energy)
print("3D Coordinates:")

for i, (res, coord) in enumerate(zip(sequence, best_coords)):
    print(f"Residue {i+1} ({res}): {coord}")


def plot_protein(coords, sequence):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    color_map = {}
    for aa in "AVLIMFWC":      # hydrophobic
        color_map[aa] = "red"
    for aa in "STNQGYP":       # polar/uncharged
        color_map[aa] = "blue"
    for aa in "KRH":           # positively charged
        color_map[aa] = "green"
    for aa in "DE":            # negatively charged
        color_map[aa] = "orange"

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]

    ax.plot(xs, ys, zs, color="gray", linestyle="--", linewidth=2)

    for i, (x, y, z) in enumerate(coords):
        res_type = sequence[i]
        c = color_map.get(res_type, "black")
        ax.scatter(x, y, z, color=c, s=120)
        full_name = ONE_LETTER_TO_FULL.get(res_type, res_type)
        ax.text(x, y, z + 0.2, f"{i+1}:{full_name}", fontsize=9)

    ax.set_title(f"3D Structure for '{sequence}' (Energy: {min_energy})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


plot_protein(best_coords, sequence)