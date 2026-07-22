import matplotlib.pyplot as plt

from vqe import get_best_structure, run_vqe
from hamiltonian import ONE_LETTER_TO_FULL
from real_structure import get_ca_coords, normalize_coords, kabsch_align


def plot_protein(coords, sequence, min_energy=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    color_map = {}

    for aa in "AVLIMFWC":
        color_map[aa] = "red"

    for aa in "STNQGYP":
        color_map[aa] = "blue"

    for aa in "KRH":
        color_map[aa] = "green"

    for aa in "DE":
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

        ax.text(
            x,
            y,
            z + 0.2,
            f"{i + 1}:{full_name}",
            fontsize=9
        )

    title = f"3D Structure for '{sequence}'"

    if min_energy is not None:
        title += f" (Energy: {min_energy})"

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()


def plot_real_structure(real_coords, sequence, pdb_id="2KS9"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    color_map = {}

    for aa in "AVLIMFWC":
        color_map[aa] = "red"

    for aa in "STNQGYP":
        color_map[aa] = "blue"

    for aa in "KRH":
        color_map[aa] = "green"

    for aa in "DE":
        color_map[aa] = "orange"

    rx = [c[0] for c in real_coords]
    ry = [c[1] for c in real_coords]
    rz = [c[2] for c in real_coords]

    ax.plot(rx, ry, rz, color="gray", linestyle="--", linewidth=2)

    for i, (x, y, z) in enumerate(real_coords):
        res_type = sequence[i]
        c = color_map.get(res_type, "black")

        ax.scatter(x, y, z, color=c, s=120)

        full_name = ONE_LETTER_TO_FULL.get(res_type, res_type)

        ax.text(
            x,
            y,
            z + 0.2,
            f"{i + 1}:{full_name}",
            fontsize=9
        )

    ax.set_title(f"Real Structure ({pdb_id}) for '{sequence}'")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()


if __name__ == "__main__":

    sequence = "RPKPQQFFGLM"

    result, history = run_vqe(
        sequence=sequence,
        alpha=0.5,
        repetitions=2500,
        optimization_steps=200
    )

    best_bitstring, best_coords, min_energy = get_best_structure(
        result,
        sequence,
        repetitions=1000
    )

    real_coords = get_ca_coords(
        "2KS9.pdb",
        chain_id="B"
    )

    real_coords = normalize_coords(real_coords)
    best_coords_norm = normalize_coords(best_coords)

    best_coords_aligned = kabsch_align(
        best_coords_norm,
        real_coords
    )

    print("Optimal Bitstring:", best_bitstring)
    print("Lowest Energy:", min_energy)
    print("3D Coordinates:")

    for i, (res, coord) in enumerate(
        zip(sequence, best_coords)
    ):
        print(f"Residue {i + 1} ({res}): {coord}")

    plot_protein(
        best_coords_aligned,
        sequence,
        min_energy
    )

    plot_real_structure(
        real_coords,
        sequence,
        pdb_id="2KS9"
    )

    plt.show()