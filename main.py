import os

import matplotlib.pyplot as plt

from vqe import get_best_structure, run_vqe
from hamiltonian import ONE_LETTER_TO_FULL, find_disulfide_pairs
from real_structure import get_ca_coords, normalize_coords, kabsch_align, rmsd


def plot_protein(coords, sequence, title = None, min_energy=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    color_map = {
        **{aa: "red" for aa in "AVLIMFWC"},
        **{aa: "blue" for aa in "STNQGYP"},
        **{aa: "green" for aa in "KRH"},
        **{aa: "orange" for aa in "DE"}
    }

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

    if title is None:
        title = f"3D Structure for '{sequence}'"
    if min_energy is not None:
        # This is a relative Miyazawa-Jernigan lattice contact score, not a
        # physical free energy in kcal/mol -- label it as such so it is not read
        # as comparable to an experimental value.
        title += f" (Relative energy: {min_energy:.2f} MJ units)"

    ax.set_title(title, pad = 20, fontsize = 12, fontweight = "bold")
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

    sequence = "CYIQNCPLG"
    pdb_id = "7OFG"

    # Oxytocin is a cyclic peptide: its Cys1-Cys6 disulfide bond is the dominant
    # structural constraint and is present in the 7OFG reference (see its header,
    # "CYS-CYS DISULFIDE BOND"). path_energy applies this restraint by default
    # (inferred via find_disulfide_pairs), so the fold below is scored against
    # the realistic constrained model rather than a free, non-native chain.
    disulfides = find_disulfide_pairs(sequence)
    print(f"Disulfide restraint(s) applied: {disulfides or 'none'}")

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


    real_coords = get_ca_coords(f"{pdb_id}.pdb", chain_id="A")

    real_coords = normalize_coords(real_coords)
    best_coords_norm = normalize_coords(best_coords)

    best_coords_aligned = kabsch_align(
        best_coords_norm,
        real_coords
    )

    fold_rmsd = rmsd(best_coords_aligned, real_coords)

    print("Optimal Bitstring:", best_bitstring)
    print("Lowest Energy:", min_energy)
    print(f"RMSD vs real structure ({pdb_id}): {fold_rmsd:.4f}")
    print("3D Coordinates:")

    for i, (res, coord) in enumerate(
        zip(sequence, best_coords)
    ):
        print(f"Residue {i + 1} ({res}): {coord}")

    plot_protein(
        best_coords_aligned,
        sequence,
        title=f"VQE Fold for '{sequence}' (RMSD to {pdb_id}: {fold_rmsd:.3f})",
        min_energy=min_energy
    )

    plot_real_structure(
        real_coords,
        sequence,
        pdb_id=pdb_id
    )

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    fold_path = os.path.join(results_dir, f"{sequence}_vqe_fold.png")
    real_path = os.path.join(results_dir, f"{sequence}_real_{pdb_id}.png")
    plt.figure(1)
    plt.savefig(fold_path, dpi=150, bbox_inches="tight")
    plt.figure(2)
    plt.savefig(real_path, dpi=150, bbox_inches="tight")
    print(f"Plots saved to: {fold_path}")
    print(f"                {real_path}")

    if os.environ.get("SHOW_PLOTS", "1") != "0":
        plt.show()