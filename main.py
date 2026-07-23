import csv
import os
import matplotlib.pyplot as plt
from encoding import bits_to_coords
from vqe import get_best_structure, run_vqe
from hamiltonian import ONE_LETTER_TO_FULL, path_energy
from real_structure import (
    get_ca_coords,
    normalize_coords,
    kabsch_align,
    rmsd,
    real_structure_to_bitstring
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_OUTPUT = os.path.join(RESULTS_DIR, "main_results.csv")

SEQUENCE = "GYDPETGTWG"
PDB_ID = "1UAO"
CHAIN_ID = "A"
SEED = 42
ALPHA = 0.5
REPETITIONS = 5000
OPTIMIZATION_STEPS = 200
LAYERS = 4
RESTARTS = 6
FINAL_REPETITIONS = 1000

CSV_FIELDS = (
    "Sequence",
    "Residues",
    "Qubits",
    "Seed",
    "Alpha",
    "Reps",
    "Opt Steps",
    "Evals",
    "Real Energy",
    "Real Bitstring",
    "VQE Energy",
    "VQE Bitstring",
    "Energy Diff",
    "RMSD",
)


def save_main_result(filename, result_row):
    """Append one completed main-program run to a consistently shaped CSV."""

    output_parent = os.path.dirname(filename)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0

    if not write_header:
        with open(filename, "r", encoding="utf-8", newline="") as csv_file:
            existing_header = next(csv.reader(csv_file), [])
        if existing_header != list(CSV_FIELDS):
            raise ValueError(
                f"Existing CSV header in {filename!r} does not match the "
                "current main.py result schema."
            )

    with open(filename, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(result_row)


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
        title += f" (Energy: {min_energy})"

    ax.set_title(title, pad = 20, fontsize = 12, fontweight = "bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()


def plot_real_structure(real_coords, sequence, pdb_id="7OFG"):
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

    sequence = SEQUENCE
    pdb_id = PDB_ID
    n_qubits = 2 * (len(sequence) - 1)

    result, history = run_vqe(
        sequence=sequence,
        alpha=ALPHA,
        repetitions=REPETITIONS,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
        layers=LAYERS,
        restarts=RESTARTS,
    )

    best_bitstring, best_coords, min_energy = get_best_structure(
        result,
        sequence,
        repetitions=FINAL_REPETITIONS,
        seed=SEED,
        layers=LAYERS,
    )

    pdb_path = os.path.join(BASE_DIR, "pdbs", f"{pdb_id}.pdb")
    real_coords_raw = get_ca_coords(pdb_path, chain_id=CHAIN_ID)
    real_bitstring = real_structure_to_bitstring(real_coords_raw)
    real_fitted_coords = bits_to_coords(real_bitstring)
    real_energy = path_energy(real_bitstring, sequence)

    #Output stuff
    print()
    print("========== REAL STRUCTURE FITTED TO LATTICE ==========")
    print("Fitted real structure bitstring:", real_bitstring)
    print("Fitted real structure energy:", real_energy)
    print("Fitted real structure coordinates:")

    for i, (res, coord) in enumerate(
        zip(sequence, real_fitted_coords)):
        print(f"Residue {i + 1} ({res}): {coord}")

    print()
    print("=======================================================")

    real_coords = normalize_coords(real_coords_raw)
    best_coords_norm = normalize_coords(best_coords)
    best_coords_aligned = kabsch_align(best_coords_norm, real_coords)
    fold_rmsd = rmsd(best_coords_aligned, real_coords)

    print()
    print("============== ENERGY COMPARISON ==============")

    print(f"VQE optimized structure energy: "f"{min_energy:.6f}")
    print(f"Fitted real structure energy: "f"{real_energy:.6f}")
    print(f"Energy difference: "f"{min_energy - real_energy:.6f}")

    print("===============================================")

    print("Optimal Bitstring:", best_bitstring)
    print("Lowest Energy:", min_energy)
    print(f"RMSD vs real structure ({pdb_id}): {fold_rmsd:.4f}")
    print("3D Coordinates:")

    for i, (res, coord) in enumerate(
        zip(sequence, best_coords)):
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

    # Model outputs
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fold_path = os.path.join(RESULTS_DIR, f"{sequence}_vqe_fold.png")
    real_path = os.path.join(RESULTS_DIR, f"{sequence}_real_{pdb_id}.png")
    plt.figure(1)
    plt.savefig(fold_path, dpi=150, bbox_inches="tight")
    plt.figure(2)
    plt.savefig(real_path, dpi=150, bbox_inches="tight")
    print(f"Plots saved to: {fold_path}")
    print(f"                {real_path}")

    #CSV Stuff
    save_main_result(
        CSV_OUTPUT,
        {
            "Sequence": sequence,
            "Residues": len(sequence),
            "Qubits": n_qubits,
            "Seed": SEED,
            "Alpha": ALPHA,
            "Reps": REPETITIONS,
            "Opt Steps": OPTIMIZATION_STEPS,
            "Evals": len(history),
            "Real Energy": real_energy,
            "Real Bitstring": real_bitstring,
            "VQE Energy": min_energy,
            "VQE Bitstring": best_bitstring,
            "Energy Diff": min_energy - real_energy,
            "RMSD": fold_rmsd,
        },
    )
    print(f"Results saved to: {CSV_OUTPUT}")

    if os.environ.get("SHOW_PLOTS", "1") != "0":
        plt.show()
