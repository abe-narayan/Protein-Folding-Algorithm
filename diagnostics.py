from encoding import bits_to_coords
from hamiltonian import path_energy
from real_structure import normalize_coords, kabsch_align, rmsd


def brute_force_min_energy(sequence, overlap_penalty=30.0, contact_weight=1.0):
    n_bonds = len(sequence) - 1
    n_qubits = 2 * n_bonds
    fmt = f"0{n_qubits}b"

    best_energy = float("inf")
    best_bitstring = None

    for idx in range(4 ** n_bonds):
        bitstring = format(idx, fmt)
        energy = path_energy(
            bitstring,
            sequence,
            overlap_penalty=overlap_penalty,
            contact_weight=contact_weight
        )
        if energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring

    return best_bitstring, best_energy


def brute_force_best_near_structure(
    sequence,
    real_coords,
    rmsd_threshold,
    overlap_penalty=30.0,
    contact_weight=1.0
):
    n_bonds = len(sequence) - 1
    n_qubits = 2 * n_bonds
    fmt = f"0{n_qubits}b"

    real_norm = normalize_coords(real_coords)

    best_energy = float("inf")
    best_bitstring = None
    best_rmsd = None

    for idx in range(4 ** n_bonds):
        bitstring = format(idx, fmt)
        coords = bits_to_coords(bitstring)
        coords_norm = normalize_coords(coords)
        aligned = kabsch_align(coords_norm, real_norm)
        r = rmsd(aligned, real_norm)

        if r > rmsd_threshold:
            continue

        energy = path_energy(
            bitstring,
            sequence,
            overlap_penalty=overlap_penalty,
            contact_weight=contact_weight
        )

        if energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring
            best_rmsd = r

    return best_bitstring, best_energy, best_rmsd