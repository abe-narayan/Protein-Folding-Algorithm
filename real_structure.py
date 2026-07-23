import numpy as np
import random
import math
import os

from Bio.PDB import PDBParser, PDBList
from encoding import bits_to_coords
from rcsbapi.search import SeqSimilarityQuery

def search_pdb_by_sequence(sequence):
    try:
        query = SeqSimilarityQuery(
            value=sequence,
            sequence_type="protein",
            evalue_cutoff=10.0,
            identity_cutoff=0.8
        )
        results = list(query())
        return results if results else []
    except Exception as e:
        print(f"RCSB Search API error: {e}")
        return []

def ensure_pdb_downloaded(pdb_id, pdb_dir):
    os.makedirs(pdb_dir, exist_ok=True)
    pdb_path = os.path.join(pdb_dir, f"{pdb_id.upper()}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    
    pdbl = PDBList(quiet=True)
    fetched_file = pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_dir, file_format="pdb")
    
    if fetched_file and os.path.exists(fetched_file):
        os.rename(fetched_file, pdb_path)
        return pdb_path
    return None

def extract_clean_ca_coords(pdb_path, chain_id="A", expected_length=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    model = structure[0]

    if chain_id in model:
        chain = model[chain_id]
    else:
        chain = next(iter(model.get_chains()))

    ca_coords = []
    for residue in chain:
        if residue.id[0] == " " and "CA" in residue:
            ca_coords.append(residue["CA"].get_coord())
            if expected_length and len(ca_coords) == expected_length:
                break

    return np.array(ca_coords)

def get_ca_coords(pdb_path, chain_id=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coords = []
    for model in structure:
        for chain in model:
            if chain_id and chain.id != chain_id:
                continue
            for residue in chain:
                if "CA" in residue and residue.id[0] == " ":  # standard residues only
                    coords.append(tuple(residue["CA"].coord))
        break

    return coords


def normalize_coords(coords):
    coords = np.array(coords)
    coords = coords - coords.mean(axis=0)
    max_extent = np.max(np.linalg.norm(coords, axis=1))
    if max_extent > 0:
        coords = coords / max_extent
    return [tuple(c) for c in coords]


def kabsch_align(coords_to_rotate, reference_coords):
    A = np.array(coords_to_rotate)
    B = np.array(reference_coords)

    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    rotated = (R @ A.T).T
    return [tuple(p) for p in rotated]


def rmsd(coords_a, coords_b):
    """Root-mean-square deviation between two equal-length coordinate sets.

    Assumes the coordinates are already superimposed (e.g. via ``kabsch_align``).
    """
    A = np.array(coords_a)
    B = np.array(coords_b)
    if A.shape != B.shape:
        raise ValueError(f"coordinate sets differ in shape: {A.shape} vs {B.shape}")
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))

def fit_lattice_to_real(
    lattice_coords,
    real_coords,
    overlap_penalty=30.0
):

    lattice = np.array(
        lattice_coords,
        dtype=np.float64
    )

    real = np.array(
        real_coords,
        dtype=np.float64
    )

    overlaps = 0

    for i in range(len(lattice)):

        for j in range(i + 1, len(lattice)):

            if np.array_equal(
                lattice[i],
                lattice[j]
            ):

                overlaps += 1

    lattice_centered = (
        lattice - lattice.mean(axis=0)
    )

    real_centered = (
        real - real.mean(axis=0)
    )

    real_bond_lengths = np.linalg.norm(
        np.diff(real, axis=0),
        axis=1
    )

    average_real_bond_length = np.mean(
        real_bond_lengths
    )

    lattice_bond_length = np.linalg.norm(
        lattice[1] - lattice[0]
    )

    scale = (
        average_real_bond_length
        / lattice_bond_length
    )

    lattice_scaled = (
        lattice_centered * scale
    )

    H = lattice_scaled.T @ real_centered

    U, S, Vt = np.linalg.svd(H)

    d = np.sign(
        np.linalg.det(
            Vt.T @ U.T
        )
    )

    D = np.diag(
        [1, 1, d]
    )

    R = Vt.T @ D @ U.T

    aligned = (
        R @ lattice_scaled.T
    ).T

    geometric_rmsd = np.sqrt(
        np.mean(
            np.sum(
                (aligned - real_centered) ** 2,
                axis=1
            )
        )
    )

    total_score = (
        geometric_rmsd
        + overlap_penalty * overlaps
    )

    return aligned, total_score

def real_structure_to_bitstring(
    coords,
    iterations=50000,
    temperature=1.0,
    cooling_rate=0.9998,
    seed=42
):

    random.seed(seed)

    coords = np.array(coords, dtype=np.float64)

    n_bonds = len(coords) - 1

    current_bits = [
        random.randint(0, 3)
        for _ in range(n_bonds)
    ]

    def bits_to_bitstring(bits):

        return "".join(
            format(direction, "02b")
            for direction in bits
        )

    current_bitstring = bits_to_bitstring(
        current_bits
    )

    current_coords = bits_to_coords(
        current_bitstring
    )

    _, current_score = fit_lattice_to_real(
        current_coords,
        coords
    )

    best_bits = current_bits.copy()
    best_score = current_score

    current_temperature = temperature

    for iteration in range(iterations):

        new_bits = current_bits.copy()

        bond_index = random.randrange(
            n_bonds
        )

        new_direction = random.randint(0, 3)

        new_bits[bond_index] = new_direction

        new_bitstring = bits_to_bitstring(
            new_bits
        )

        new_coords = bits_to_coords(
            new_bitstring
        )

        _, new_score = fit_lattice_to_real(
            new_coords,
            coords
        )

        difference = new_score - current_score
        if (
            difference < 0
            or random.random()
            < math.exp(
                -difference / current_temperature
            )
        ):

            current_bits = new_bits
            current_score = new_score

        if current_score < best_score:

            best_bits = current_bits.copy()
            best_score = current_score

        current_temperature *= cooling_rate

        if current_temperature < 0.0001:

            current_temperature = 0.0001

    best_bitstring = bits_to_bitstring(
        best_bits
    )

    print(
        f"Closest tetrahedral lattice fitting score: "
        f"{best_score:.4f}"
    )

    return best_bitstring