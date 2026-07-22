import numpy as np
from Bio.PDB import PDBParser


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

from Bio.PDB import PDBParser
parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "7OFG.pdb")
for model in structure:
    for chain in model:
        print(chain.id, len(chain))
    break

from real_structure import get_ca_coords

coords = get_ca_coords("7OFG.pdb", chain_id="A")
print(len(coords))