import os
import numpy 
import matplotlib.pyplot as plt
from hamiltonian import path_energy
from encoding import bits_to_coords
from main import plot_protein, plot_real_structure
from real_structure import (
    extract_clean_ca_coords,
    kabsch_align,
    normalize_coords,
    rmsd,
    search_pdb_by_sequence,
    ensure_pdb_downloaded,
    real_structure_to_bitstring
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOCAL_PDB_MAP = {
    "GYDPETGTWG": ("1UAO", "A"),   
    "RPKPQQFFGLM": ("2KS9", "A"),  
    "CYIQNCPLG": ("7OFG", "A"), 
}

SEQUENCE = "RPKPQQFFGLM"  

PDB_ID = None
CHAIN_ID = "A"

if SEQUENCE in LOCAL_PDB_MAP:
    PDB_ID, CHAIN_ID = LOCAL_PDB_MAP[SEQUENCE]
    print(f"Found sequence in local map: {PDB_ID} (Chain {CHAIN_ID})")

else:
    print(f"Sequence not in local map. Querying RCSB API for '{SEQUENCE}'...")
    pdb_hits = search_pdb_by_sequence(SEQUENCE)
    if pdb_hits:
        PDB_ID = pdb_hits[0]
        print(f"Found matching PDB online: {PDB_ID}")
    else:
        print(f"No experimental structure found on RCSB for '{SEQUENCE}'.")

pdb_path = None
has_real_structure = False

if PDB_ID:
    pdb_path = os.path.join(BASE_DIR, "pdbs", f"{PDB_ID}.pdb")
    
    if not os.path.exists(pdb_path):
        pdb_path = ensure_pdb_downloaded(PDB_ID, os.path.join(BASE_DIR, "pdbs"))
        
    has_real_structure = pdb_path is not None and os.path.exists(pdb_path)

if not has_real_structure:
    print(f"\nWarning: PDB file for '{SEQUENCE}' not found at '{pdb_path}'.")
    print("Proceeding with Brute-Force ONLY (skipping pdb comp)\n")
n_turns = len(SEQUENCE)-1
n_qubits = 2*n_turns

#Brute force classical algorithm that searches through every single bitstring possibility
energy_list = []
fmt = f'0{n_qubits}b'
total = 2 ** n_qubits
print(total, " total searches commencing")
for idx in range(total):
    if idx % 50000 == 0:
        print(f"Checking fold {idx} / {total}...")
    bitstring = format(idx, fmt)
    energy = path_energy(bitstring, SEQUENCE)
    energy_list.append(energy)
energy_table = numpy.array(energy_list)
print('energy table built')

idx = energy_table.argmin()
lowest_energy = energy_table[idx]
best_bitstring = format(idx, fmt)
best_coords = bits_to_coords(best_bitstring)

if has_real_structure:
    real_coords_raw = extract_clean_ca_coords(
        pdb_path, chain_id=CHAIN_ID, expected_length=len(SEQUENCE)
    )
    
    real_coords = normalize_coords(real_coords_raw)
    best_coords_norm = normalize_coords(best_coords)
    
    best_coords_aligned = kabsch_align(best_coords_norm, real_coords)
    fold_rmsd = rmsd(best_coords_aligned, real_coords)

    real_bitstring = real_structure_to_bitstring(real_coords_raw)
    real_energy = path_energy(real_bitstring, SEQUENCE)

    print(f"Lowest Energy (Classical): {lowest_energy}")
    print(f"Real Structure Energy ({PDB_ID}): {real_energy}")
    print(f"RMSD vs Real ({PDB_ID}): {fold_rmsd:.4f}")

    plot_protein(
        best_coords_aligned,
        SEQUENCE,
        title=f"Classical Fold | E: {lowest_energy:.2f} (RMSD: {fold_rmsd:.3f})",
        min_energy=lowest_energy
    )
    plot_real_structure(
        real_coords,
        SEQUENCE,
        pdb_id=PDB_ID
    )
    plt.gca().set_title(f"Real Structure ({PDB_ID}) | Lattice E: {real_energy:.2f}")
else:
    print(f"Lowest Energy: {lowest_energy}")
    print(f"Current structure: {best_coords}, in bitstring format |{best_bitstring}>")

    plot_protein(
        best_coords,
        SEQUENCE,
        title=f"Classical Brute Force Fold for '{SEQUENCE}'",
        min_energy=lowest_energy
    )

plt.show()