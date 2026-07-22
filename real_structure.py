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
                if "CA" in residue:
                    coords.append(tuple(residue["CA"].coord))
        break

    return coords

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "6F3V.pdb")
for model in structure:
    for chain in model:
        print(chain.id, len(chain))
    break