from encoding import bits_to_coords


AMINO_ACIDS = [
    "CYSTEINE", "METHIONINE", "PHENYLALANINE", "ISOLEUCINE", "LEUCINE",
    "VALINE", "TRYPTOPHAN", "TYROSINE", "ALANINE", "GLYCINE",
    "THREONINE", "SERINE", "ASPARAGINE", "GLUTAMINE", "ASPARTATE",
    "GLUTAMATE", "HISTIDINE", "ARGININE", "LYSINE", "PROLINE",
]

ONE_LETTER_TO_FULL = {
    "C": "CYSTEINE", "M": "METHIONINE", "F": "PHENYLALANINE", "I": "ISOLEUCINE",
    "L": "LEUCINE", "V": "VALINE", "W": "TRYPTOPHAN", "Y": "TYROSINE",
    "A": "ALANINE", "G": "GLYCINE", "T": "THREONINE", "S": "SERINE",
    "N": "ASPARAGINE", "Q": "GLUTAMINE", "D": "ASPARTATE", "E": "GLUTAMATE",
    "H": "HISTIDINE", "R": "ARGININE", "K": "LYSINE", "P": "PROLINE",
}


_UPPER_TRIANGLE = [
    [-5.44, -4.99, -5.80, -5.50, -5.83, -4.96, -4.95, -4.16, -3.57, -3.16,
     -3.11, -2.86, -2.59, -2.85, -2.41, -2.27, -3.60, -2.57, -1.95, -3.07],
    [-4.99, -5.46, -6.56, -6.02, -6.41, -5.32, -5.55, -4.91, -3.94, -3.39,
     -3.51, -3.03, -2.95, -3.30, -2.57, -2.89, -3.98, -3.12, -2.48, -3.45],
    [-5.80, -6.56, -7.26, -6.84, -7.28, -6.29, -6.16, -5.66, -4.81, -4.13,
     -4.28, -4.02, -3.75, -4.10, -3.48, -3.56, -4.77, -3.98, -3.36, -4.25],
    [-5.50, -6.02, -6.84, -6.54, -7.04, -6.05, -5.78, -5.25, -4.58, -3.78,
     -4.03, -3.75, -3.52, -3.67, -3.17, -3.27, -4.14, -3.63, -3.01, -3.76],
    [-5.83, -6.41, -7.28, -7.04, -7.37, -6.48, -6.14, -5.67, -4.91, -4.16,
     -4.34, -4.08, -3.75, -4.04, -3.40, -3.59, -4.54, -4.03, -3.37, -4.20],
    [-4.96, -5.32, -6.29, -6.05, -6.48, -5.52, -5.18, -4.62, -4.04, -3.38,
     -3.46, -3.30, -3.07, -3.28, -2.83, -2.90, -3.58, -3.07, -2.49, -3.32],
    [-4.95, -5.55, -6.16, -5.78, -6.14, -5.18, -5.06, -4.66, -3.82, -3.42,
     -3.22, -3.07, -3.07, -3.11, -2.84, -2.99, -3.98, -3.41, -2.69, -3.73],
    [-4.16, -4.91, -5.66, -5.25, -5.67, -4.62, -4.66, -4.17, -3.36, -3.01,
     -3.01, -2.78, -2.83, -2.97, -2.76, -2.79, -3.52, -3.16, -2.60, -3.19],
    [-3.57, -3.94, -4.81, -4.58, -4.91, -4.04, -3.82, -3.36, -2.72, -2.31,
     -2.32, -2.01, -1.84, -1.89, -1.70, -1.51, -2.41, -1.83, -1.31, -2.03],
    [-3.16, -3.39, -4.13, -3.78, -4.16, -3.38, -3.42, -3.01, -2.31, -2.24,
     -2.08, -1.82, -1.74, -1.66, -1.59, -1.22, -2.15, -1.72, -1.15, -1.87],
    [-3.11, -3.51, -4.28, -4.03, -4.34, -3.46, -3.22, -3.01, -2.32, -2.08,
     -2.12, -1.96, -1.88, -1.90, -1.80, -1.74, -2.42, -1.90, -1.31, -1.90],
    [-2.86, -3.03, -4.02, -3.75, -4.08, -3.30, -3.07, -2.78, -2.01, -1.82,
     -1.96, -1.67, -1.58, -1.49, -1.63, -1.48, -2.11, -1.62, -1.05, -1.57],
    [-2.59, -2.95, -3.75, -3.52, -3.75, -3.07, -3.07, -2.83, -1.84, -1.74,
     -1.88, -1.58, -1.68, -1.71, -1.68, -1.51, -2.08, -1.64, -1.21, -1.53],
    [-2.85, -3.30, -4.10, -3.67, -4.04, -3.28, -3.11, -2.97, -1.89, -1.66,
     -1.90, -1.49, -1.71, -1.54, -1.46, -1.42, -1.98, -1.80, -1.29, -1.73],
    [-2.41, -2.57, -3.48, -3.17, -3.40, -2.83, -2.84, -2.76, -1.70, -1.59,
     -1.80, -1.63, -1.68, -1.46, -1.21, -1.02, -2.32, -2.29, -1.68, -1.33],
    [-2.27, -2.89, -3.56, -3.27, -3.59, -2.90, -2.99, -2.79, -1.51, -1.22,
     -1.74, -1.48, -1.51, -1.42, -1.02, -0.91, -2.15, -2.27, -1.80, -1.26],
    [-3.60, -3.98, -4.77, -4.14, -4.54, -3.58, -3.98, -3.52, -2.41, -2.15,
     -2.42, -2.11, -2.08, -1.98, -2.32, -2.15, -3.05, -2.16, -1.35, -2.25],
    [-2.57, -3.12, -3.98, -3.63, -4.03, -3.07, -3.41, -3.16, -1.83, -1.72,
     -1.90, -1.62, -1.64, -1.80, -2.29, -2.27, -2.16, -1.55, -0.59, -1.70],
    [-1.95, -2.48, -3.36, -3.01, -3.37, -2.49, -2.69, -2.60, -1.31, -1.15,
     -1.31, -1.05, -1.21, -1.29, -1.68, -1.80, -1.35, -0.59, -0.12, -0.97],
    [-3.07, -3.45, -4.25, -3.76, -4.20, -3.32, -3.73, -3.19, -2.03, -1.87,
     -1.90, -1.57, -1.53, -1.73, -1.33, -1.26, -2.25, -1.70, -0.97, -1.75],
]

def _build_full_matrix():
    full = {}
    for i, aa_i in enumerate(AMINO_ACIDS):
        for j, aa_j in enumerate(AMINO_ACIDS):
            val = _UPPER_TRIANGLE[i][j] if j >= i else _UPPER_TRIANGLE[j][i]
            full[(aa_i, aa_j)] = val
    return full

MJ_MATRIX = _build_full_matrix()

_MAX_PAIR_ENERGY = max(abs(v) for v in MJ_MATRIX.values())


def get_interaction(aa1: str, aa2: str) -> float:
    a1 = ONE_LETTER_TO_FULL.get(aa1.strip().upper(), aa1.strip().upper())
    a2 = ONE_LETTER_TO_FULL.get(aa2.strip().upper(), aa2.strip().upper())
    return MJ_MATRIX[(a1, a2)]


def find_disulfide_pairs(sequence):
    """Infer disulfide-bonded cysteine pairs from a sequence.

    Disulfide bonds are covalent Cys-Cys links that are often the dominant
    structural constraint in a small peptide (they cyclize it). We can't read
    the true bonding topology from sequence alone, so we use the common,
    well-defined case: a peptide with exactly two cysteines is assumed to form
    a single disulfide between them. Oxytocin ("CYIQNCPLG") is exactly this --
    Cys1-Cys6 -- and 7OFG's header confirms the "CYS-CYS DISULFIDE BOND".

    Returns a list of (i, j) index pairs, or an empty list when the topology is
    ambiguous (zero, one, or three+ cysteines), in which case no constraint is
    applied rather than guessing.
    """
    cys_positions = [i for i, aa in enumerate(sequence) if aa.strip().upper() in ("C", "CYSTEINE")]
    if len(cys_positions) == 2:
        return [(cys_positions[0], cys_positions[1])]
    return []


# On this lattice the nearest non-bonded shell is squared-distance 8; a disulfide
# is a covalent bond, so bonded cysteines must sit within that shell or closer.
DISULFIDE_BOND_D2 = 8


def path_energy_specific(
    bitstring,
    sequence,
    overlap_penalty=None,
    contact_weight=1.0,
    compactness_weight=0.5,
    disulfide_pairs=None,
    disulfide_weight=0.5,
    disulfide_hard=False,
):
    coords = bits_to_coords(bitstring)
    n_residues = len(sequence)

    if overlap_penalty is None:
        n_pairs = n_residues * (n_residues - 1) // 2
        overlap_penalty = contact_weight * _MAX_PAIR_ENERGY * n_pairs + 1.0

    energy = 0.0

    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            if coords[i] == coords[j]:
                energy += overlap_penalty

    for i in range(n_residues):
        for j in range(i + 2, n_residues):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]

            if dx * dx + dy * dy + dz * dz == 8:
                energy += contact_weight * get_interaction(
                    sequence[i],
                    sequence[j]
                )

    # Disulfide bonds are covalent Cys-Cys links that cyclize the peptide. A
    # covalent bond is not optional, so the principled model is a *hard*
    # topological constraint: folds that leave a bonded pair outside the bond
    # shell are forbidden (given an overlap-scale penalty), which restricts the
    # search to the native cyclic class. Empirically this enriches native-like
    # conformations at the ensemble level (see analysis.disulfide_effect). The
    # softer distance restraint (disulfide_weight) is retained for comparison but
    # is off by default, since a distance-graded pull is not physically stronger
    # than the bond simply existing.
    if disulfide_pairs:
        for i, j in disulfide_pairs:
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dz = coords[i][2] - coords[j][2]
            d2 = dx * dx + dy * dy + dz * dz
            if disulfide_hard:
                if d2 > DISULFIDE_BOND_D2:
                    energy += overlap_penalty
            else:
                energy += disulfide_weight * d2

    if compactness_weight:
        cx = sum(c[0] for c in coords) / n_residues
        cy = sum(c[1] for c in coords) / n_residues
        cz = sum(c[2] for c in coords) / n_residues

        rg_squared = sum(
            (c[0] - cx) ** 2
            + (c[1] - cy) ** 2
            + (c[2] - cz) ** 2
            for c in coords
        ) / n_residues

        energy += compactness_weight * rg_squared

    return energy


def path_energy(bitstring, sequence, overlap_penalty=30.0, use_disulfide=True):
    # Disulfide bonds are inferred from the sequence (see find_disulfide_pairs)
    # and applied by default as a hard topological constraint, so the whole
    # pipeline -- brute force, VQE, validation -- searches the same physically
    # constrained fold space. Set use_disulfide=False for the unconstrained
    # baseline. NOTE: for solvent-exposed peptides the MJ contact energy is
    # anti-correlated with native similarity (see analysis.predictive_validity);
    # the disulfide constraint helps at the ensemble level, but a single
    # energy-minimum fold should not be reported as a structure prediction.
    disulfide_pairs = find_disulfide_pairs(sequence) if use_disulfide else None
    return path_energy_specific(
        bitstring,
        sequence,
        overlap_penalty=overlap_penalty,
        disulfide_pairs=disulfide_pairs,
        disulfide_hard=True,
    )