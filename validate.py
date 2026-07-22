"""Single-sequence validation harness.

Runs the exhaustive brute-force search and the CVaR-VQE on the same small
sequence, using the identical energy model in ``hamiltonian.py``, and reports
whether the VQE recovers the true minimum-energy fold that brute force
guarantees. This is the first concrete quantum-vs-classical comparison and the
seed a length-sweep scalability study can later wrap in a loop.
"""

import time

import numpy as np

from encoding import bits_to_coords
from hamiltonian import path_energy
from vqe import run_vqe, best_fold_from_params


# Small enough that brute force is exact and the circuit is cheap to simulate:
# 5 residues -> 2 * (5 - 1) = 8 qubits (256 folds).
#
# Why 5 and not 4: in this FCC-style lattice a contact needs squared-distance 8,
# but a 4-residue chain's only non-consecutive pairs (0-2, 0-3, 1-3) can reach
# squared-distances 4, 11, or 12 -- never 8 (a parity constraint). So every
# 4-residue ground state is a trivial 0.0, which would not actually test whether
# the VQE finds a specific non-trivial fold. At 5 residues the end-to-end pair
# (0-4) can contact, giving a genuine negative minimum. "HPPPH" folds so its two
# hydrophobic (H) ends touch: an H-H contact worth -1.0.
SEQUENCE = "HPPPH"

# Fixed seed for reproducible parameter initialization and sampling.
SEED = 42

# VQE settings. alpha is the CVaR tail fraction; small alpha focuses the
# objective on the lowest-energy folds.
ALPHA = 0.1
REPETITIONS = 1000
OPTIMIZATION_STEPS = 100

# Tolerance for declaring the VQE energy a match to the brute-force minimum.
MATCH_TOLERANCE = 1e-6


def brute_force(sequence):
    """Enumerate every fold and return (min_energy, best_bitstring)."""

    n_qubits = 2 * (len(sequence) - 1)

    best_energy = None
    best_bitstring = None

    for idx in range(2 ** n_qubits):

        bitstring = format(idx, f"0{n_qubits}b")
        energy = path_energy(bitstring, sequence)

        if best_energy is None or energy < best_energy:

            best_energy = energy
            best_bitstring = bitstring

    return best_energy, best_bitstring


def main():

    n_qubits = 2 * (len(SEQUENCE) - 1)

    print(f"Sequence: {SEQUENCE}   ({len(SEQUENCE)} residues, {n_qubits} qubits)")
    print(f"Seed: {SEED}")
    print()

    # --- Brute force (ground truth) ---
    t0 = time.perf_counter()
    bf_energy, bf_bitstring = brute_force(SEQUENCE)
    bf_time = time.perf_counter() - t0

    print(
        f"Brute force:  min energy = {bf_energy:.2f}   "
        f"fold = |{bf_bitstring}>   ({bf_time:.2f} s)"
    )
    print(f"              coords = {bits_to_coords(bf_bitstring)}")

    # --- CVaR-VQE ---
    t0 = time.perf_counter()
    result, history = run_vqe(
        sequence=SEQUENCE,
        alpha=ALPHA,
        repetitions=REPETITIONS,
        optimization_steps=OPTIMIZATION_STEPS,
        seed=SEED,
    )
    vqe_energy, vqe_bitstring = best_fold_from_params(
        result.x,
        SEQUENCE,
        repetitions=REPETITIONS,
        seed=SEED,
    )
    vqe_time = time.perf_counter() - t0

    print(
        f"VQE (CVaR):   best energy = {vqe_energy:.2f}   "
        f"fold = |{vqe_bitstring}>   ({vqe_time:.2f} s)"
    )
    print(f"              final CVaR = {result.fun:.2f}")

    # --- Verdict ---
    gap = vqe_energy - bf_energy
    matched = abs(gap) <= MATCH_TOLERANCE

    print()
    print(f"Match: {'YES' if matched else 'NO'}  (gap = {gap:.2f})")


if __name__ == "__main__":
    main()
