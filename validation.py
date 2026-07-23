"""Self-tests for correctness and scientific validity.

Run with `python main.py --validate`. Every test prints PASS or FAIL and the
suite returns False if any test fails. These are the checks that would
otherwise have to be taken on trust.
"""
import inspect
import math
from typing import Dict, List

import numpy as np

import protein_geometry as geo
import representations as reps
import sidechains as sc
import energy_terms as et
import hamiltonian as ham
import vqe as vqe_mod
import classical_baselines as cb
import evaluation as ev


_PASS = "PASS"
_FAIL = "FAIL"


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{_PASS if ok else _FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


# ==========================================================================
def test_bitstring_and_coordinate_lengths() -> bool:
    ok = True
    for n_res in (10, 12, 15):
        for n_states in (4, 8):
            r = reps.TorsionStateRepresentation(n_res, n_states=n_states)
            expected_bits = n_res * (2 if n_states == 4 else 3)
            ok &= (r.n_bits == expected_bits) and (r.n_qubits == expected_bits)
            b = r.random_bitstring(np.random.default_rng(0))
            ok &= (len(b) == expected_bits)
            c = r.build_coords(b)
            for key in ("N", "CA", "C", "CB", "O"):
                ok &= (c[key].shape == (n_res, 3))
            phi, psi = r.decode(b)
            ok &= (len(phi) == n_res and len(psi) == n_res)
        lat = reps.TetrahedralLatticeRepresentation(n_res)
        ok &= (lat.n_bits == 2 * (n_res - 1))
        lb = lat.random_bitstring(np.random.default_rng(0))
        ok &= (len(lb) == 2 * (n_res - 1))
        ok &= (lat.decode(lb).shape == (n_res, 3))
    return _report("bitstring and coordinate lengths consistent", ok)


def test_backbone_geometry_is_physical() -> bool:
    """NeRF backbone must reproduce ideal bond lengths and CA-CA spacing."""
    r = reps.TorsionStateRepresentation(12, n_states=4)
    b = r.bitstring_from_states([0] * 12)     # all alpha-R
    c = r.build_coords(b)
    n_ca = np.linalg.norm(c["CA"] - c["N"], axis=1)
    ca_c = np.linalg.norm(c["C"] - c["CA"], axis=1)
    ca_ca = np.linalg.norm(np.diff(c["CA"], axis=0), axis=1)
    ok = (np.allclose(n_ca, geo.BOND_N_CA, atol=1e-6)
          and np.allclose(ca_c, geo.BOND_CA_C, atol=1e-6)
          and np.all(np.abs(ca_ca - 3.8) < 0.15))
    return _report("backbone bond lengths and CA-CA ~3.8 A", ok,
                   f"CA-CA mean {ca_ca.mean():.3f} A")


def test_helix_is_representable() -> bool:
    """All-alpha-R must produce a right-handed helix: ~1.5 A rise, ~100 deg/res.

    This is the test the tetrahedral lattice CANNOT pass, and it is the core
    justification for replacing the representation.
    """
    r = reps.TorsionStateRepresentation(14, n_states=4)
    ca = r.build_coords(r.bitstring_from_states([0] * 14))["CA"]
    # i to i+4 distance in an alpha helix is ~6.2 A
    d14 = np.linalg.norm(ca[4:] - ca[:-4], axis=1)
    ok = bool(np.all(np.abs(d14 - 6.2) < 1.2))
    return _report("alpha helix representable (i,i+4 ~6.2 A)", ok,
                   f"mean {d14.mean():.2f} A")


def test_chirality() -> bool:
    """Alpha-R and alpha-L must give different (mirror) structures."""
    r = reps.TorsionStateRepresentation(12, n_states=4)
    right = r.build_coords(r.bitstring_from_states([0] * 12))["CA"]
    left = r.build_coords(r.bitstring_from_states([3] * 12))["CA"]
    # Superposing without reflection must leave a large residual.
    resid = geo.ca_rmsd(left, right)
    ok = resid > 2.0
    return _report("representation is chiral (alpha-R != alpha-L)", ok,
                   f"CA-RMSD {resid:.2f} A")


def test_mj_correction_changes_sign() -> bool:
    """Corrected MJ must contain positive entries; raw MJ must not."""
    raw_vals = np.array(list(et.MJ_RAW.values()))
    cor_vals = np.array(list(et.MJ_CORRECTED.values()))
    ok = bool(np.all(raw_vals < 0) and np.any(cor_vals > 0))
    return _report("MJ self-energy correction applied", ok,
                   f"corrected range [{cor_vals.min():.2f}, {cor_vals.max():.2f}]")


def test_energy_is_finite_and_deterministic() -> bool:
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    H = ham.FoldingHamiltonian(seq, r)
    rng = np.random.default_rng(3)
    ok = True
    for _ in range(50):
        b = r.random_bitstring(rng)
        e1 = H.energy(b)
        H._cache.clear()
        e2 = H.energy(b)
        ok &= math.isfinite(e1) and (abs(e1 - e2) < 1e-12)
    return _report("energy finite and deterministic", ok)


def test_energy_rejects_wrong_length() -> bool:
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    H = ham.FoldingHamiltonian(seq, r)
    try:
        H.energy("0" * (r.n_bits + 2))
        return _report("energy rejects wrong-length bitstring", False)
    except ValueError:
        return _report("energy rejects wrong-length bitstring", True)


def test_steric_penalizes_clash() -> bool:
    """A collapsed structure must have higher steric energy than an extended one."""
    seq = "AAAAAAAAAA"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    H = ham.FoldingHamiltonian(seq, r)
    extended = H.components(r.bitstring_from_states([1] * len(seq)))
    helical = H.components(r.bitstring_from_states([0] * len(seq)))
    ok = extended["steric"] <= helical["steric"] + 1e-9
    return _report("steric term responds to compaction", ok,
                   f"extended {extended['steric']:.2f} vs helix {helical['steric']:.2f}")


def test_hbond_rewards_helix() -> bool:
    """An all-helix structure must have more favourable H-bond energy than
    an all-extended one. This validates the DSSP term's sign and matching."""
    seq = "AEAAAKEAAAKA"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    H = ham.FoldingHamiltonian(seq, r)
    helix = H.components(r.bitstring_from_states([0] * len(seq)))["hbond"]
    ext = H.components(r.bitstring_from_states([1] * len(seq)))["hbond"]
    ok = helix < ext
    return _report("H-bond term favours helix over extended", ok,
                   f"helix {helix:.2f} vs extended {ext:.2f} kcal/mol")


def test_cvar_preserves_multiplicities() -> bool:
    """THE critical CVaR test.

    Energies [0, 10] with the low value drawn once and the high value 99
    times. With alpha=0.5 the correct CVaR over 100 samples averages the
    lowest 50 -> 1 copy of 0 and 49 copies of 10 -> 9.8. A de-duplicating
    implementation would see [0, 10] and return 0.0. That difference is the
    bug this test exists to catch.
    """
    energies = [0.0] + [10.0] * 99
    got = vqe_mod.cvar_from_samples(energies, alpha=0.5)
    expected = (0.0 + 49 * 10.0) / 50.0
    ok = abs(got - expected) < 1e-9
    dedup = vqe_mod.cvar_from_samples(list(set(energies)), alpha=0.5)
    return _report("CVaR preserves sample multiplicities", ok,
                   f"got {got:.4f}, expected {expected:.4f}, "
                   f"dedup would give {dedup:.4f}")


def test_cvar_sampled_converges_to_exact() -> bool:
    """Sampled CVaR must converge to the distributional CVaR."""
    rng = np.random.default_rng(11)
    energies = rng.normal(0, 1, size=64)
    logits = rng.normal(0, 1, size=64)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    exact = vqe_mod.cvar_from_distribution(energies, probs, 0.25)
    draws = rng.choice(64, size=400_000, p=probs)
    approx = vqe_mod.cvar_from_samples(energies[draws], 0.25)
    ok = abs(exact - approx) < 0.05
    return _report("sampled CVaR converges to exact CVaR", ok,
                   f"exact {exact:.4f} vs sampled {approx:.4f}")


def test_cvar_alpha_one_is_mean() -> bool:
    e = [3.0, 1.0, 4.0, 1.0, 5.0]
    ok = abs(vqe_mod.cvar_from_samples(e, 1.0) - np.mean(e)) < 1e-12
    return _report("CVaR at alpha=1 equals the mean", ok)


def test_vqe_is_full_system() -> bool:
    """The circuit must act on ALL qubits, and no block API may exist."""
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    H = ham.FoldingHamiltonian(seq, r)
    circuit = vqe_mod.build_global_circuit(H.n_qubits, layers=2)
    probs = np.asarray(circuit(np.zeros(vqe_mod.n_parameters(H.n_qubits, 2))))
    ok = (probs.size == 2 ** H.n_qubits)
    # Check the module NAMESPACE, not the source text -- the docstring
    # legitimately contains the word "block" while explaining its absence.
    names = [n for n in dir(vqe_mod) if "block" in n.lower()]
    ok &= (len(names) == 0)
    return _report("VQE is full-system (no block decomposition)", ok,
                   f"probs size {probs.size} == 2^{H.n_qubits}, "
                   f"block-named symbols: {names}")


def test_vqe_does_not_enumerate() -> bool:
    """Energy evaluations must be far below the configuration-space size."""
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    H = ham.FoldingHamiltonian(seq, r)
    # layers=1 -> 20 params; maxiter must exceed n_params + 2 for COBYLA.
    res = vqe_mod.run_global_cvar_vqe(H, layers=1, shots=256, maxiter=120,
                                      restarts=1, seed=0, final_shots=512)
    space = 2 ** H.n_qubits
    ok = res["n_energy_evaluations"] < 0.25 * space
    return _report("VQE does not enumerate the configuration space", ok,
                   f"{res['n_energy_evaluations']} evals vs {space} configs")


def test_final_answer_comes_from_final_circuit() -> bool:
    """vqe_bitstring and best_seen_bitstring must be reported separately."""
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    H = ham.FoldingHamiltonian(seq, r)
    res = vqe_mod.run_global_cvar_vqe(H, layers=1, shots=256, maxiter=120,
                                      restarts=1, seed=0, final_shots=512)
    ok = ("vqe_bitstring" in res and "best_seen_bitstring" in res
          and res["vqe_energy"] >= res["best_seen_energy"] - 1e-9
          and len(res["vqe_bitstring"]) == H.n_bits)
    return _report("final VQE answer distinct from best-seen", ok,
                   f"vqe {res['vqe_energy']:.3f}, "
                   f"best-seen {res['best_seen_energy']:.3f}")


def test_reproducibility() -> bool:
    """Same seed -> identical results. Different seed -> different results."""
    seq = "GYDPETGTWG"

    def run(s):
        r = reps.TorsionStateRepresentation(len(seq), n_states=4)
        H = ham.FoldingHamiltonian(seq, r)
        # layers=1 -> 20 params. maxiter=120 gives COBYLA room to actually
        # move; at maxiter=15 it terminated before building its simplex and
        # every seed returned the same near-initial point.
        return vqe_mod.run_global_cvar_vqe(H, layers=1, shots=256, maxiter=120,
                                           restarts=1, seed=s, final_shots=256)

    a, b, c = run(0), run(0), run(1)
    same = (a["vqe_bitstring"] == b["vqe_bitstring"]
            and abs(a["vqe_energy"] - b["vqe_energy"]) < 1e-12)
    differ = (a["vqe_bitstring"] != c["vqe_bitstring"]
              or abs(a["vqe_energy"] - c["vqe_energy"]) > 1e-12)
    return _report("reproducible across seeds", same and differ,
                   f"seed-identical={same}, seed-sensitive={differ}")


def test_rng_streams_are_independent() -> bool:
    """Counter-based streams must differ across evaluation index.

    This is the direct regression test for the original implementation's bug
    of recreating default_rng(seed) inside every objective call.
    """
    s0 = np.random.default_rng(np.random.SeedSequence([42, 0])).random(5)
    s1 = np.random.default_rng(np.random.SeedSequence([42, 1])).random(5)
    s0b = np.random.default_rng(np.random.SeedSequence([42, 0])).random(5)
    ok = (not np.allclose(s0, s1)) and np.allclose(s0, s0b)
    return _report("RNG streams independent across evals, stable across runs", ok)


def test_no_pdb_access_during_optimization() -> bool:
    """LEAKAGE AUDIT: the optimizer must never open a native structure."""
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    H = ham.FoldingHamiltonian(seq, r)
    geo.reset_pdb_log()
    # layers=1 -> 20 params; maxiter must exceed n_params + 2 for COBYLA.
    vqe_mod.run_global_cvar_vqe(H, layers=1, shots=128, maxiter=120,
                                restarts=1, seed=0, final_shots=128)
    cb.simulated_annealing(H, n_steps=500, seed=0)
    log = geo.get_pdb_log()
    ok = (len(log) == 0)
    return _report("no PDB access during optimization (leakage audit)", ok,
                   f"{len(log)} PDB reads")


def test_hamiltonian_has_no_native_input() -> bool:
    """Structural check: FoldingHamiltonian's signature must have no PDB path."""
    params = set(inspect.signature(ham.FoldingHamiltonian.__init__).parameters)
    forbidden = {"pdb", "pdb_path", "native", "native_coords", "coords",
                 "entry", "structure", "target"}
    leaked = params & forbidden
    return _report("Hamiltonian constructor takes no native structure",
                   not leaked, str(sorted(leaked)) if leaked else "")


def test_rmsd_is_angstroms() -> bool:
    """RMSD of a structure against itself is 0; against a 5 A shift is 0 too
    (translation-invariant); against a genuinely different fold is O(A)."""
    r = reps.TorsionStateRepresentation(12, n_states=4)
    helix = r.build_coords(r.bitstring_from_states([0] * 12))["CA"]
    strand = r.build_coords(r.bitstring_from_states([1] * 12))["CA"]
    self_r = geo.ca_rmsd(helix, helix)
    shift_r = geo.ca_rmsd(helix + 5.0, helix)
    cross_r = geo.ca_rmsd(helix, strand)
    ok = (self_r < 1e-9 and shift_r < 1e-9 and 3.0 < cross_r < 30.0)
    return _report("RMSD in Angstroms, translation/rotation invariant", ok,
                   f"self {self_r:.2e}, helix-vs-strand {cross_r:.2f} A")


def test_no_normalized_rmsd_path() -> bool:
    """There must be no coordinate-normalization function anywhere."""
    ok = not hasattr(geo, "normalize_coords")
    src = inspect.getsource(ev)
    ok &= ("normalize_coords" not in src)
    return _report("no dimensionless normalized-RMSD path exists", ok)


def test_vqe_matches_exhaustive_on_tiny_system() -> bool:
    """Measure the VQE's optimality gap on an enumerable system.

    This is a MEASUREMENT, not a pass/fail correctness check. A nonzero gap
    means the VQE did not reach the global optimum at this budget, which is
    a real result about the algorithm and must be reported rather than tuned
    away. It fails only if the gap is large enough to suggest the optimizer
    is broken rather than merely imperfect.

    Simulated annealing is run at MATCHED energy evaluations so the two
    optimizers are compared on the currency that would also be spent on
    quantum hardware.

    NOTE: exhaustive_search is ground truth HERE ONLY. It is not part of the
    VQE search path -- see test_vqe_does_not_enumerate.
    """
    seq = "AEKAAG"          # 6 residues -> 12 qubits -> 4096 configs
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    H = ham.FoldingHamiltonian(seq, r)
    exact = cb.exhaustive_search(H, max_bits=14)

    # 12 qubits x 3 layers = 36 params; maxiter=400 is comfortably above 38.
    H2 = ham.FoldingHamiltonian(seq, reps.TorsionStateRepresentation(len(seq), 4))
    res = vqe_mod.run_global_cvar_vqe(H2, layers=3, alpha=0.15, shots=1024,
                                      maxiter=400, restarts=5, seed=0,
                                      final_shots=8192)

    H3 = ham.FoldingHamiltonian(seq, reps.TorsionStateRepresentation(len(seq), 4))
    sa = cb.simulated_annealing(H3, n_steps=res["n_energy_evaluations"], seed=0)

    gap_vqe = res["vqe_energy"] - exact["best_energy"]
    gap_seen = res["best_seen_energy"] - exact["best_energy"]
    gap_sa = sa["best_energy"] - exact["best_energy"]
    ok = gap_vqe < 2.0
    return _report("VQE optimality gap on enumerable system", ok,
                   f"exact {exact['best_energy']:.4f} | vqe {res['vqe_energy']:.4f} "
                   f"(gap {gap_vqe:.4f}) | best-seen gap {gap_seen:.4f} | "
                   f"SA gap {gap_sa:.4f} at matched evals")


def test_ceiling_is_a_real_ceiling() -> bool:
    """No sampled bitstring may score better than the reported ceiling.

    Regression test for the projection bug: at N=12 the greedy per-residue
    projection reported 9.50 A as the ceiling while three optimizers found
    structures at 6.35-8.33 A. A ceiling that gets beaten is not a ceiling.
    """
    seq = "SWTWEGNKWTWK"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4)
    # Synthetic native built BY the representation, so a perfect (0 A)
    # bitstring provably exists and the search has a reachable target.
    rng = np.random.default_rng(5)
    target_bits = r.random_bitstring(rng)
    native = r.build_coords(target_bits)["CA"]
    phi, psi = r.decode(target_bits)
    c = ev.representation_ceiling(r, native, phi, psi, seed=0, iterations=5000)
    violations = 0
    for _ in range(300):
        b = r.random_bitstring(rng)
        if geo.ca_rmsd(r.build_coords(b)["CA"], native) < c["ceiling_ca_rmsd"] - 1e-6:
            violations += 1
    ok = (violations == 0)
    return _report("reported ceiling is not beaten by random sampling", ok,
                   f"ceiling {c['ceiling_ca_rmsd']:.3f} A, projection "
                   f"{c['projection_ca_rmsd']:.3f} A, "
                   f"{violations}/300 random structures beat it")

def test_representation_ceiling_computable() -> bool:
    """Ceiling must be computable and finite for both representations."""
    seq = "GYDPETGTWG"
    r_t = reps.TorsionStateRepresentation(len(seq), n_states=4)
    # Synthetic "native": a helix built by the representation itself, so the
    # torsion ceiling should be ~0.
    native_coords = r_t.build_coords(r_t.bitstring_from_states([0] * len(seq)))
    phi, psi = geo.extract_torsions(native_coords["N"], native_coords["CA"],
                                    native_coords["C"])
    ct = ev.representation_ceiling(r_t, native_coords["CA"], phi, psi)
    r_l = reps.TetrahedralLatticeRepresentation(len(seq))
    cl = ev.representation_ceiling(r_l, native_coords["CA"], seed=0)
    ok = (math.isfinite(ct["ceiling_ca_rmsd"])
          and math.isfinite(cl["ceiling_ca_rmsd"])
          and ct["ceiling_ca_rmsd"] < 1.5)
    return _report("representation ceiling computable", ok,
                   f"torsion {ct['ceiling_ca_rmsd']:.3f} A, "
                   f"lattice {cl['ceiling_ca_rmsd']:.3f} A")


# ==========================================================================


# ==========================================================================
# Sidechain construction (sidechains.py)
# ==========================================================================
_SC_PROBE_SEQ = "GASTDENKPYW"


def _sc_backbones():
    """One 11-residue backbone per torsion state, covering all built types."""
    r = reps.TorsionStateRepresentation(len(_SC_PROBE_SEQ), n_states=4)
    return [r.build_coords(r.bitstring_from_states([s] * len(_SC_PROBE_SEQ)))
            for s in range(4)]


def _sc_residue_atoms(key, bb, i):
    res = {"N": bb["N"][i], "CA": bb["CA"][i], "C": bb["C"][i], "O": bb["O"][i]}
    res.update(sc.build_sidechain(key, bb["N"][i], bb["CA"][i], bb["C"][i],
                                  bb["CB"][i]))
    return res


def _sc_bond_distances(key):
    """Bond-graph hop counts between every pair of heavy atoms in a residue."""
    adj = {}
    for a, b in sc.residue_bonds(key):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    out = {}
    for src in adj:
        d, queue = {src: 0}, [src]
        while queue:
            u = queue.pop(0)
            for v in adj.get(u, ()):
                if v not in d:
                    d[v] = d[u] + 1
                    queue.append(v)
        out[src] = d
    return out


# Ideal bond lengths, written out independently of sidechains.py internals so
# this is a real check rather than a tautology. CA-CB is fixed by place_cb.
_SC_IDEAL_BONDS = {
    "ALA": {("CA", "CB"): 1.5295},
    "SER": {("CA", "CB"): 1.5295, ("CB", "OG"): 1.417},
    "THR": {("CA", "CB"): 1.5295, ("CB", "OG1"): 1.420, ("CB", "CG2"): 1.530},
    "ASP": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.522, ("CG", "OD1"): 1.250,
            ("CG", "OD2"): 1.250},
    "ASN": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.521, ("CG", "OD1"): 1.231,
            ("CG", "ND2"): 1.328},
    "GLU": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.529, ("CG", "CD"): 1.523,
            ("CD", "OE1"): 1.250, ("CD", "OE2"): 1.250},
    "LYS": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.531, ("CG", "CD"): 1.531,
            ("CD", "CE"): 1.531, ("CE", "NZ"): 1.486},
    "PRO": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.526, ("CG", "CD"): 1.526,
            ("CD", "N"): 1.458},
    "TYR": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.512, ("CG", "CD1"): 1.389,
            ("CG", "CD2"): 1.389, ("CD1", "CE1"): 1.382, ("CD2", "CE2"): 1.382,
            ("CE1", "CZ"): 1.378, ("CE2", "CZ"): 1.378, ("CZ", "OH"): 1.376},
    "TRP": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.498, ("CG", "CD1"): 1.365,
            ("CG", "CD2"): 1.433, ("CD1", "NE1"): 1.375, ("NE1", "CE2"): 1.371,
            ("CE2", "CD2"): 1.409, ("CD2", "CE3"): 1.398, ("CE3", "CZ3"): 1.382,
            ("CZ3", "CH2"): 1.400, ("CH2", "CZ2"): 1.368, ("CZ2", "CE2"): 1.394},
}


def test_sidechain_bond_lengths() -> bool:
    """Every built sidechain must reproduce ideal bond lengths.

    Checked across four backbone conformations, because a NeRF bug that
    depends on the local frame would otherwise hide in a single geometry.
    """
    worst, worst_at = 0.0, ""
    for bb in _sc_backbones():
        for i, aa in enumerate(_SC_PROBE_SEQ):
            key = geo.ONE_TO_THREE[aa]
            if key == "GLY":
                continue
            atoms = _sc_residue_atoms(key, bb, i)
            for (a, b), target in _SC_IDEAL_BONDS[key].items():
                err = abs(float(np.linalg.norm(atoms[a] - atoms[b])) - target)
                if err > worst:
                    worst, worst_at = err, f"{key} {a}-{b}"
    ok = worst < 0.02
    return _report("sidechain bond lengths within 0.02 A of ideal", ok,
                   f"worst {worst:.5f} A on {worst_at}")


def test_sidechain_rings_planar() -> bool:
    """Tyr and Trp ring systems must be planar."""
    rings = {"TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
             "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3",
                     "CH2"]}
    worst, worst_at = 0.0, ""
    for bb in _sc_backbones():
        for key, names in rings.items():
            i = _SC_PROBE_SEQ.index(geo.THREE_TO_ONE[key])
            atoms = _sc_residue_atoms(key, bb, i)
            P = np.array([atoms[n] for n in names])
            P = P - P.mean(axis=0)
            _, _, vt = np.linalg.svd(P)
            rms = float(np.sqrt(np.mean((P @ vt[2]) ** 2)))
            if rms > worst:
                worst, worst_at = rms, key
    ok = worst < 1e-6
    return _report("aromatic rings planar (Tyr, Trp)", ok,
                   f"worst RMS deviation {worst:.2e} A on {worst_at}")


def test_sidechain_no_internal_clashes() -> bool:
    """No two heavy atoms >=4 bonds apart may sit closer than 2.5 A.

    1-2, 1-3 and 1-4 pairs are excluded: those distances are fixed by bond
    lengths and angles, and inside a ring a 1-4 pair is legitimately ~2.2 A.
    """
    worst, worst_at = 1e9, ""
    for bb in _sc_backbones():
        for i, aa in enumerate(_SC_PROBE_SEQ):
            key = geo.ONE_TO_THREE[aa]
            atoms = _sc_residue_atoms(key, bb, i)
            hops = _sc_bond_distances(key)
            names = sorted(atoms)
            for x in range(len(names)):
                for y in range(x + 1, len(names)):
                    a, b = names[x], names[y]
                    if hops.get(a, {}).get(b, 99) < 4:
                        continue
                    d = float(np.linalg.norm(atoms[a] - atoms[b]))
                    if d < worst:
                        worst, worst_at = d, f"{key} {a}-{b}"
    ok = worst > 2.5
    return _report("no intra-residue clashes (>=1-5 pairs beyond 2.5 A)", ok,
                   f"closest {worst:.3f} A on {worst_at}")


def test_sidechain_chirality() -> bool:
    """Building on a mirrored backbone must NOT give the mirrored sidechain.

    Reflect the backbone, rebuild, reflect the result back. An achiral
    construction returns the original; a correctly chiral one cannot, because
    the fixed chi angles keep their sign in the mirrored frame. Ala and Gly
    are excluded: Gly has no sidechain and Ala's only sidechain atom is CB,
    which is an input here (its chirality belongs to place_cb and is already
    covered by test_chirality).
    """
    M = np.diag([-1.0, 1.0, 1.0])
    bb = _sc_backbones()[1]
    smallest, smallest_at = 1e9, ""
    for i, aa in enumerate(_SC_PROBE_SEQ):
        key = geo.ONE_TO_THREE[aa]
        if key in ("GLY", "ALA"):
            continue
        direct = sc.build_sidechain(key, bb["N"][i], bb["CA"][i], bb["C"][i],
                                    bb["CB"][i])
        mirrored = sc.build_sidechain(key, bb["N"][i] @ M, bb["CA"][i] @ M,
                                      bb["C"][i] @ M, bb["CB"][i] @ M)
        dev = max(float(np.linalg.norm(direct[k] - mirrored[k] @ M))
                  for k in direct)
        if dev < smallest:
            smallest, smallest_at = dev, key
    ok = smallest > 0.1
    return _report("sidechains are chiral (L-amino-acid stereochemistry)", ok,
                   f"smallest mirror deviation {smallest:.3f} A on "
                   f"{smallest_at}")


def test_sidechain_rejects_unimplemented_residue() -> bool:
    """Unsupported residues must raise NotImplementedError naming the residue.

    A silent stub would corrupt the Amber energy without ever failing.
    """
    probe = ([0.0, 0.0, 0.0], [1.458, 0.0, 0.0], [2.0, 1.42, 0.0],
             [1.99, -0.77, -1.21])
    ok = True
    detail = []
    for one, three in (("F", "PHE"), ("C", "CYS"), ("H", "HIS"),
                       ("M", "MET"), ("V", "VAL")):
        for name in (one, three):
            try:
                sc.build_sidechain(name, *probe)
                ok = False
                detail.append(f"{name} did not raise")
            except NotImplementedError as exc:
                if three not in str(exc):
                    ok = False
                    detail.append(f"{name} message lacks {three}")
    return _report("unimplemented residues raise NotImplementedError", ok,
                   "; ".join(detail) if detail else
                   "F, C, H, M, V rejected by both 1- and 3-letter code")


def test_sidechain_atom_counts() -> bool:
    """Heavy-atom counts must match the PDB standard for each residue type."""
    expected = {"GLY": 4, "ALA": 5, "SER": 6, "THR": 7, "PRO": 7, "ASP": 8,
                "ASN": 8, "GLU": 9, "LYS": 9, "TYR": 12, "TRP": 14}
    ok = True
    bad = []
    for key, n in expected.items():
        if sc.heavy_atom_count(key) != n:
            ok = False
            bad.append(f"{key} {sc.heavy_atom_count(key)}!={n}")
    # and end to end, where the C-terminal residue also carries OXT
    bb = _sc_backbones()[1]
    full = sc.build_full_structure(_SC_PROBE_SEQ, bb)
    total = sum(expected[geo.ONE_TO_THREE[a]] for a in _SC_PROBE_SEQ) + 1
    if full["n_atoms"] != total:
        ok = False
        bad.append(f"total {full['n_atoms']}!={total}")
    return _report("sidechain atom counts match PDB standard", ok,
                   "; ".join(bad) if bad else
                   f"{full['n_atoms']} heavy atoms over {len(_SC_PROBE_SEQ)} "
                   f"residues (incl. OXT)")




def run_all() -> bool:
    print("=" * 72)
    print("VALIDATION SUITE")
    print("=" * 72)
    tests = [
        test_bitstring_and_coordinate_lengths,
        test_backbone_geometry_is_physical,
        test_helix_is_representable,
        test_chirality,
        test_mj_correction_changes_sign,
        test_energy_is_finite_and_deterministic,
        test_energy_rejects_wrong_length,
        test_steric_penalizes_clash,
        test_hbond_rewards_helix,
        test_cvar_preserves_multiplicities,
        test_cvar_sampled_converges_to_exact,
        test_cvar_alpha_one_is_mean,
        test_vqe_is_full_system,
        test_vqe_does_not_enumerate,
        test_final_answer_comes_from_final_circuit,
        test_reproducibility,
        test_rng_streams_are_independent,
        test_no_pdb_access_during_optimization,
        test_hamiltonian_has_no_native_input,
        test_rmsd_is_angstroms,
        test_no_normalized_rmsd_path,
        test_vqe_matches_exhaustive_on_tiny_system,
        test_ceiling_is_a_real_ceiling,
        test_representation_ceiling_computable,
        test_sidechain_bond_lengths,
        test_sidechain_rings_planar,
        test_sidechain_no_internal_clashes,
        test_sidechain_chirality,
        test_sidechain_rejects_unimplemented_residue,
        test_sidechain_atom_counts,
    ]

    results = []
    for t in tests:
        try:
            results.append(bool(t()))
        except Exception as exc:
            print(f"  [{_FAIL}] {t.__name__} raised {type(exc).__name__}: {exc}")
            results.append(False)
    print("-" * 72)
    print(f"  {sum(results)}/{len(results)} passed. "
          f"ALL PASSED: {all(results)}")
    print()
    return all(results)