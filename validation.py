"""Correctness, leakage, budget and *objective-quality* suite.

The objective-quality group (see `energy_quality`) is the important addition of the
2026-07-28 pass. Before it, the suite reported 35/35 passing while the energy model was
anti-correlated with correctness -- it verified that searches found the minimum and that
metrics were in Angstroms, but never that the function being minimised pointed at the
answer. Those gates fail loudly on the pre-rewrite model, which is the point.
"""
import inspect
import math
import os
import warnings

import numpy as np

import protein_geometry as geo
import representations as reps
import sidechains as sc
import energy_terms as et
import hamiltonian as ham
import budget as bud
try:
    import amber_hamiltonian as amb
    _HAVE_AMBER = True
except ImportError as _exc:          # openmm is an optional dependency
    amb = None
    _HAVE_AMBER = False
    _AMBER_IMPORT_ERROR = _exc
import vqe as vqe_mod
import classical_baselines as cb
import evaluation as ev
import energy_quality as eq
import dataset as ds
import experiments as exp

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{_PASS if ok else _FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _skip(name: str, why: str) -> None:
    print(f"  [{_SKIP}] {name} -- {why}")


# ==========================================================================
# Representation and geometry
# ==========================================================================
def test_bitstring_and_coordinate_lengths() -> bool:
    ok = True
    seqs = {10: "GYDPETGTWG", 12: "SWTWEGNKWTWK", 15: "AEKAAGKPWYTDENL"}
    for n_res, seq in seqs.items():
        n_chi = sum(1 for a in seq if a in reps.CHI1_ENCODED)
        for n_states in (4, 8):
            r = reps.TorsionStateRepresentation(n_res, n_states=n_states,
                                                sequence=seq)
            backbone_bits = n_res * (2 if n_states == 4 else 3)
            expected_bits = backbone_bits + n_chi
            ok &= (r.n_backbone_bits == backbone_bits) and (r.n_chi_bits == n_chi)
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
    seq = "AAAAAAAAAAAA"
    r = reps.TorsionStateRepresentation(12, n_states=4, sequence=seq)
    b = r.bitstring_from_states([0] * 12)     # all helical
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
    """Index 0 must be the helical state in every per-residue-class library."""
    seq = "AEKAAGKPWYTDEN"          # includes G, P and a pre-Pro (K) on purpose
    r = reps.TorsionStateRepresentation(14, n_states=4, sequence=seq)
    ca = r.build_coords(r.bitstring_from_states([0] * 14))["CA"]
    d14 = np.linalg.norm(ca[4:] - ca[:-4], axis=1)
    # Gly/Pro/pre-Pro helical states differ slightly from the general one, so the
    # tolerance is wider than for a homopolymer -- but state 0 must still be a helix.
    ok = bool(np.all(np.abs(d14 - 6.2) < 2.0))
    return _report("state 0 is helical in every residue-class library", ok,
                   f"i,i+4 mean {d14.mean():.2f} A (alpha helix ~6.2 A)")


def test_extended_is_representable() -> bool:
    """Index 1 must be the extended state in every library."""
    seq = "AEKAAGKPWYTDEN"
    r = reps.TorsionStateRepresentation(14, n_states=4, sequence=seq)
    ca = r.build_coords(r.bitstring_from_states([1] * 14))["CA"]
    span = float(np.linalg.norm(ca[-1] - ca[0]))
    # 14 residues fully extended spans ~3.3 A per residue
    ok = span > 0.6 * 3.3 * 13
    return _report("state 1 is extended in every residue-class library", ok,
                   f"end-to-end {span:.1f} A over 13 bonds")


def test_chirality() -> bool:
    r = reps.TorsionStateRepresentation(12, n_states=8, sequence="A" * 12)
    right = r.build_coords(r.bitstring_from_states([0] * 12))["CA"]   # alpha-R
    left = r.build_coords(r.bitstring_from_states([7] * 12))["CA"]    # alpha-L
    resid = geo.ca_rmsd(left, right)
    ok = resid > 2.0
    return _report("representation is chiral (alpha-R != alpha-L)", ok,
                   f"CA-RMSD {resid:.2f} A")


def test_libraries_are_residue_class_specific() -> bool:
    """Gly, Pro and pre-Pro must get different states from a general residue.

    This is the fix for a real modelling error: one global (phi, psi) list was applied to
    every residue, so proline was offered phi = +60 (geometrically impossible for the
    pyrrolidine ring) and glycine was offered no more of the left-handed region than
    isoleucine, despite that region being heavily populated for Gly and rare otherwise.
    """
    seq = "AGPKA"                      # A general, G gly, P pro, K pre-nothing, A
    r = reps.TorsionStateRepresentation(5, n_states=4, sequence=seq)
    classes = r.classes
    detail = []
    ok = True

    if classes != [reps.CLASS_GENERAL, reps.CLASS_GLY, reps.CLASS_PRO,
                   reps.CLASS_GENERAL, reps.CLASS_GENERAL]:
        ok = False
        detail.append(f"classes {classes}")

    # pre-Pro detection: the residue before P
    r2 = reps.TorsionStateRepresentation(3, n_states=4, sequence="APA")
    if r2.classes[0] != reps.CLASS_PRE_PRO:
        ok = False
        detail.append(f"pre-Pro not detected: {r2.classes}")

    # every proline state must keep phi inside the ring-allowed window
    for phi_deg, _ in reps.PRO_4 + reps.PRO_8:
        if not (-90.0 <= phi_deg <= -40.0):
            ok = False
            detail.append(f"Pro phi {phi_deg} outside ring-allowed window")

    # glycine must have access to a positive-phi (left-handed) state
    if not any(p > 0 for p, _ in reps.GLY_4):
        ok = False
        detail.append("Gly library has no positive-phi state")

    # a general residue should NOT spend one of only four states on alpha-L
    if any(p > 30.0 for p, _ in reps.GENERAL_4):
        ok = False
        detail.append("GENERAL_4 spends a state on alpha-L")

    return _report("torsion libraries are residue-class specific", ok,
                   "; ".join(detail) if detail else
                   "Gly/Pro/pre-Pro/general all distinct, Pro phi locked")


def test_legacy_library_is_byte_stable() -> bool:
    """The pinned pre-2026-07-28 torsion library must never drift.

    `run_8state_seed0.py` reproduces a persisted reference set of bitstring -> energy
    pairs that `partest/golden_check.py` verifies to max absolute error 0.0. Those
    energies are a function of the (phi, psi) values a bitstring decodes to, so editing
    `LEGACY_4`/`LEGACY_8` silently invalidates a 4-hour reference run with no test
    failure anywhere. This is that test failure.
    """
    expect_4 = [(-63.0, -42.0), (-120.0, 130.0), (-75.0, 150.0), (60.0, 45.0)]
    expect_8 = [(-63.0, -42.0), (-120.0, 130.0), (-75.0, 150.0), (-140.0, 160.0),
                (-60.0, -30.0), (-90.0, 0.0), (60.0, 45.0), (90.0, -10.0)]
    ok = (reps.LEGACY_4 == expect_4) and (reps.LEGACY_8 == expect_8)
    # and legacy_library=True must bypass the residue-class libraries AND the chi1 bits:
    # it means the whole pre-2026-07-28 *encoding*, so n_bits and the bitstring ->
    # structure map are unchanged.
    r = reps.TorsionStateRepresentation(5, n_states=8, sequence="AGPKW",
                                        legacy_library=True)
    ok &= all(c == reps.CLASS_LEGACY for c in r.classes)
    ok &= all(lib == reps.LEGACY_8 for lib in r.libraries)
    ok &= (r.n_chi_bits == 0) and (r.n_bits == 3 * 5)
    r2 = reps.TorsionStateRepresentation(5, n_states=8, sequence="AGPKW")
    ok &= r2.libraries[1] != r.libraries[1]      # Gly must differ without the flag
    ok &= (r2.n_chi_bits == 1)                   # and the Trp gains a chi bit
    return _report("legacy torsion library is byte-stable (golden runs)", ok,
                   "LEGACY_4/8 unchanged; legacy_library=True bypasses "
                   "residue-class libraries and chi1 bits")


def test_parallel_builder_carries_the_library_setting() -> bool:
    """A worker must rebuild the *same* representation the parent used.

    `mps.parallel.OBC2Builder` reconstructs a Hamiltonian inside a spawned process,
    because an `openmm.Context` cannot be pickled. It used to carry only
    `(sequence, n_states)` -- so once torsion libraries became sequence- and
    setting-dependent, a worker would decode the same bitstring to different (phi, psi)
    values than the parent and return a *different energy for the same structure*, with no
    error, breaking the byte-identical invariant that `partest/golden_check.py` exists to
    verify.
    """
    try:
        from mps.parallel import OBC2Builder
    except Exception as exc:
        _skip("test_parallel_builder_carries_the_library_setting",
              f"mps package unavailable ({type(exc).__name__})")
        return True
    params = inspect.signature(OBC2Builder.__init__).parameters
    ok = "legacy_library" in params
    b = OBC2Builder("GYDPETGTWG", 8, legacy_library=True)
    ok &= (b.legacy_library is True and b.sequence == "GYDPETGTWG")
    return _report("parallel builder carries the torsion-library setting", ok,
                   f"OBC2Builder args: {sorted(params)}")


def test_lattice_coordinates_are_in_angstroms() -> bool:
    """Adjacent lattice CA atoms must be 3.80 A apart.

    The lattice used to emit raw lattice units (bond norm sqrt(3) ~ 1.732) into an energy
    model parameterised entirely in Angstroms -- a 2.194x scale error that made the
    steric term charge 20.80 for a |i-j| = 2 turn against 0.45 for a straight
    continuation, a 46x penalty for turning that came purely from the units and drove
    every search on this representation to the extended chain.
    """
    lat = reps.TetrahedralLatticeRepresentation(10)
    bits = lat.random_bitstring(np.random.default_rng(0))
    ca = lat.decode(bits)
    step = np.linalg.norm(np.diff(ca, axis=0), axis=1)
    ok = bool(np.allclose(step, reps.CA_CA_DISTANCE, atol=1e-9))
    # and the sep-2 minimum must clear the steric CA threshold
    d2 = np.linalg.norm(ca[2:] - ca[:-2], axis=1)
    turn_min = 2.0 * reps.LATTICE_SCALE
    ok &= bool(np.all(d2 >= turn_min - 1e-9))
    ok &= turn_min > 3.8
    return _report("lattice coordinates are in Angstroms", ok,
                   f"CA-CA {step.mean():.4f} A, sep-2 minimum "
                   f"{turn_min:.2f} A (> 3.8 steric threshold)")


def test_lattice_steric_is_quiet_for_valid_traces() -> bool:
    """A self-avoiding lattice trace must incur no steric penalty at all.

    With correct units the only lattice geometry that can overlap is a site collision:
    the minimum distance between two *distinct* sites is sqrt(3) * LATTICE_SCALE = 3.80 A,
    above the 0.80 * (1.70 + 1.70) = 2.72 A soft-sphere limit for a carbon pair. Before
    the fix, every turn paid a large steric cost.
    """
    seq = "GYDPETGTWG"
    lat = reps.TetrahedralLatticeRepresentation(len(seq))
    rng = np.random.default_rng(1)
    worst, checked = 0.0, 0
    ii, jj = np.triu_indices(len(seq), 1)
    for _ in range(2000):
        bits = lat.random_bitstring(rng)
        if et.backtracking_term(lat, bits) > 0:
            continue
        ca = lat.decode(bits)
        # A non-backtracking walk can still return to a site it already occupied. Two
        # residues in the same place is a genuine clash and steric SHOULD fire on it, so
        # those traces are not what this test is about.
        if np.any(np.linalg.norm(ca[ii] - ca[jj], axis=1) < 1e-9):
            continue
        checked += 1
        worst = max(worst, et.steric_term(lat.build_coords(bits), seq))
    ok = (checked > 0) and (worst < 1e-9)
    return _report("lattice steric is zero for self-avoiding traces", ok,
                   f"worst {worst:.3e} over {checked} traces")


def test_umeyama_scale_beats_bond_ratio() -> bool:
    """The scaled superposition must use the RMSD-optimal scale."""
    rng = np.random.default_rng(3)
    P = rng.normal(size=(12, 3)) * 2.0
    Q = P * 1.7 + rng.normal(size=(12, 3)) * 0.3
    _, s = geo.kabsch_superpose_with_scale(P, Q)
    opt = geo.rmsd(geo.kabsch_superpose(P * s, Q), Q)
    p_bond = np.linalg.norm(np.diff(P, axis=0), axis=1).mean()
    q_bond = np.linalg.norm(np.diff(Q, axis=0), axis=1).mean()
    old = geo.rmsd(geo.kabsch_superpose(P * (q_bond / p_bond), Q), Q)
    ok = opt <= old + 1e-9
    return _report("scaled superposition uses the optimal (Umeyama) scale", ok,
                   f"optimal {opt:.4f} A vs bond-ratio {old:.4f} A")


# ==========================================================================
# Energy terms
# ==========================================================================
def test_mj_correction_changes_sign() -> bool:
    raw_vals = np.array(list(et.MJ_RAW.values()))
    cor_vals = np.array(list(et.MJ_CORRECTED.values()))
    ok = bool(np.all(raw_vals < 0) and np.any(cor_vals > 0))
    return _report("MJ self-energy correction applied", ok,
                   f"corrected range [{cor_vals.min():.2f}, {cor_vals.max():.2f}]")


def test_burial_scale_places_aromatics_hydrophobic() -> bool:
    """The burial scale must call the aromatics hydrophobic.

    Kyte-Doolittle does not (W -0.9, Y -1.3), and under KD a peptide whose core is an
    aromatic cluster is penalised for forming it. Every residue of chignolin is negative
    under KD, which made `solvation_term` purely expansive for it -- and the same is true
    of trpzip's four-tryptophan core. This is a property of the scale, so it is tested on
    the scale rather than on any sequence.
    """
    b = et.BURIAL
    ok = len(b) == 20
    ok &= all(b[a] > 0.5 for a in "WYFILM")          # aromatics + aliphatics buried
    ok &= all(b[a] < 0.0 for a in "DEKRNQ")          # charged/polar exposed
    ok &= b["W"] == max(b.values())
    ok &= b["W"] > b["S"] and b["Y"] > b["S"]
    return _report("burial scale places aromatics on the hydrophobic side", ok,
                   f"W {b['W']:+.2f}, Y {b['Y']:+.2f}, F {b['F']:+.2f}, "
                   f"D {b['D']:+.2f}, K {b['K']:+.2f}")


def test_hbond_is_not_length_normalised() -> bool:
    """H-bond energy must grow with the number of bonds, not stay flat.

    `energy_components` used to return `(local + longrange) / n`, which suppressed the
    only long-range structural term by a factor of the chain length -- 10x at N=10 -- and
    nothing else in the model was normalised that way. A helix of 20 residues has roughly
    twice the H-bonds of one of 10, so the term must roughly double.
    """
    vals = {}
    for n in (10, 20):
        seq = "A" * n
        r = reps.TorsionStateRepresentation(n, n_states=4, sequence=seq)
        bits = r.bitstring_from_states([0] * n)
        coords = r.build_coords(bits)
        vals[n] = et.energy_components(seq, coords, *r.decode(bits))["hbond_local"]
    ok = vals[20] < vals[10] * 1.5      # both negative; 20-mer must be clearly lower
    return _report("hbond term is not divided by chain length", ok,
                   f"N=10 {vals[10]:.3f}, N=20 {vals[20]:.3f} kcal/mol")


def test_hbond_longrange_is_a_separate_calibrated_term() -> bool:
    """Long-range and local H-bonds must be separately weighted, and the long-range
    emphasis must be calibrated rather than hardcoded.

    Local i->i+4 bonds are available to a helix; the long-range ladder is what makes a
    hairpin a hairpin. Adding the two with equal weight -- which the code did after
    carefully separating them -- lets a helix substitute for the correct fold. But a *fixed*
    long-range multiplier is the opposite error: it is a standing bet that every target is a
    hairpin, and it gives a helix nothing. So the multiplier is a free weight, which also
    means the ablation produces separate `no_hbond_local` / `no_hbond_longrange` arms.
    """
    ok = ("hbond_local" in et.TERM_NAMES) and ("hbond_longrange" in et.TERM_NAMES)
    ok &= "hbond" not in et.TERM_NAMES          # the merged term must be gone
    ok &= et.DEFAULT_WEIGHTS["hbond_longrange"] > et.DEFAULT_WEIGHTS["hbond_local"]
    ok &= "hbond_longrange" in et.FREE_WEIGHTS
    ok &= "hbond_local" not in et.FREE_WEIGHTS  # pinned: DSSP kcal/mol at unit weight
    ok &= all(t in et.COUPLED_TERMS for t in ("hbond_local", "hbond_longrange"))
    return _report("long-range H-bond weight is separate and calibrated", ok,
                   f"local {et.DEFAULT_WEIGHTS['hbond_local']} (pinned), "
                   f"long-range {et.DEFAULT_WEIGHTS['hbond_longrange']} (free)")


def test_electrostatics_has_physical_magnitude() -> bool:
    """A salt bridge must be worth ~1 kcal/mol, not ~0.08.

    The term omitted the 332 kcal A / (mol e^2) Coulomb constant entirely, making it
    13-40x too weak and giving it the smallest spread of any term in the model.
    """
    q = np.array([-1.0, 0.0, 1.0])
    di, dj = np.triu_indices(3, 1)
    sep = dj - di
    d = np.array([4.0, 4.0, 4.0])
    v = abs(et.electrostatic_term(q, sep, di, dj, d))
    ok = 0.3 < v < 10.0
    return _report("electrostatics has a physical magnitude", ok,
                   f"|D...K at 4 A| = {v:.3f} kcal/mol "
                   f"(prefactor {et.COULOMB}/{et.DIELECTRIC})")


def test_steric_sees_backbone_atoms() -> bool:
    """Steric must detect a clash between backbone atoms, not only CA/CB.

    The previous term tested CA-CA < 3.8 and CB-CB < 3.4 only, so N, C and O could
    interpenetrate at no cost -- removing the constraint that should force an aromatic
    pair to stack rather than overlap. Here two residues' full backbones are pushed
    together while CA and CB are left far enough apart that the old test would have
    reported nothing.
    """
    seq = "AAAA"
    r = reps.TorsionStateRepresentation(4, n_states=4, sequence=seq)
    coords = r.build_coords(r.bitstring_from_states([1, 1, 1, 1]))
    clean = et.steric_term(coords, seq)
    bad = {k: v.copy() for k, v in coords.items()}
    # Move residue 3's carbonyl O on top of residue 0's carbonyl C.
    bad["O"][3] = coords["C"][0] + np.array([0.5, 0.0, 0.0])
    clashed = et.steric_term(bad, seq)
    ca_far = np.linalg.norm(bad["CA"][3] - bad["CA"][0]) > 3.8
    cb_far = np.linalg.norm(bad["CB"][3] - bad["CB"][0]) > 3.4
    ok = (clashed > clean + 0.1) and ca_far and cb_far
    return _report("steric term sees backbone N/C/O clashes", ok,
                   f"clean {clean:.3f} -> clashed {clashed:.3f} with CA and CB "
                   f"both beyond the old thresholds")


def test_steric_exempts_hbond_partners() -> bool:
    """N...O pairs must not be penalised, or steric fights the H-bond term."""
    n = 12
    seq = "A" * n
    r = reps.TorsionStateRepresentation(n, n_states=4, sequence=seq)
    coords = r.build_coords(r.bitstring_from_states([0] * n))   # helix: real H-bonds
    s = et.steric_term(coords, seq)
    bonds = geo.dssp_hbonds(coords, min_sep=2)
    ok = (len(bonds) > 0) and (s < 1.0)
    return _report("steric exempts N...O H-bond partners", ok,
                   f"{len(bonds)} H-bonds present, steric {s:.3f}")


def test_hbond_cannot_pay_for_interpenetration() -> bool:
    """A collapsed N...O pair must score zero, not a jackpot.

    `_steric_layout` exempts every hetero N/O pair and the amide H is not in the steric
    layout at all, so before `HB_MIN_ON` the DSSP `-1/dOH` term was unbounded below with
    nothing opposing it: steric is quadratic and capped near 27 weighted, while the H-bond
    term diverges. Simulated annealing found this on 2 of 3 seeds at the default budget,
    returning structures near -58 against an ideal helix's -28, with one matched "bond" at
    dOH = 0.503 -- the old numerical guard itself.

    Two halves, and both matter. The collapsed pair must be rejected outright, and a real
    helix must be completely unaffected: its bonds sit at dON 3.079, far outside the cutoff,
    so the fix must not cost a single one of them.
    """
    n = 12
    seq = "A" * n
    r = reps.TorsionStateRepresentation(n, n_states=4, sequence=seq)
    helix = r.build_coords(r.bitstring_from_states([0] * n))
    local, lr, pairs = et.hbond_terms(helix)
    collapsed = {k: v.copy() for k, v in helix.items()}
    collapsed["O"][7] = collapsed["N"][0] + np.array([0.9, 0.0, 0.0])
    c_local, c_lr, c_pairs = et.hbond_terms(collapsed)
    ok = len(pairs) > 0 and local < -1.0          # the helix still bonds
    ok &= all((0, 7) != p for p in c_pairs)       # the 0.9 A pair is rejected
    ok &= (c_local + c_lr) > (local + lr) - 1e-9  # and pays nothing for existing
    # Behavioural, not a restatement of the constants: drive a lone acceptor carbonyl
    # through a valid donor and confirm no single admissible bond ever scores below the
    # floor. The sweep is anti-linear -- carbonyl C jammed toward the donor N with the =O
    # swung away -- because that is the ONLY geometry in which the DSSP form dips below
    # HB_E_FLOOR while still clearing dON > HB_MIN_ON; a collinear N-H...O=C bond reaches
    # interpenetration only once dON has already fallen inside the cutoff. Unclamped, the
    # deepest admissible sample here is about -5.5, so if the maximum() clamp were removed
    # this assertion would fail -- it is load-bearing, not a restatement of the constant.
    # Done on an extended chain, which carries no H-bonds of its own, and the acceptor is
    # the terminal residue (no following amide H to perturb), so local+lr IS the probe bond.
    probe = {k: v.copy() for k, v in
             r.build_coords(r.bitstring_from_states([1] * n)).items()}
    base_l, base_r, base_p = et.hbond_terms(probe, desolvation_cost=0.0)
    ok &= (len(base_p) == 0)                      # nothing to confound the sweep
    n_don = probe["N"][1]                         # residue 1 has a real amide H; residue 0 has none
    h_don = geo.amide_h_positions(probe["N"], probe["C"], probe["O"])[1]
    axis = (h_don - n_don) / np.linalg.norm(h_don - n_don)
    deepest, n_probed = 0.0, 0
    for s in np.arange(1.30, 3.01, 0.02):         # s = acceptor C ... donor N separation
        probe["C"][n - 1] = n_don - axis * float(s)          # carbonyl C on the far side of N
        probe["O"][n - 1] = n_don - axis * float(s + 1.23)   # =O swung further away
        pl, pr, pp = et.hbond_terms(probe, desolvation_cost=0.0)
        if len(pp) == 1:                          # only single-bond samples are comparable
            deepest = min(deepest, pl + pr)
            n_probed += 1
    ok &= n_probed > 0 and deepest >= et.HB_E_FLOOR - 1e-9
    return _report("H-bond term cannot pay for interpenetration", ok,
                   f"helix keeps {len(pairs)} bonds at local {local:.3f}; "
                   f"a 0.9 A N...O pair is rejected, not scored")


def test_build_all_matches_separate_calls() -> bool:
    """The one-pass hot path must agree exactly with the individual accessors.

    `hamiltonian.components` now calls `rep.build_all` instead of `decode` +
    `build_coords` + `build_rings`, which removed three redundant bitstring validations and
    one redundant decode per energy evaluation. This is the test that the shortcut did not
    change what is computed.
    """
    ok = True
    detail = []
    for seq in ("GYDPETGTWG", "SWTWEGNKWTWK", "AEKAAG"):
        r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
        rng = np.random.default_rng(0)
        for _ in range(20):
            bits = r.random_bitstring(rng)
            phi, psi, coords, rings = r.build_all(bits)
            phi0, psi0 = r.decode(bits)
            coords0 = r.build_coords(bits)
            rings0 = r.build_rings(bits) or None
            if not (np.array_equal(phi, phi0) and np.array_equal(psi, psi0)):
                ok = False
                detail.append(f"{seq}: torsions differ")
                break
            if any(not np.array_equal(coords[k], coords0[k]) for k in coords0):
                ok = False
                detail.append(f"{seq}: coords differ")
                break
            if (rings is None) != (rings0 is None):
                ok = False
                detail.append(f"{seq}: ring presence differs")
                break
            if rings and any(not np.array_equal(rings[i][n], rings0[i][n])
                             for i in rings for n in rings[i]):
                ok = False
                detail.append(f"{seq}: ring atoms differ")
                break
    # the lattice must honour the same interface
    lat = reps.TetrahedralLatticeRepresentation(10)
    lb = lat.random_bitstring(np.random.default_rng(0))
    lphi, lpsi, lcoords, lrings = lat.build_all(lb)
    ok &= (lphi is None and lpsi is None and lrings is None)
    ok &= np.array_equal(lcoords["CA"], lat.build_coords(lb)["CA"])
    return _report("build_all matches the separate accessors", ok,
                   "; ".join(detail) if detail else
                   "60 structures over 3 sequences plus the lattice, exact equality")


def test_steric_layout_cache_is_coordinate_independent() -> bool:
    """The cached steric pair layout must depend on layout only, never on positions.

    `_steric_layout` memoises the pair indices, radii and the N/O exemption mask, keyed on
    (atom names, residue count, ring atom keys, min_sep). If coordinates ever leaked into
    that cache, two different structures with the same layout would score identically --
    silently, and only on the second one.
    """
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    et.clear_sequence_cache()
    rng = np.random.default_rng(2)
    vals_cold, vals_warm = [], []
    bitstrings = [r.random_bitstring(rng) for _ in range(12)]
    for bits in bitstrings:                      # cache warms during this pass
        _, _, coords, rings = r.build_all(bits)
        vals_warm.append(et.steric_term(coords, seq, rings=rings))
    for bits in bitstrings:                      # each one recomputed from a cold cache
        et.clear_sequence_cache()
        _, _, coords, rings = r.build_all(bits)
        vals_cold.append(et.steric_term(coords, seq, rings=rings))
    same = all(abs(a - b) < 1e-12 for a, b in zip(vals_cold, vals_warm))
    n_distinct = len({round(v, 9) for v in vals_warm})
    ok = same and n_distinct > 1
    return _report("steric layout cache is coordinate-independent", ok,
                   f"cold == warm for 12 structures: {same}; "
                   f"{n_distinct} distinct values (must be > 1)")


def test_solvation_matches_naive_pair_sum() -> bool:
    """Accumulating coordination from the upper triangle must equal the (n, n) form.

    `solvation_term` was rebuilding a full square distance matrix and calling
    `fill_diagonal`; it now accumulates from the pair distances `energy_components` has
    already computed. Same number, half the distance work, no square allocation.
    """
    seq = "GYDPETGTWG"
    n = len(seq)
    r = reps.TorsionStateRepresentation(n, n_states=4, sequence=seq)
    burial, _, _ = et.sequence_arrays(seq)
    di, dj, _ = et.pair_index(n)
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(25):
        CB = np.asarray(r.build_coords(r.random_bitstring(rng))["CB"], dtype=float)
        d_cb = np.linalg.norm(CB[di] - CB[dj], axis=1)
        fast = et.solvation_term(burial, d_cb, di, dj, n)
        D = np.linalg.norm(CB[:, None, :] - CB[None, :, :], axis=2)
        coord = et.switch(D, 6.0, 10.0)
        np.fill_diagonal(coord, 0.0)
        naive = float(-np.sum((burial / et.BURIAL_NORM) * coord.sum(axis=1)))
        worst = max(worst, abs(fast - naive))
    ok = worst < 1e-9
    return _report("solvation matches the naive pair sum", ok,
                   f"worst |fast - naive| = {worst:.2e} over 25 structures")


def test_cooperativity_distinguishes_runs_from_scattered_bonds() -> bool:
    """A contiguous ladder must beat the same number of scattered long-range bonds.

    This is the one thing an additive pairwise potential structurally cannot express.
    `hbond_longrange` is a sum over independent matched pairs, so two scattered bonds score
    identically to two adjacent rungs forming a correct register -- which is why "right
    topology, wrong register" is the expected residual failure mode. Same argument for
    helices and `hbond_local`, hence both terms.

    Tested on synthetic pair lists so it is exact and sequence-free: no coordinates, no
    amino acids, no distance scale.
    """
    ok = True
    detail = []

    # sheet: a 3-rung antiparallel ladder (i, j) -> (i+2, j-2) scores 2; the same
    # number of long-range bonds, scattered, scores 0.
    ladder = ((2, 12), (4, 10), (6, 8))
    scattered = ((2, 12), (5, 9), (0, 15))
    lad = et.coop_sheet_term(ladder)
    sca = et.coop_sheet_term(scattered)
    if not (lad == -2.0 and sca == 0.0):
        ok = False
        detail.append(f"sheet ladder {lad} vs scattered {sca}")

    # and it must be superlinear in ladder length: 4 rungs -> 3, not 4/3 of 3 rungs
    if et.coop_sheet_term(ladder + ((8, 6),)) != -3.0:
        ok = False
        detail.append("sheet term not linear in adjacent-rung count")

    # helix: consecutive n-turns (donor - acceptor in {3,4}, run condition (d+1, a+1))
    run = ((4, 0), (5, 1), (6, 2))
    broken = ((4, 0), (7, 3), (10, 6))
    hel = et.coop_helix_term(run)
    brk = et.coop_helix_term(broken)
    if not (hel == -2.0 and brk == 0.0):
        ok = False
        detail.append(f"helix run {hel} vs broken {brk}")

    # the two terms must not double-count: a helix run has no ladder and vice versa
    if et.coop_sheet_term(run) != 0.0 or et.coop_helix_term(ladder) != 0.0:
        ok = False
        detail.append("helix and sheet cooperativity overlap")

    return _report("cooperativity distinguishes runs from scattered bonds", ok,
                   "; ".join(detail) if detail else
                   "3-rung ladder -2 vs scattered 0; 3-turn helix run -2 vs broken 0; "
                   "no overlap")


def test_cooperativity_is_topology_symmetric() -> bool:
    """Cooperativity must not smuggle in a preference for hairpins.

    `hbond_longrange` at 3x local is already a hairpin-favouring choice, which is why it is
    calibrated rather than fixed. A ladder-only cooperativity term would be the same bet
    again, and a helix would get nothing from it -- so the helix pattern (consecutive
    n-turns, which is what DSSP means by a helix) is rewarded by the same mechanism with the
    same default weight.

    Equal weights are necessary and NOT sufficient, which is what this test used to miss.
    Uncapped, the helix run advances one residue at a time along the whole chain (ceiling
    n-5) while the ladder steps two and is bounded by strand length (ceiling ~n/4): 5 vs 2
    at n=10, 15 vs 8 at n=20. Equal weights on unequal ceilings is a preference, and it was
    large enough to make an ideal helix the global minimum of the chignolin landscape. The
    ceiling parity assertion below is the part that actually has teeth; `COOP_RUN_CAP` is
    what supplies it.
    """
    ok = (et.DEFAULT_WEIGHTS["coop_helix"] == et.DEFAULT_WEIGHTS["coop_sheet"])
    # Ceiling parity: a full-length helix must not out-earn a full-length ladder by more
    # than a small factor. Built from real bond patterns, not from the constant, so this
    # fails if the cap is removed rather than merely restating COOP_RUN_CAP.
    n = 20
    helix_pairs = tuple((i + 4, i) for i in range(n - 4))          # every i->i+4 turn
    ladder_pairs = tuple((2 + 2 * k, 18 - 2 * k) for k in range(8))  # every ladder rung
    max_h = abs(et.coop_helix_term(helix_pairs))
    max_s = abs(et.coop_sheet_term(ladder_pairs))
    ok &= max_s > 0.0 and max_h <= 2.0 * max_s
    ok &= all(t in et.FREE_WEIGHTS for t in ("coop_helix", "coop_sheet"))
    ok &= all(t in et.COUPLED_TERMS for t in ("coop_helix", "coop_sheet"))
    ok &= all(t in et.TERM_NAMES for t in ("coop_helix", "coop_sheet"))
    # both must be favourable-or-zero, never a penalty
    ok &= et.coop_helix_term(((4, 0), (5, 1))) <= 0.0
    ok &= et.coop_sheet_term(((2, 12), (4, 10))) <= 0.0
    ok &= et.coop_helix_term(()) == 0.0 and et.coop_sheet_term(()) == 0.0
    return _report("cooperativity is topology-symmetric", ok,
                   f"equal weights ({et.DEFAULT_WEIGHTS['coop_helix']}) AND comparable "
                   f"ceilings at n=20: helix {max_h:.0f} vs ladder {max_s:.0f} "
                   f"(uncapped the helix would be 15)")


def test_coop_helix_run_is_capped() -> bool:
    """A longer helix must stop earning extra cooperativity once it is established.

    Uncapped, `coop_helix_term` returns one unit per residue for the whole chain, so the
    reward for being helical grows without bound while `coop_sheet_term`'s is bounded by
    strand length. That is why an ideal alpha-helix was the global minimum of chignolin's
    4,194,304-structure landscape at 5.27 A CA-RMSD; with the cap the minimum is a 1.96 A
    hairpin and the mean RMSD of the 100 lowest-energy clash-free structures falls from
    5.124 A to 2.567 A.

    Both halves matter. The count must saturate, and it must still DISTINGUISH a helix from
    scattered bonds below the cap -- a cap that flattened everything to one value would
    destroy the cooperativity the term exists to provide.
    """
    def run_of(k):
        """k consecutive n-turns as matched (donor, acceptor) pairs."""
        return tuple((i + 4, i) for i in range(k + 1))
    vals = [abs(et.coop_helix_term(run_of(k))) for k in range(0, 9)]
    ok = max(vals) == float(et.COOP_RUN_CAP)          # saturates
    ok &= vals[0] == 0.0                              # a lone turn is not a run
    ok &= vals[1] == 1.0 and vals[2] == 2.0           # still graded below the cap
    ok &= all(v == float(et.COOP_RUN_CAP) for v in vals[et.COOP_RUN_CAP:])
    # scattered turns must still score zero -- the cap must not turn into a helix bonus
    ok &= et.coop_helix_term(((4, 0), (12, 8))) == 0.0
    # and the sheet term must NOT be capped: capping both costs the real hairpin natives
    # 4.0 kcal/mol (1LE1, 1LE3 at coop_sheet -3.0) for no measured benefit
    long_ladder = tuple((2 + 2 * k, 18 - 2 * k) for k in range(8))
    ok &= abs(et.coop_sheet_term(long_ladder)) > float(et.COOP_RUN_CAP)
    return _report("coop_helix run count is capped", ok,
                   f"runs of 0..8 turns score {[int(v) for v in vals]} "
                   f"(cap {et.COOP_RUN_CAP}); coop_sheet uncapped at "
                   f"{abs(et.coop_sheet_term(long_ladder)):.0f}")


def test_cooperativity_reaches_the_energy() -> bool:
    """The terms must be wired through `energy_components`, not just defined.

    A helix is the cheapest structure to build that has a genuine run of n-turns, so it is
    the check that the pair list actually reaches `coop_helix_term` with real geometry.
    """
    n = 14
    seq = "A" * n
    r = reps.TorsionStateRepresentation(n, n_states=4, sequence=seq)
    helix_bits = r.bitstring_from_states([0] * n)
    ext_bits = r.bitstring_from_states([1] * n)
    H = ham.FoldingHamiltonian(seq, r)
    ch = H.components(helix_bits)
    ce = H.components(ext_bits)
    ok = ("coop_helix" in ch) and ("coop_sheet" in ch)
    ok &= ch["coop_helix"] < 0.0        # a helix has consecutive n-turns
    ok &= ce["coop_helix"] == 0.0       # an extended chain has none
    return _report("cooperativity terms reach the energy", ok,
                   f"helix coop_helix {ch['coop_helix']:.1f}, "
                   f"extended {ce['coop_helix']:.1f}")


def test_chi1_is_encoded_for_aromatics_only() -> bool:
    """One bit per aromatic, appended after the backbone bits, and nothing else.

    Chi1 buys the geometry that the aromatic term needs; it costs +2 qubits for a peptide
    with one aromatic pair, against +10 to move a 10-mer from 4 to 8 backbone states for
    0.04 A of ceiling. Appending rather than interleaving matters: search moves and the
    ceiling search address backbone slots as `i * bits_per_residue`, and both must keep
    working untouched.
    """
    seq = "GYDPETGTWG"           # Y and W -> 2 chi bits; G/D/P/E/T get none
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    ok = (r.chi_residues == (1, 8))
    ok &= (r.n_chi_bits == 2) and (r.n_bits == 2 * len(seq) + 2)
    # His is aromatic but its ring is not implemented, so it gets no chi bit
    r_h = reps.TorsionStateRepresentation(3, n_states=4, sequence="AHA")
    ok &= (r_h.n_chi_bits == 0)
    # rotamer index 0 must reproduce sidechains.CHI_ANGLES exactly
    chi0 = r.chi1_degrees(r.bitstring_from_states([0] * len(seq)))
    ok &= all(abs(chi0[i] - sc.CHI_ANGLES[geo.ONE_TO_THREE[seq[i]]][0]) < 1e-9
              for i in r.chi_residues)
    chi1 = r.chi1_degrees(r.bitstring_from_states([0] * len(seq), chi_states=[1, 1]))
    ok &= all(abs(chi1[i] - 180.0) < 1e-9 for i in r.chi_residues)
    return _report("chi1 encoded for buildable aromatics only", ok,
                   f"{seq}: chi residues {r.chi_residues}, "
                   f"{r.n_backbone_bits} + {r.n_chi_bits} = {r.n_bits} bits")


def test_chi1_moves_rings_but_not_ca() -> bool:
    """Flipping a chi bit must change the energy and leave every CA where it was.

    This is the property that makes chi1 bits cheap to add: CA-RMSD, the representation
    ceiling, and every backbone metric are untouched, so the extra dimensions cost qubits
    and nothing else. It is also why `representation_ceiling` can ignore them.
    """
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    # Scanned over random backbones rather than one fixed state: on a fully extended
    # chain the two rings sit ~24 A apart, both aromatic wells evaluate to ~1e-144 and
    # ring steric to 0, so chi1 correctly cannot move the energy there. The property
    # under test is that it moves the energy *somewhere*, and never moves a CA.
    rng = np.random.default_rng(0)
    ca_same, moved, best_de = True, 0.0, 0.0
    for _ in range(200):
        backbone = list(rng.integers(0, 4, size=len(seq)))
        base = r.bitstring_from_states(backbone, chi_states=[0, 0])
        flipped = r.bitstring_from_states(backbone, chi_states=[1, 1])
        ca_same &= bool(np.allclose(r.build_coords(base)["CA"],
                                    r.build_coords(flipped)["CA"], atol=1e-12))
        rings_a, rings_b = r.build_rings(base), r.build_rings(flipped)
        moved = max(moved, max(float(np.linalg.norm(rings_a[i][nm] - rings_b[i][nm]))
                               for i in rings_a for nm in rings_a[i]))
        best_de = max(best_de, abs(H.energy(base) - H.energy(flipped)))
    ok = ca_same and moved > 0.5 and best_de > 1e-9
    return _report("chi1 moves ring atoms, not CA positions", ok,
                   f"CA identical={ca_same} over 200 backbones, max ring shift "
                   f"{moved:.2f} A, max |dE| {best_de:.4f}")


def test_aromatic_term_is_orientation_dependent() -> bool:
    """With real rings the aromatic term must distinguish stacking geometries.

    The CB-CB proxy cannot: it sees only a distance, so a stacked pair and a pair whose
    rings point away from each other score identically. Orientation is exactly what
    separates a 2 A prediction from a 4 A one, which is why chi1 had to be encoded before
    this term could be sharpened.
    """
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(40):
        backbone = list(rng.integers(0, 4, size=len(seq)))
        vals = []
        for chi in ([0, 0], [0, 1], [1, 0], [1, 1]):
            bits = r.bitstring_from_states(backbone, chi_states=chi)
            coords = r.build_coords(bits)
            vals.append(et.aromatic_term(seq, coords,
                                         rings=r.build_rings(bits, coords)))
        diffs.append(max(vals) - min(vals))
    spread = float(np.max(diffs))
    # and the proxy path must be blind to chi by construction
    bits = r.bitstring_from_states([1] * len(seq), chi_states=[0, 0])
    bits2 = r.bitstring_from_states([1] * len(seq), chi_states=[1, 1])
    proxy_same = abs(et.aromatic_term(seq, r.build_coords(bits))
                     - et.aromatic_term(seq, r.build_coords(bits2))) < 1e-12
    ok = spread > 1e-6 and proxy_same
    return _report("aromatic term responds to ring orientation", ok,
                   f"max spread over chi combinations {spread:.4f}; "
                   f"CB proxy is chi-blind={proxy_same}")


def test_stacking_geometry_is_rewarded_over_interpenetration() -> bool:
    """Two rings on top of each other must not beat a proper stack.

    Ring atoms now have excluded volume, so interpenetration is penalised -- previously
    `sidechains` built the rings and the energy never looked at them, which left nothing
    stopping two rings occupying the same space.
    """
    ring = np.array([[math.cos(a), math.sin(a), 0.0]
                     for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)]) * 1.39
    names = ("CG", "CD1", "CD2", "CE1", "CE2", "CZ")
    def rings_at(dz, tilt_deg):
        t = math.radians(tilt_deg)
        rot = np.array([[1, 0, 0],
                        [0, math.cos(t), -math.sin(t)],
                        [0, math.sin(t), math.cos(t)]])
        return {0: dict(zip(names, ring)),
                5: dict(zip(names, (ring @ rot.T) + np.array([0.0, 0.0, dz])))}
    stacked = rings_at(4.0, 0.0)
    fused = rings_at(0.3, 0.0)
    c0, n0 = et.ring_frame(np.array(list(stacked[0].values())))
    c1, n1 = et.ring_frame(np.array(list(stacked[5].values())))
    d = float(np.linalg.norm(c0 - c1))
    normal_parallel = abs(float(np.dot(n0, n1))) > 0.99
    seq = "F" * 6
    coords = {"CA": np.zeros((6, 3)), "CB": np.zeros((6, 3)),
              "N": np.zeros((6, 3)), "C": np.zeros((6, 3)), "O": np.zeros((6, 3))}
    e_stacked = et.aromatic_term(seq, coords, rings=stacked)
    e_fused = et.aromatic_term(seq, coords, rings=fused)
    s_stacked = et.steric_term(coords, seq, rings=stacked)
    s_fused = et.steric_term(coords, seq, rings=fused)
    ok = (abs(d - 4.0) < 1e-6 and normal_parallel
          and e_stacked < e_fused                 # stacking is the deeper well
          and s_fused > s_stacked + 1.0)          # interpenetration costs
    return _report("stacking beats interpenetration", ok,
                   f"centroid d {d:.2f} A, aromatic {e_stacked:.3f} vs "
                   f"{e_fused:.3f}, steric {s_stacked:.2f} vs {s_fused:.2f}")


def test_native_energy_uses_the_same_aromatic_geometry() -> bool:
    """The native must be scored with ring geometry too, and with rotamer freedom.

    If predictions used real rings while `energy_from_coords` fell back to the CB proxy,
    the two energies would be computed by different functional forms and `energy_gap` would
    be meaningless. And pinning the native to one rotamer, when the encoding lets a
    prediction choose, would handicap it on a degree of freedom it never got to optimise.
    """
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    bits = r.bitstring_from_states([1] * len(seq))
    coords = r.build_coords(bits)
    phi, psi = r.decode(bits)
    rings = H._rings_from_coords(coords)
    ok = (len(rings) == 2) and all(len(v) >= 6 for v in rings.values())
    scanned = H.energy_from_coords(coords, phi, psi, scan_chi=True)
    pinned0 = H.energy_from_coords(coords, phi, psi, scan_chi=False)
    ok &= scanned <= pinned0 + 1e-9        # the scan can only help
    # and it must agree with the bitstring path when the rotamers match
    direct = et.total_from_components(H.components(bits), H.weights)
    matched = H.energy_from_coords(coords, phi, psi,
                                   chi1=r.chi1_degrees(bits))
    ok &= abs(direct - matched) < 1e-9
    return _report("native energy uses ring geometry and rotamer freedom", ok,
                   f"{len(rings)} rings built; scanned {scanned:.4f} <= "
                   f"pinned {pinned0:.4f}; bitstring path agrees to "
                   f"{abs(direct - matched):.2e}")


def test_compactness_is_one_sided() -> bool:
    """Compactness must penalise expansion but not compaction."""
    n = 12
    ca_tight = np.random.default_rng(0).normal(scale=1.0, size=(n, 3))
    ca_extended = np.stack([np.arange(n) * 3.3, np.zeros(n), np.zeros(n)], axis=1)
    tight = et.compactness_term(ca_tight, n)
    ext = et.compactness_term(ca_extended, n)
    ok = (tight == 0.0) and (ext > 1.0)
    return _report("compactness restraint is one-sided", ok,
                   f"over-compact {tight:.3f}, extended {ext:.3f}")


def test_energy_is_finite_and_deterministic() -> bool:
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
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
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    try:
        H.energy("0" * (r.n_bits + 2))
        return _report("energy rejects wrong-length bitstring", False)
    except ValueError:
        return _report("energy rejects wrong-length bitstring", True)


def test_hbond_rewards_helix() -> bool:
    seq = "AEAAAKEAAAKA"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    helix = H.components(r.bitstring_from_states([0] * len(seq)))["hbond_local"]
    ext = H.components(r.bitstring_from_states([1] * len(seq)))["hbond_local"]
    ok = helix < ext
    return _report("H-bond term favours helix over extended", ok,
                   f"helix {helix:.2f} vs extended {ext:.2f} kcal/mol")


# ==========================================================================
# Objective quality -- the group that was missing
# ==========================================================================
def test_coupled_terms_outweigh_the_separable_one() -> bool:
    """The fold-aware terms must carry more variance than the separable one.

    This is the mechanism behind the anti-correlation, expressed as a test.

    `torsion` is a sum of independent per-residue penalties, so its minimum is found
    residue by residue and a landscape it dominates has no fold in it *at any weight*.
    Pre-rewrite it outvoted `contact` and `hbond` by roughly 8:1 and 23:1, which is why
    the global energy minimum sat 5.54 A from native. (Steric is excluded from the
    accounting: it carried 87% of the variance across all random structures but is exactly
    zero for both native and the global minimum, so among clash-free structures it is
    flat.)

    The test has teeth in both directions: it also asserts that restoring the old
    `torsion` weight of 1.0 *violates* the invariant, so it cannot pass vacuously.
    """
    seqs = ["GYDPETGTWG", "SWTWEGNKWTWK", "AEKAAGKPWYTDENL"]
    stats = eq.term_statistics(seqs, n_sample=300, seed=0)

    shipped = eq.separable_share(stats, et.DEFAULT_WEIGHTS)
    calibrated = eq.separable_share(
        stats, eq.calibrate_weights(seqs, n_sample=300, seed=0))
    # The actual pre-rewrite model: torsion at 1.0, compactness at 0.05, and none of the
    # terms this pass added. Anything less faithful would understate how badly it failed.
    old = eq.separable_share(stats, dict(et.DEFAULT_WEIGHTS,
                                         torsion=1.0, compactness=0.05,
                                         aromatic=0.0, coop_helix=0.0,
                                         coop_sheet=0.0))

    ok = calibrated <= eq.MAX_SEPARABLE_SHARE
    ok &= old > eq.MAX_SEPARABLE_SHARE          # the test is not vacuous
    return _report("coupled terms outweigh the separable one", ok,
                   f"separable share: shipped {100 * shipped:.1f}%, "
                   f"calibrated {100 * calibrated:.1f}%, "
                   f"pre-rewrite weights {100 * old:.1f}% "
                   f"(limit {100 * eq.MAX_SEPARABLE_SHARE:.0f}%)")


def test_calibration_is_sequence_set_derived() -> bool:
    """`calibrate_weights` must not depend on any native structure.

    Structural guard against the failure mode `docs/energy-model-diagnosis.md` warns
    about: with seven weights and one target you can always make native win. The
    calibrator's signature takes sequences only -- no coordinates, no PDB -- so it cannot
    see an answer to fit to. It must also produce the same weights regardless of which
    order the training sequences arrive in.
    """
    params = set(inspect.signature(eq.calibrate_weights).parameters)
    forbidden = {"native_ca", "native", "coords", "entry", "pdb", "target", "rmsd"}
    leaked = params & forbidden
    seqs = ["GYDPETGTWG", "SWTWEGNKWTWK"]
    geo.reset_pdb_log()
    w1 = eq.calibrate_weights(seqs, n_sample=200, seed=0)
    reads = len(geo.get_pdb_log())
    ok = (not leaked) and reads == 0
    ok &= all(w1[t] == et.DEFAULT_WEIGHTS[t]
              for t in et.TERM_NAMES if t not in et.FREE_WEIGHTS)
    return _report("weight calibration cannot see a native structure", ok,
                   f"forbidden args {sorted(leaked)}, {reads} PDB reads, "
                   f"pinned weights unchanged")


def _quality_target():
    """A cached real target for the correlation gates, or None."""
    for pdb_id in ("1UAO", "5AWL", "1LE0"):
        path = os.path.join(ds.PDB_DIR, f"{pdb_id}.pdb")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            seq, ncoords, nphi, npsi = geo.native_coords_from_pdb(path)
            if len(seq) <= 11:      # keep the enumeration tractable
                return pdb_id, seq, ncoords, nphi, npsi
    return None


def test_energy_correlates_with_native_proximity() -> bool:
    """Spearman(energy, RMSD) must be positive.

    The pre-rewrite model scored -0.351: lower energy meant *further* from native, so a
    converged search did worse than random sampling. This is the single most important
    property of the objective and it was untested.
    """
    tgt = _quality_target()
    if tgt is None:
        _skip("test_energy_correlates_with_native_proximity",
              "no small cached PDB in pdbs/ (run --main-comparison once)")
        return True
    pdb_id, seq, ncoords, nphi, npsi = tgt
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    q = eq.energy_quality(H, r, ncoords["CA"], nphi, npsi, n_sample=6000, seed=0)
    ok = q["spearman"] > 0.0
    return _report("energy correlates positively with native proximity", ok,
                   f"{pdb_id}: Spearman {q['spearman']:+.4f} "
                   f"(pre-rewrite model: -0.3505)")


def test_low_energy_structures_are_enriched() -> bool:
    """Optimising the energy must move toward native, not away from it."""
    tgt = _quality_target()
    if tgt is None:
        _skip("test_low_energy_structures_are_enriched", "no small cached PDB")
        return True
    pdb_id, seq, ncoords, nphi, npsi = tgt
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    q = eq.energy_quality(H, r, ncoords["CA"], nphi, npsi, n_sample=6000, seed=0)
    ok = q["enrichment"] > 0.0
    return _report("low-energy structures are enriched toward native", ok,
                   f"{pdb_id}: enrichment {q['enrichment']:+.2f} A "
                   f"(all {q['mean_rmsd_all']:.2f}, low-E "
                   f"{q['mean_rmsd_low_energy']:.2f}; pre-rewrite: -0.25)")


# ==========================================================================
# CVaR
# ==========================================================================
def test_cvar_preserves_multiplicities() -> bool:
    energies = [0.0] + [10.0] * 99
    got = vqe_mod.cvar_from_samples(energies, alpha=0.5)
    expected = (0.0 + 49 * 10.0) / 50.0
    ok = abs(got - expected) < 1e-9
    dedup = vqe_mod.cvar_from_samples(list(set(energies)), alpha=0.5)
    return _report("CVaR preserves sample multiplicities", ok,
                   f"got {got:.4f}, expected {expected:.4f}, "
                   f"dedup would give {dedup:.4f}")


def test_cvar_sampled_converges_to_exact() -> bool:
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


# ==========================================================================
# VQE structure and reproducibility
# ==========================================================================
def test_vqe_is_full_system() -> bool:
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    circuit = vqe_mod.build_global_circuit(H.n_qubits, layers=2)
    probs = np.asarray(circuit(np.zeros(vqe_mod.n_parameters(H.n_qubits, 2))))
    ok = (probs.size == 2 ** H.n_qubits)
    names = [n for n in dir(vqe_mod) if "block" in n.lower()]
    ok &= (len(names) == 0)
    return _report("VQE is full-system (no block decomposition)", ok,
                   f"probs size {probs.size} == 2^{H.n_qubits}, "
                   f"block-named symbols: {names}")


def test_vqe_does_not_enumerate() -> bool:
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = vqe_mod.run_global_cvar_vqe(H, layers=1, shots=256, maxiter=120,
                                          restarts=1, seed=0, final_shots=512)
    space = 2 ** H.n_qubits
    ok = res["n_energy_evaluations"] < 0.25 * space
    return _report("VQE does not enumerate the configuration space", ok,
                   f"{res['n_energy_evaluations']} evals vs {space} configs")


def test_final_answer_comes_from_final_circuit() -> bool:
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = vqe_mod.run_global_cvar_vqe(H, layers=1, shots=256, maxiter=120,
                                          restarts=1, seed=0, final_shots=512)
    ok = ("vqe_bitstring" in res and "best_seen_bitstring" in res
          and res["vqe_energy"] >= res["best_seen_energy"] - 1e-9
          and len(res["vqe_bitstring"]) == H.n_bits)
    return _report("final VQE answer distinct from best-seen", ok,
                   f"vqe {res['vqe_energy']:.3f}, "
                   f"best-seen {res['best_seen_energy']:.3f}")


def test_reproducibility() -> bool:
    seq = "GYDPETGTWG"

    def run(s):
        r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
        H = ham.FoldingHamiltonian(seq, r)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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
    s0 = np.random.default_rng(np.random.SeedSequence([42, 0])).random(5)
    s1 = np.random.default_rng(np.random.SeedSequence([42, 1])).random(5)
    s0b = np.random.default_rng(np.random.SeedSequence([42, 0])).random(5)
    ok = (not np.allclose(s0, s1)) and np.allclose(s0, s0b)
    return _report("RNG streams independent across evals, stable across runs", ok)


# ==========================================================================
# Budget
# ==========================================================================
def test_spsa_costs_two_evaluations_per_iteration() -> bool:
    """SPSA must spend exactly 2 objective evaluations per iteration.

    The previous version spent 1 + 3n: an initial call plus a third call at the updated `x`
    purely for best-tracking. The midpoint `(fp + fm) / 2` is an unbiased estimate of f(x)
    with *lower* variance than a fresh evaluation, so that call bought nothing and cost a
    third of the iteration budget.
    """
    calls = {"n": 0}

    def obj(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x, dtype=float) ** 2))

    n_iter = 7
    res = vqe_mod._spsa(obj, np.ones(5), n_iter, np.random.default_rng(0))
    ok = (calls["n"] == 2 * n_iter)
    ok &= hasattr(res, "x") and hasattr(res, "fun")
    ok &= np.isfinite(res.fun)
    return _report("SPSA costs two evaluations per iteration", ok,
                   f"{calls['n']} evaluations over {n_iter} iterations "
                   f"(old form would be {1 + 3 * n_iter})")


def test_spsa_uses_common_random_numbers() -> bool:
    """Both arms of SPSA's finite difference must share one sample stream.

    The CVaR objective is a sampled estimate. If the +/- arms draw independently, the
    difference `fp - fm` carries twice the noise variance of a single evaluation -- the worst
    case for a finite-difference gradient. Pinning one stream per iteration makes the noise
    largely cancel in the difference, while varying it across iterations keeps the estimator
    unbiased. The tag must also be released afterwards, so the final read-out is not stuck
    on one stream.
    """
    tags, seen = [], []

    def pin(t):
        tags.append(t)

    def obj(x):
        seen.append(tags[-1] if tags else "unset")
        return float(np.sum(np.asarray(x, dtype=float) ** 2))

    n_iter = 4
    vqe_mod._spsa(obj, np.zeros(3), n_iter, np.random.default_rng(0),
                  set_sample_tag=pin)
    # one pin per iteration, plus one release at the end
    ok = (tags == [0, 1, 2, 3, None])
    # both evaluations of an iteration saw the same tag, and consecutive iterations differ
    pairs = [seen[2 * k:2 * k + 2] for k in range(n_iter)]
    ok &= all(p[0] == p[1] == k for k, p in enumerate(pairs))
    return _report("SPSA uses common random numbers across the +/- pair", ok,
                   f"tags {tags}; per-iteration arm tags {pairs}")


def test_spsa_schedule_is_driven_by_progress() -> bool:
    """SPSA's gain schedule must follow budget consumed, not the raw iteration index.

    `experiments.run_one` sets `maxiter` far above what the shared budget affords so the
    budget is what terminates the search. Indexing the schedule by `k` then means `A` is
    calibrated to a horizon the run never reaches -- at `maxiter = 50 x n_params = 4000`,
    `A = 400` and the first step is `a / 401**0.602 = 0.007`, five times too small, with a
    decay that never completes. Exactly the defect that made simulated annealing never cool.

    Checked by driving `progress` from 0 to 1 over few actual iterations and asserting the
    step size falls by the full amount the schedule intends.
    """
    n_iter, n_real, dim = 4000, 6, 4

    class _Stop(Exception):
        pass

    def run(with_progress: bool):
        """Recover the perturbation size ck actually used, per iteration.

        Each iteration evaluates `x + ck*d` and `x - ck*d` with d entries in {-1, +1}, so
        `|arg_plus - arg_minus| / (2*sqrt(dim))` recovers ck exactly. That observes the
        schedule the implementation used rather than re-deriving the formula.

        `_spsa` is handed the full `maxiter` and cut off after `n_real` iterations, which
        is exactly what the shared budget does in a real run. That gap between the
        allowance and the iterations actually executed IS the defect under test -- pass
        `n_real` as `n_iter` and the two schedules become identical by construction.
        """
        seen = []

        def obj(x):
            if len(seen) >= 2 * n_real:
                raise _Stop
            seen.append(np.array(x, dtype=float))
            return float(np.sum(np.asarray(x, dtype=float) ** 2))

        def prog():
            return min(1.0, (len(seen) // 2) / float(n_real))

        try:
            vqe_mod._spsa(obj, np.zeros(dim), n_iter, np.random.default_rng(0),
                          progress=prog if with_progress else None)
        except _Stop:
            pass
        return [float(np.linalg.norm(seen[2 * k] - seen[2 * k + 1])
                      / (2.0 * math.sqrt(dim))) for k in range(n_real)]

    ck_prog = run(True)
    ck_index = run(False)
    decay_prog = ck_prog[0] / ck_prog[-1]
    decay_index = ck_index[0] / ck_index[-1]
    # Driven by progress the schedule traverses its whole intended range in `n_real`
    # iterations; driven by the raw index it barely moves, because it is calibrated to a
    # horizon of n_iter=1000 that the run never reaches.
    ok = decay_prog > 1.5 * decay_index
    ok &= all(b <= a + 1e-12 for a, b in zip(ck_prog, ck_prog[1:]))   # monotone decay
    return _report("SPSA schedule is driven by budget progress", ok,
                   f"ck decays {decay_prog:.2f}x over {n_real} iterations when driven by "
                   f"progress vs {decay_index:.2f}x by raw index "
                   f"(maxiter={n_iter} the run never reaches)")


def test_optimizer_guard_covers_spsa() -> bool:
    """The optimizer guard must reject a too-short SPSA run, not only a small COBYLA one.

    SPSA's requirement is an absolute iteration count -- its cost per iteration does not
    grow with the parameter count -- so a per-n_params floor is the wrong test for it. The
    guard previously said nothing at all about SPSA, which meant `maxiter=10` passed.
    """
    ok = True
    detail = []
    try:
        bud.check_optimizer_budget(10, 80, "SPSA", eval_budget=1000)
        ok = False
        detail.append("SPSA maxiter=10 accepted")
    except ValueError:
        pass
    # a generous SPSA allowance must pass, and must not be judged against n_params
    try:
        bud.check_optimizer_budget(bud.MIN_SPSA_ITERATIONS, 5000, "SPSA",
                                   eval_budget=1000)
    except ValueError:
        ok = False
        detail.append("SPSA floor is being scaled by n_params")
    return _report("optimizer guard covers SPSA", ok,
                   "; ".join(detail) if detail else
                   f"floor {bud.MIN_SPSA_ITERATIONS} iterations, absolute not per-param")


def test_default_optimizer_suits_a_stochastic_objective() -> bool:
    """The default must be the stochastic-approximation optimizer, with the other kept.

    The CVaR objective re-samples on every call, so the same parameters give different
    values. COBYLA is a trust-region method that requires a reproducible objective; SPSA's
    convergence theory is for noisy ones. COBYLA stays selectable so the choice is
    measurable -- `experiment_vqe_hyperparameters` runs both under one shared budget.
    """
    cfg = exp.default_vqe_config()
    ok = (cfg["optimizer"] == "SPSA")
    src = inspect.getsource(vqe_mod._run_single)
    ok &= "COBYLA" in src               # still reachable
    ok &= "sample_tag" in src           # the CRN mechanism is wired
    return _report("default optimizer suits a stochastic objective", ok,
                   f"default {cfg['optimizer']}, COBYLA still selectable")


def test_one_layer_direct_sampler_is_exact() -> bool:
    """The fast sampler must reproduce the dense circuit probability distribution."""
    from direct_sampler import OneLayerDirectSampler

    ok = True
    worst = 0.0
    rng = np.random.default_rng(2217)
    for ring in (False, True):
        for n_qubits in (2, 3, 5, 8):
            params = rng.normal(0.2, 1.7, size=n_qubits)
            dense = np.asarray(
                vqe_mod.build_global_circuit(n_qubits, 1, ring=ring)(params))
            direct = OneLayerDirectSampler(
                n_qubits, layers=1, ring=ring).probs(params)
            err = float(np.max(np.abs(dense - direct)))
            worst = max(worst, err)
            ok &= err < 2e-12
    try:
        OneLayerDirectSampler(6, layers=2)
        ok = False
    except ValueError:
        pass
    return _report("one-layer direct sampler is distribution-exact", ok,
                   f"worst probability error {worst:.2e}")


def test_vqe_exposes_sampler_batch_and_trace_hooks() -> bool:
    """Dense and direct-sampling studies must use the same optimizer implementation."""
    params = set(inspect.signature(vqe_mod.run_global_cvar_vqe).parameters)
    needed = {"sampler", "energy_batch", "spsa_a", "spsa_c", "trace_interval"}
    src = inspect.getsource(vqe_mod.run_global_cvar_vqe)
    ok = needed <= params and '"objective_trace"' in src and '"backend"' in src
    return _report("VQE exposes sampler, batch and trace hooks", ok,
                   f"hooks present: {sorted(needed & params)}")


def test_cem_respects_the_shared_budget() -> bool:
    """The strongest classical distribution baseline must use the same hard allowance."""
    seq = "GYDPETGTWG"
    rep = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, rep, eval_budget=80)
    result = cb.cross_entropy_search(
        H, n_samples=2000, batch_size=16, seed=4, trace_interval=20)
    trace_ok = bool(result["trace"]) and all(
        int(point["n_energy_evaluations"]) <= 80 for point in result["trace"])
    ok = 0 < H.n_energy_evaluations <= 80 and trace_ok
    return _report("CEM respects the shared evaluation budget", ok,
                   f"spent {H.n_energy_evaluations}/80; "
                   f"{len(result['trace'])} trace points")


def test_budget_is_enforced() -> bool:
    """The budget is a hard ceiling, not a hint."""
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r, eval_budget=200)
    rng = np.random.default_rng(0)
    raised = False
    for _ in range(2000):
        try:
            H.energy(r.random_bitstring(rng))
        except bud.BudgetExhausted:
            raised = True
            break
    ok = raised and H.n_energy_evaluations == 200
    return _report("budget_is_enforced", ok,
                   f"raised={raised}, spent {H.n_energy_evaluations}/200")


def test_arms_are_cost_matched() -> bool:
    """No arm may exceed the shared allowance, and the annealer must use it."""
    seq = "GYDPETGTWG"
    budget = 1500
    spends = {}
    for name in ("vqe", "sa", "random"):
        r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
        H = ham.FoldingHamiltonian(seq, r, eval_budget=budget)
        if name == "vqe":
            vqe_mod.run_global_cvar_vqe(H, layers=1, shots=128, maxiter=None,
                                        restarts=2, seed=0, final_shots=256)
        elif name == "sa":
            cb.simulated_annealing(H, n_steps=20 * budget, seed=0)
        else:
            cb.random_search(H, n_samples=2 * budget, seed=0)
        spends[name] = H.n_energy_evaluations
    ok = all(v <= budget for v in spends.values())
    ok &= spends["sa"] >= 0.90 * budget
    ok &= spends["random"] >= 0.90 * budget
    return _report("arms_are_cost_matched", ok,
                   ", ".join(f"{k} {v}" for k, v in spends.items())
                   + f" of {budget}")


def test_restarts_survive_shared_budget() -> bool:
    """Every restart must run; none may swallow the pool."""
    seq = "AEKAAG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r, eval_budget=3000)
    res = vqe_mod.run_global_cvar_vqe(H, layers=2, shots=64, maxiter=None,
                                      restarts=4, seed=0, final_shots=256)
    ok = res["restarts_completed"] == 4
    return _report("restarts_survive_shared_budget", ok,
                   f"{res['restarts_completed']}/4 restarts completed, "
                   f"{H.n_energy_evaluations} evals spent")


def test_optimizer_budget_guard_rejects_simplex_only() -> bool:
    """Configurations that can only build a COBYLA simplex must be rejected.

    Both historical cases: `run_8state_seed0.py` at 120 params / maxiter=200, and 1LE0 in
    the old `main_comparison.csv` at 96 params / maxiter=150.
    """
    ok = True
    detail = []
    for maxiter, n_params, tag in ((200, 120, "8state seed0"), (150, 96, "1LE0")):
        try:
            bud.check_optimizer_budget(maxiter, n_params, "COBYLA", eval_budget=1000)
            ok = False
            detail.append(f"{tag} accepted")
        except ValueError:
            pass
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bud.check_optimizer_budget(4000, 80, "COBYLA", eval_budget=None)
        if not any(issubclass(x.category, RuntimeWarning) for x in w):
            ok = False
            detail.append("no warning when maxiter binds without a shared budget")
    return _report("optimizer_budget_guard_rejects_simplex_only", ok,
                   "; ".join(detail) if detail else
                   "both historical configs rejected, unbudgeted run warns")


def test_annealing_actually_cools() -> bool:
    """SA must reach a low temperature within the budget that terminates it.

    `experiments.run_one` sets n_steps = 20 x eval_budget so the budget binds, but the
    schedule was a function of the step index -- so the run ended at 3.67 of a starting
    4.0, i.e. 92% of t_start, and the arm was a random walk with a best-seen memory
    rather than an annealer. Scheduling on budget consumed is the fix.
    """
    seq = "GYDPETGTWG"
    budget = 800
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r, eval_budget=budget)
    res = cb.simulated_annealing(H, n_steps=20 * budget, t_start=4.0, seed=0)
    ok = (res["anneal_on"] == "budget") and (res["final_temperature"] < 0.5)
    return _report("annealing_actually_cools", ok,
                   f"schedule on {res['anneal_on']}, final T "
                   f"{res['final_temperature']:.4f} of t_start 4.0 "
                   f"(step-scheduled would end near 3.67)")


# ==========================================================================
# Leakage
# ==========================================================================
def test_no_pdb_access_during_optimization() -> bool:
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    geo.reset_pdb_log()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vqe_mod.run_global_cvar_vqe(H, layers=1, shots=128, maxiter=120,
                                    restarts=1, seed=0, final_shots=128)
    cb.simulated_annealing(H, n_steps=500, seed=0)
    log = geo.get_pdb_log()
    ok = (len(log) == 0)
    return _report("no PDB access during optimization (leakage audit)", ok,
                   f"{len(log)} PDB reads")


def test_hamiltonian_has_no_native_input() -> bool:
    params = set(inspect.signature(ham.FoldingHamiltonian.__init__).parameters)
    forbidden = {"pdb", "pdb_path", "native", "native_coords", "coords",
                 "entry", "structure", "target"}
    leaked = params & forbidden
    return _report("Hamiltonian constructor takes no native structure",
                   not leaked, str(sorted(leaked)) if leaked else "")


# ==========================================================================
# Metrics
# ==========================================================================
def test_rmsd_is_angstroms() -> bool:
    r = reps.TorsionStateRepresentation(12, n_states=4, sequence="A" * 12)
    helix = r.build_coords(r.bitstring_from_states([0] * 12))["CA"]
    strand = r.build_coords(r.bitstring_from_states([1] * 12))["CA"]
    self_r = geo.ca_rmsd(helix, helix)
    shift_r = geo.ca_rmsd(helix + 5.0, helix)
    cross_r = geo.ca_rmsd(helix, strand)
    ok = (self_r < 1e-9 and shift_r < 1e-9 and 3.0 < cross_r < 30.0)
    return _report("RMSD in Angstroms, translation/rotation invariant", ok,
                   f"self {self_r:.2e}, helix-vs-strand {cross_r:.2f} A")


def test_no_normalized_rmsd_path() -> bool:
    ok = not hasattr(geo, "normalize_coords")
    src = inspect.getsource(ev)
    ok &= ("normalize_coords" not in src)
    return _report("no dimensionless normalized-RMSD path exists", ok)


def test_ensemble_rmsd_is_min_over_models() -> bool:
    """NMR targets must be scored against every deposited model.

    `parse_pdb` used to take the first model and stop, so RMSD was measured against one
    arbitrary draw from an ensemble whose own members differ by several tenths of an
    Angstrom. This is a reporting correction, not a prediction improvement.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(size=(10, 3)) * 3.0
    models = [base + rng.normal(size=(10, 3)) * 0.4 for _ in range(5)]
    pred = base + rng.normal(size=(10, 3)) * 0.5
    ens = geo.ca_rmsd_to_ensemble(pred, models)
    spread = geo.ensemble_spread(models)
    ok = (ens["min"] <= ens["model1"] + 1e-12
          and ens["min"] <= ens["mean"] + 1e-12
          and ens["n_models"] == 5
          and spread > 0.0)
    return _report("ensemble RMSD reports min over deposited models", ok,
                   f"min {ens['min']:.3f} <= model1 {ens['model1']:.3f}, "
                   f"spread {spread:.3f} A")


def test_evaluate_structure_rejects_length_mismatch() -> bool:
    """A native/representation length mismatch must raise, not silently truncate."""
    seq = "GYDPETGTWG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    coords = r.build_coords(r.bitstring_from_states([1] * len(seq)))
    phi, psi = r.decode(r.bitstring_from_states([1] * len(seq)))
    try:
        ev.evaluate_structure(r.bitstring_from_states([0] * len(seq)), r, H,
                              seq + "AAA", coords, phi, psi)
        return _report("evaluate_structure rejects length mismatch", False,
                       "truncated silently")
    except ValueError:
        return _report("evaluate_structure rejects length mismatch", True)


def test_well_determined_mask_is_data_derived() -> bool:
    """The structured core must come from the ensemble, never from a hardcoded range."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(12, 3)) * 3.0
    models = []
    for _ in range(8):
        m = base.copy()
        m[:2] += rng.normal(size=(2, 3)) * 4.0      # floppy N-terminus
        m[-2:] += rng.normal(size=(2, 3)) * 4.0     # floppy C-terminus
        m[2:-2] += rng.normal(size=(8, 3)) * 0.15   # ordered core
        models.append(m)
    mask = ev.well_determined_mask(models)
    ok = bool(mask[2:-2].all() and not mask[0] and not mask[-1])
    return _report("well-determined core is derived from the ensemble", ok,
                   f"{int(mask.sum())}/12 residues flagged ordered")


def test_identity_is_alignment_based() -> bool:
    """Clustering identity must penalise gaps.

    It was a longest-common-subsequence ratio, which charges nothing for gaps: a short
    sequence whose residues merely appear in order inside a longer one scored as highly
    similar. Since identity clustering is the only thing keeping
    `energy_quality.holdout_report` honest, a permissive measure leaks train sequences
    into the held-out set.
    """
    a, b = "ACDEFGHIKL", "AXCXDXEXFXGXHXIXKXL"
    ident = ds._identity(a, b)
    ok = ident < 0.9
    ok &= abs(ds._identity(a, a) - 1.0) < 1e-9
    ok &= ds._identity("AAAAAAAAAA", "WWWWWWWWWW") < 0.1
    return _report("cluster identity is alignment-based, not LCS", ok,
                   f"gapped pair scores {ident:.2f} (LCS would give 1.00)")


def test_dataset_rejects_unrebuildable_native() -> bool:
    """A native the torsion builder cannot reproduce must not enter the evidence set.

    `build_dataset` used to filter on sequence length alone and discard the coordinates it
    had just parsed, so 1B03 -- a cis X-Pro peptide bond that the trans-only builder rebuilds
    as a ~180 deg mid-chain hinge -- sat in every calibration and held-out report occupying
    one of ~10 identity clusters while carrying an ~11 A RMSD floor nothing could clear. The
    gate rebuilds each native from its own torsions and rejects it past
    `REBUILD_RMSD_TOLERANCE`. Both directions matter: 1B03 out, a buildable native (1UAO,
    which rebuilds at 0.18 A) still in, and disabling the gate restores the raw set.
    """
    have = {p: os.path.join(ds.PDB_DIR, f"{p}.pdb") for p in ("1B03", "1UAO")}
    if not all(os.path.exists(p) and os.path.getsize(p) > 0 for p in have.values()):
        _skip("test_dataset_rejects_unrebuildable_native",
              "1B03/1UAO not cached in pdbs/ (run --main-comparison once)")
        return True
    bb_bad = ds.rebuild_backbone_rmsd(have["1B03"])
    bb_ok = ds.rebuild_backbone_rmsd(have["1UAO"])
    ok = bb_bad is not None and bb_bad > ds.REBUILD_RMSD_TOLERANCE
    ok &= bb_ok is not None and bb_ok < ds.REBUILD_RMSD_TOLERANCE
    gated = {e.pdb_id for e in
             ds.build_dataset(pdb_ids=["1UAO", "1B03"], verbose=False)}
    raw = {e.pdb_id for e in ds.build_dataset(pdb_ids=["1UAO", "1B03"],
                                              max_rebuild_rmsd=None, verbose=False)}
    ok &= gated == {"1UAO"} and raw == {"1UAO", "1B03"}
    return _report("dataset rejects natives it cannot rebuild", ok,
                   f"1B03 rebuilds at {bb_bad:.2f} A (rejected), "
                   f"1UAO at {bb_ok:.2f} A (kept)")


# ==========================================================================
# Search correctness
# ==========================================================================
def test_vqe_matches_exhaustive_on_tiny_system() -> bool:
    seq = "AEKAAG"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, r)
    exact = cb.exhaustive_search(H, max_bits=14)

    H2 = ham.FoldingHamiltonian(
        seq, reps.TorsionStateRepresentation(len(seq), 4, sequence=seq))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = vqe_mod.run_global_cvar_vqe(H2, layers=3, alpha=0.15, shots=1024,
                                          maxiter=400, restarts=5, seed=0,
                                          final_shots=8192)

    H3 = ham.FoldingHamiltonian(
        seq, reps.TorsionStateRepresentation(len(seq), 4, sequence=seq))
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
    seq = "SWTWEGNKWTWK"
    r = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
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
    seq = "GYDPETGTWG"
    r_t = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
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


def test_ceiling_cache_keys_on_residue_classes() -> bool:
    """Two representations differing only in composition must not share a ceiling.

    The memo key previously covered (name, n_bits, n_states, seed, iterations, native);
    now that libraries are per-residue-class, two same-length sequences have different
    reachable sets and must not collide.
    """
    n = 10
    # An alpha-L helix: exactly representable in the Gly library (GLY_4 state 2) and not
    # in the general one, which spends no state on positive phi. A native the two
    # libraries approximate *differently* is what makes a shared cache entry detectable
    # at all -- with a right-handed helix, GLY_4[0] and GENERAL_4[0] are the same state
    # and both score 0.000 whether or not the cache collides.
    phi = np.full(n, math.radians(60.0))
    psi = np.full(n, math.radians(45.0))
    native = geo.build_backbone(phi, psi)["CA"]
    a = ev.representation_ceiling(
        reps.TorsionStateRepresentation(n, 4, sequence="A" * n),
        native, phi, psi, seed=0, iterations=400)
    b = ev.representation_ceiling(
        reps.TorsionStateRepresentation(n, 4, sequence="G" * n),
        native, phi, psi, seed=0, iterations=400)
    ok = (b["ceiling_ca_rmsd"] < 1e-6
          and a["ceiling_ca_rmsd"] - b["ceiling_ca_rmsd"] > 0.5)
    return _report("ceiling cache distinguishes residue-class libraries", ok,
                   f"poly-A {a['ceiling_ca_rmsd']:.3f} A vs "
                   f"poly-G {b['ceiling_ca_rmsd']:.3f} A")


# ==========================================================================
# Sidechains
# ==========================================================================
_SC_PROBE_SEQ = "GASTDENKPFYW"


def _sc_backbones():
    r = reps.TorsionStateRepresentation(len(_SC_PROBE_SEQ), n_states=4,
                                        sequence=_SC_PROBE_SEQ)
    return [r.build_coords(r.bitstring_from_states([s] * len(_SC_PROBE_SEQ)))
            for s in range(4)]


def _sc_residue_atoms(key, bb, i):
    res = {"N": bb["N"][i], "CA": bb["CA"][i], "C": bb["C"][i], "O": bb["O"][i]}
    res.update(sc.build_sidechain(key, bb["N"][i], bb["CA"][i], bb["C"][i],
                                  bb["CB"][i]))
    return res


def _sc_bond_distances(key):
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
    "PHE": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.502, ("CG", "CD1"): 1.389,
            ("CG", "CD2"): 1.389, ("CD1", "CE1"): 1.382, ("CD2", "CE2"): 1.382,
            ("CE1", "CZ"): 1.378, ("CE2", "CZ"): 1.378},
    "TYR": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.512, ("CG", "CD1"): 1.389,
            ("CG", "CD2"): 1.389, ("CD1", "CE1"): 1.382, ("CD2", "CE2"): 1.382,
            ("CE1", "CZ"): 1.378, ("CE2", "CZ"): 1.378, ("CZ", "OH"): 1.376},
    "TRP": {("CA", "CB"): 1.5295, ("CB", "CG"): 1.498, ("CG", "CD1"): 1.365,
            ("CG", "CD2"): 1.433, ("CD1", "NE1"): 1.375, ("NE1", "CE2"): 1.371,
            ("CE2", "CD2"): 1.409, ("CD2", "CE3"): 1.398, ("CE3", "CZ3"): 1.382,
            ("CZ3", "CH2"): 1.400, ("CH2", "CZ2"): 1.368, ("CZ2", "CE2"): 1.394},
}


def test_sidechain_bond_lengths() -> bool:
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
    rings = {"PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
             "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
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
    probe = ([0.0, 0.0, 0.0], [1.458, 0.0, 0.0], [2.0, 1.42, 0.0],
             [1.99, -0.77, -1.21])
    ok = True
    detail = []
    # PHE is now implemented (it reuses the verified TYR ring template minus the hydroxyl),
    # so it is deliberately absent from this list.
    for one, three in (("C", "CYS"), ("H", "HIS"), ("M", "MET"),
                       ("V", "VAL"), ("I", "ILE"), ("R", "ARG")):
        for name in (one, three):
            try:
                sc.build_sidechain(name, *probe)
                ok = False
                detail.append(f"{name} did not raise")
            except NotImplementedError as exc:
                if three not in str(exc):
                    ok = False
                    detail.append(f"{name} message lacks {three}")
    # and PHE must build
    try:
        atoms = sc.build_sidechain("PHE", *probe)
        ok &= set(sc.ring_atom_names("PHE")).issubset(atoms)
        ok &= "OH" not in atoms
    except Exception as exc:
        ok = False
        detail.append(f"PHE failed: {type(exc).__name__}")
    return _report("unimplemented residues raise NotImplementedError", ok,
                   "; ".join(detail) if detail else
                   "C, H, M, V, I, R rejected by both codes; PHE builds")


def test_sidechain_atom_counts() -> bool:
    expected = {"GLY": 4, "ALA": 5, "SER": 6, "THR": 7, "PRO": 7, "ASP": 8,
                "ASN": 8, "GLU": 9, "LYS": 9, "PHE": 11, "TYR": 12, "TRP": 14}
    ok = True
    bad = []
    for key, n in expected.items():
        if sc.heavy_atom_count(key) != n:
            ok = False
            bad.append(f"{key} {sc.heavy_atom_count(key)}!={n}")
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


# ==========================================================================
# Amber (skipped without openmm)
# ==========================================================================
_AMBER_SEQ = "GYDPETGTWG"
_AMBER_CACHE = {}


def _amber_pdb_path(pdb_id: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "pdbs", f"{pdb_id}.pdb")


def _amber_hamiltonian():
    if "H" not in _AMBER_CACHE:
        rep = reps.TorsionStateRepresentation(len(_AMBER_SEQ), n_states=4,
                                              sequence=_AMBER_SEQ)
        _AMBER_CACHE["rep"] = rep
        _AMBER_CACHE["H"] = amb.AmberHamiltonian(_AMBER_SEQ, rep)
    return _AMBER_CACHE["H"], _AMBER_CACHE["rep"]


def _amber_native():
    if "native" not in _AMBER_CACHE:
        _AMBER_CACHE["native"] = geo.native_coords_from_pdb(
            _amber_pdb_path("1UAO"))
    return _AMBER_CACHE["native"]


def test_amber_energy_is_deterministic() -> bool:
    H, rep = _amber_hamiltonian()
    rng = np.random.default_rng(3)
    worst = 0.0
    for bits in ([rep.bitstring_from_states([0] * len(_AMBER_SEQ))]
                 + [rep.random_bitstring(rng) for _ in range(2)]):
        vals = []
        for _ in range(5):
            H._cache.clear()
            vals.append(H.energy(bits))
        if not all(math.isfinite(v) for v in vals):
            return _report("Amber energy deterministic to 1e-12", False,
                           "non-finite energy")
        worst = max(worst, max(vals) - min(vals))
    ok = worst < 1e-12
    return _report("Amber energy deterministic to 1e-12", ok,
                   f"max spread {worst:.3e} kcal/mol")


def test_amber_native_below_helix() -> bool:
    H, rep = _amber_hamiltonian()
    _, coords, phi, psi = _amber_native()
    e_native = H.energy_from_coords(coords, phi, psi)
    e_helix = H.energy(rep.bitstring_from_states([0] * len(_AMBER_SEQ)))
    ok = e_native < e_helix
    return _report("native scores below all-helix under Amber", ok,
                   f"native {e_native:.2f} vs helix {e_helix:.2f} kcal/mol "
                   f"(gap {e_native - e_helix:.2f})")


def test_amber_backbone_restraints_hold() -> bool:
    H, rep = _amber_hamiltonian()
    rng = np.random.default_rng(11)
    worst, where = 0.0, ""
    for bits in ([rep.bitstring_from_states([0] * len(_AMBER_SEQ))]
                 + [rep.random_bitstring(rng) for _ in range(3)]):
        pre, post = H.minimized_ca(bits)
        d = geo.rmsd(pre, post)
        if d > worst:
            worst, where = d, bits
    ok = worst < 0.5
    return _report("backbone restraints survive minimization", ok,
                   f"max CA-RMSD pre->post {worst:.3f} A (worst {where})")


def test_amber_rejects_wrong_length() -> bool:
    H, rep = _amber_hamiltonian()
    ok = True
    detail = []
    for bad in ("0" * (rep.n_bits + 2), "0" * (rep.n_bits - 1), ""):
        try:
            H.energy(bad)
            ok = False
            detail.append(f"len {len(bad)} accepted")
        except ValueError:
            pass
    return _report("Amber energy rejects wrong-length bitstring", ok,
                   "; ".join(detail) if detail else
                   "too long, too short and empty all raise ValueError")


def test_amber_no_pdb_access_during_energy() -> bool:
    rep = reps.TorsionStateRepresentation(len(_AMBER_SEQ), n_states=4,
                                          sequence=_AMBER_SEQ)
    geo.reset_pdb_log()
    H = amb.AmberHamiltonian(_AMBER_SEQ, rep)
    rng = np.random.default_rng(0)
    for _ in range(3):
        H.energy(rep.random_bitstring(rng))
    H.components(rep.bitstring_from_states([0] * len(_AMBER_SEQ)))
    log = geo.get_pdb_log()
    params = set(inspect.signature(amb.AmberHamiltonian.__init__).parameters)
    forbidden = {"pdb", "pdb_path", "native", "native_coords", "coords",
                 "entry", "structure", "target"}
    leaked = params & forbidden
    ok = (len(log) == 0) and not leaked
    return _report("no PDB access during Amber construction or .energy()", ok,
                   f"{len(log)} PDB reads, no native args in signature")


# ==========================================================================
def run_all() -> bool:
    print("=" * 72)
    print("VALIDATION SUITE")
    print("=" * 72)
    tests = [
        # representation and geometry
        test_bitstring_and_coordinate_lengths,
        test_backbone_geometry_is_physical,
        test_helix_is_representable,
        test_extended_is_representable,
        test_chirality,
        test_libraries_are_residue_class_specific,
        test_legacy_library_is_byte_stable,
        test_parallel_builder_carries_the_library_setting,
        test_lattice_coordinates_are_in_angstroms,
        test_lattice_steric_is_quiet_for_valid_traces,
        test_umeyama_scale_beats_bond_ratio,
        # energy terms
        test_mj_correction_changes_sign,
        test_burial_scale_places_aromatics_hydrophobic,
        test_hbond_is_not_length_normalised,
        test_hbond_longrange_is_a_separate_calibrated_term,
        test_electrostatics_has_physical_magnitude,
        test_steric_sees_backbone_atoms,
        test_steric_exempts_hbond_partners,
        test_hbond_cannot_pay_for_interpenetration,
        test_build_all_matches_separate_calls,
        test_steric_layout_cache_is_coordinate_independent,
        test_solvation_matches_naive_pair_sum,
        test_cooperativity_distinguishes_runs_from_scattered_bonds,
        test_cooperativity_is_topology_symmetric,
        test_coop_helix_run_is_capped,
        test_cooperativity_reaches_the_energy,
        test_chi1_is_encoded_for_aromatics_only,
        test_chi1_moves_rings_but_not_ca,
        test_aromatic_term_is_orientation_dependent,
        test_stacking_geometry_is_rewarded_over_interpenetration,
        test_native_energy_uses_the_same_aromatic_geometry,
        test_compactness_is_one_sided,
        test_energy_is_finite_and_deterministic,
        test_energy_rejects_wrong_length,
        test_hbond_rewards_helix,
        # objective quality
        test_coupled_terms_outweigh_the_separable_one,
        test_calibration_is_sequence_set_derived,
        test_energy_correlates_with_native_proximity,
        test_low_energy_structures_are_enriched,
        # CVaR
        test_cvar_preserves_multiplicities,
        test_cvar_sampled_converges_to_exact,
        test_cvar_alpha_one_is_mean,
        # VQE
        test_vqe_is_full_system,
        test_vqe_does_not_enumerate,
        test_final_answer_comes_from_final_circuit,
        test_reproducibility,
        test_rng_streams_are_independent,
        # optimizer
        test_spsa_costs_two_evaluations_per_iteration,
        test_spsa_uses_common_random_numbers,
        test_spsa_schedule_is_driven_by_progress,
        test_optimizer_guard_covers_spsa,
        test_default_optimizer_suits_a_stochastic_objective,
        test_one_layer_direct_sampler_is_exact,
        test_vqe_exposes_sampler_batch_and_trace_hooks,
        # budget
        test_budget_is_enforced,
        test_cem_respects_the_shared_budget,
        test_arms_are_cost_matched,
        test_restarts_survive_shared_budget,
        test_optimizer_budget_guard_rejects_simplex_only,
        test_annealing_actually_cools,
        # leakage
        test_no_pdb_access_during_optimization,
        test_hamiltonian_has_no_native_input,
        # metrics
        test_rmsd_is_angstroms,
        test_no_normalized_rmsd_path,
        test_ensemble_rmsd_is_min_over_models,
        test_evaluate_structure_rejects_length_mismatch,
        test_well_determined_mask_is_data_derived,
        test_identity_is_alignment_based,
        test_dataset_rejects_unrebuildable_native,
        # search correctness
        test_vqe_matches_exhaustive_on_tiny_system,
        test_ceiling_is_a_real_ceiling,
        test_representation_ceiling_computable,
        test_ceiling_cache_keys_on_residue_classes,
        # sidechains
        test_sidechain_bond_lengths,
        test_sidechain_rings_planar,
        test_sidechain_no_internal_clashes,
        test_sidechain_chirality,
        test_sidechain_rejects_unimplemented_residue,
        test_sidechain_atom_counts,
        # amber
        test_amber_energy_is_deterministic,
        test_amber_native_below_helix,
        test_amber_backbone_restraints_hold,
        test_amber_rejects_wrong_length,
        test_amber_no_pdb_access_during_energy,
    ]

    results = []
    skipped = 0
    for t in tests:
        if not _HAVE_AMBER and t.__name__.startswith("test_amber_"):
            print(f"  [{_SKIP}] {t.__name__} -- openmm not installed")
            skipped += 1
            continue
        try:
            results.append(bool(t()))
        except Exception as exc:
            print(f"  [{_FAIL}] {t.__name__} raised {type(exc).__name__}: {exc}")
            results.append(False)
    print("-" * 72)
    print(f"  {sum(results)}/{len(results)} passed"
          + (f", {skipped} skipped (no openmm)" if skipped else "")
          + f". ALL PASSED: {all(results)}")
    print()
    return all(results)
