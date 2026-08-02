"""Reproduce every measurement in `docs/hbond-channel-findings.md`.

The energy-model diagnosis left a question open: the encodable hairpin appeared to score
`hbond_longrange` and `coop_sheet` exactly 0.000, which would mean every free weight that
could favour a hairpin multiplies zero and calibration is powerless. That turned out to be
false, and this script exists to show where the false step was.

The short version: `coop_helix` alone decides the ordering. It is worth up to -10.0 while
the true helix/hairpin gap is 3.89, and it fires on 1.56% of structures. Lowering it to the
guardrail floor `base/3` takes RMSD enrichment over the full 4,194,304-structure enumeration
from -0.274 ("ANTI-correlated") to +2.613 ("helps"), moving the global minimum from a 5.27 A
helix to a 1.96 A hairpin.

Run:  python diagnose_hbond_channel.py                 # all sections
      python diagnose_hbond_channel.py --section cliff # one section

Sections:
  chi        chi1 is part of the bitstring -- the all-0 helix is NOT the ideal helix
  sweep      desolvation_cost 1.0 -> 0.0; the snapped native's channel never opens
  matcher    why: a contested acceptor, and what three matchers do about it
  gap        does softening close the helix/hairpin gap or widen it? -- the answer
             depends entirely on WHICH hairpin, which is the point
  cliff      cross-strand H-bond survival vs CA-RMSD
  rings      aromatic ring excluded volume as a monotonic anti-native potential
  amber      test_amber_native_below_helix passes only on the clashing rotamer
             (needs openmm; without it `--validate` prints 75/75 and never runs it)

READ THIS BEFORE USING THE SNAPPED NATIVE FOR ANYTHING. `rep.angles_to_bits(phi, psi)`
is a per-residue nearest-angle projection, NOT the best structure the encoding can
express. On chignolin it lands at 2.67 A with zero cross-strand H-bonds, while
enumeration finds `0000110000111001000111` at 1.96 A carrying three of them
(`hbond_longrange` -10.556 weighted) -- closer to the native AND 10.8 kcal/mol lower.
Treating the projection as the encodable optimum produced three wrong conclusions in
earlier analyses, including a sign error on whether a softer H-bond form helps. The
`gap` section reports both so the difference is visible rather than assumed.

Nothing here mutates the model. Alternative matchers are computed locally so the
shipped `hbond_terms` stays the single source of truth for the shipped numbers.
"""
import argparse
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

import energy_terms as et
import protein_geometry as geo
import representations as reps

W = et.DEFAULT_WEIGHTS
SEP = et.HB_LONGRANGE_SEP
PDB = "pdbs/1UAO.pdb"


# ----------------------------------------------------------------------------
# candidate pairs and the three matchers
# ----------------------------------------------------------------------------
def hbond_candidates(coords: Dict[str, np.ndarray], c: float
                     ) -> Tuple[List[Tuple[float, int, int]], int]:
    """Every donor/acceptor pair clearing the gate, post-offset, sorted by energy.

    Deliberately mirrors `energy_terms.hbond_terms` up to the point where the greedy
    match begins, so the alternatives below differ from the shipped code in exactly
    one respect: which admissible pairs they keep.
    """
    N, C, O = coords["N"], coords["C"], coords["O"]
    H = geo.amide_h_positions(N, C, O)
    n = len(N)
    valid = np.isfinite(H).all(axis=1)
    dON = np.linalg.norm(N[:, None, :] - O[None, :, :], axis=2)
    dCH = np.linalg.norm(C[None, :, :] - H[:, None, :], axis=2)
    dOH = np.linalg.norm(H[:, None, :] - O[None, :, :], axis=2)
    dCN = np.linalg.norm(N[:, None, :] - C[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = 0.084 * 332.0 * (1.0 / dON + 1.0 / dCH - 1.0 / dOH - 1.0 / dCN)
    E = np.maximum(E, et.HB_E_FLOOR)
    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    ok = valid[:, None] & (sep >= 2)
    ok &= (dON > et.HB_MIN_ON) & (dCH > 0.5) & (dOH > 0.5) & (dCN > 0.5)
    ok &= np.isfinite(E) & (E < -0.5)
    di, aj = np.where(ok)
    out = [(float(E[i, j]) + c, int(i), int(j)) for i, j in zip(di, aj)
           if float(E[i, j]) + c < 0.0]
    out.sort()
    return out, n


def match_pairs(coords: Dict[str, np.ndarray], c: float, mode: str
                ) -> List[Tuple[float, int, int]]:
    """`mode` in {'shipped', 'cap2', 'optimal-weighted'}.

    'shipped'          one bond per donor and per acceptor, greedy by raw energy.
    'cap2'             as shipped, but an acceptor carbonyl O may take two bonds --
                       it has two lone pairs, and bifurcated H-bonds are real.
    'optimal-weighted' maximum-weight matching under the objective `energy_components`
                       actually sums (local x1.0, long-range x3.0).
    """
    cands, n = hbond_candidates(coords, c)
    if mode in ("shipped", "cap2"):
        cap = 1 if mode == "shipped" else 2
        d_used = np.zeros(n, dtype=int)
        a_used = np.zeros(n, dtype=int)
        kept = []
        for e, i, j in cands:
            if d_used[i] >= 1 or a_used[j] >= cap:
                continue
            d_used[i] += 1
            a_used[j] += 1
            kept.append((e, i, j))
        return kept
    if mode != "optimal-weighted":
        raise ValueError(f"unknown matcher {mode!r}")
    from scipy.optimize import linear_sum_assignment
    if not cands:
        return []
    cost = np.zeros((n, n))
    for e, i, j in cands:
        w = W["hbond_longrange"] if abs(i - j) >= SEP else W["hbond_local"]
        cost[i, j] = min(cost[i, j], w * e)
    rows, cols = linear_sum_assignment(cost)
    return [(next(e for e, a, b in cands if a == i and b == j), int(i), int(j))
            for i, j in zip(rows, cols) if cost[i, j] < 0]


def total_energy(seq: str, coords, phi, psi, rings, c: float = 1.0,
                 mode: str = "shipped") -> Tuple[float, int, int]:
    """(weighted total, n_bonds, n_cross_strand) under one (c, matcher) setting."""
    comp = et.energy_components(seq, coords, phi, psi, rings=rings)
    pairs = match_pairs(coords, c, mode)
    pl = tuple((i, j) for _, i, j in pairs)
    comp["hbond_local"] = sum(e for e, i, j in pairs if abs(i - j) < SEP)
    comp["hbond_longrange"] = sum(e for e, i, j in pairs if abs(i - j) >= SEP)
    comp["coop_helix"] = et.coop_helix_term(pl)
    comp["coop_sheet"] = et.coop_sheet_term(pl)
    n_lr = sum(1 for i, j in pl if abs(i - j) >= SEP)
    return et.total_from_components(comp, W), len(pl), n_lr


# ----------------------------------------------------------------------------
def _load(n_states: int = 4):
    seq, ncoords, nphi, npsi = geo.native_coords_from_pdb(PDB)
    rep = reps.TorsionStateRepresentation(len(seq), n_states=n_states, sequence=seq)
    return seq, ncoords, nphi, npsi, rep


def _best_chi(rep, states: Sequence[int] = None, phi=None, psi=None):
    """Lowest-energy chi assignment for a fixed backbone. chi1 is part of the
    bitstring, so comparing structures at chi=0 compares clashing variants."""
    seq = rep.sequence
    best = None
    for k in range(1 << rep.n_chi_bits):
        cs = [(k >> b) & 1 for b in range(rep.n_chi_bits)]
        bits = (rep.bitstring_from_states(states, chi_states=cs) if states is not None
                else rep.angles_to_bits(phi, psi, chi_states=cs))
        ph, ps, co, rg = rep.build_all(bits)
        e = et.total_from_components(
            et.energy_components(seq, co, ph, ps, rings=rg), W)
        if best is None or e < best[0]:
            best = (e, bits, ph, ps, co, rg, cs)
    return best


def section_chi():
    print("\n== chi1 is part of the bitstring ==================================")
    seq, nc, nphi, npsi, rep = _load()
    print("  all-helix backbone; chi bits are NOT free parameters:")
    for k in range(1 << rep.n_chi_bits):
        cs = [(k >> b) & 1 for b in range(rep.n_chi_bits)]
        bits = rep.bitstring_from_states([0] * len(seq), chi_states=cs)
        ph, ps, co, rg = rep.build_all(bits)
        comp = et.energy_components(seq, co, ph, ps, rings=rg)
        print(f"    chi={tuple(cs)}  E {et.total_from_components(comp, W):9.3f}"
              f"   steric {W['steric']*comp['steric']:8.3f}")
    print("  -> bitstring_from_states([0]*n) zeroes chi and does NOT give the ideal helix")


def section_sweep():
    print("\n== desolvation_cost sweep =========================================")
    seq, nc, nphi, npsi, rep = _load()
    nat = np.asarray(nc["CA"], float)
    co_n = geo.build_backbone(nphi, npsi)
    phi_i = np.full(len(seq), math.radians(-57.0))
    psi_i = np.full(len(seq), math.radians(-47.0))
    co_i = geo.build_backbone(phi_i, psi_i)
    _, _, ph_h, ps_h, co_h, rg_h, _ = _best_chi(rep, states=[0] * len(seq))
    _, _, ph_s, ps_s, co_s, rg_s, _ = _best_chi(rep, phi=nphi, psi=npsi)
    print(f"  snapped native CA-RMSD {geo.ca_rmsd(co_s['CA'], nat):.3f} A ; "
          f"rebuild {geo.ca_rmsd(co_n['CA'], nat):.3f} A")
    print(f"  {'c':>5}{'ideal hel':>11}{'0.18A':>9}{'cont gap':>10}"
          f"{'enc hel':>10}{'snapped':>10}{'snap hb_lr':>12}")
    for c in (1.0, 0.75, 0.5, 0.25, 0.0):
        ei, _, _ = total_energy(seq, co_i, phi_i, psi_i, None, c)
        en, _, _ = total_energy(seq, co_n, nphi, npsi, None, c)
        eh, _, _ = total_energy(seq, co_h, ph_h, ps_h, rg_h, c)
        es, _, _ = total_energy(seq, co_s, ph_s, ps_s, rg_s, c)
        _, lr, _ = et.hbond_terms(co_s, desolvation_cost=c)
        print(f"  {c:>5}{ei:>11.3f}{en:>9.3f}{en-ei:>10.3f}"
              f"{eh:>10.3f}{es:>10.3f}{W['hbond_longrange']*lr:>12.3f}")
    print("  -> the last column never moves: the encodable hairpin's channel stays shut")


def section_matcher():
    print("\n== why: a contested acceptor ======================================")
    seq, nc, nphi, npsi, rep = _load()
    _, _, _, _, co_s, _, _ = _best_chi(rep, phi=nphi, psi=npsi)
    for c in (1.0, 0.5, 0.0):
        cands, _ = hbond_candidates(co_s, c)
        print(f"  c={c}: candidates clearing the gate, in matcher order")
        for e, i, j in cands:
            kind = "LONG-RANGE" if abs(i - j) >= SEP else "local"
            print(f"      e {e:7.3f}  donor N{i:<2} acceptor O{j:<2} "
                  f"|i-j|={abs(i-j)}  {kind}")
        for mode in ("shipped", "optimal-weighted", "cap2"):
            pr = match_pairs(co_s, c, mode)
            lr = sum(W["hbond_longrange"] * e for e, i, j in pr
                     if abs(i - j) >= SEP)
            print(f"      {mode:<18} keeps {tuple((i,j) for _,i,j in pr)}"
                  f"   hb_lr(w) {lr:7.3f}")


#: Lowest-energy hairpin-like structure found by exhaustive enumeration of all 4,194,304
#: chignolin bitstrings, at DEFAULT_WEIGHTS with `coop_helix` removed. 1.9632 A CA-RMSD with
#: three cross-strand H-bonds -- (7,2), (2,7), (1,8). It exists to make one point: the
#: snapped native is a projection, not the encodable optimum, and every question of the form
#: "can the encoding express X" must be asked of this structure rather than of that one.
BEST_ENCODABLE_HAIRPIN = "0000110000111001000111"


def section_gap():
    print("\n== does softening close the gap, or widen it? =====================")
    seq, nc, nphi, npsi, rep = _load()
    _, _, ph_h, ps_h, co_h, rg_h, _ = _best_chi(rep, states=[0] * len(seq))
    _, _, ph_s, ps_s, co_s, rg_s, _ = _best_chi(rep, phi=nphi, psi=npsi)
    ph_b, ps_b, co_b, rg_b = rep.build_all(BEST_ENCODABLE_HAIRPIN)
    co_n = geo.build_backbone(nphi, npsi)
    phi_i = np.full(len(seq), math.radians(-57.0))
    psi_i = np.full(len(seq), math.radians(-47.0))
    co_i = geo.build_backbone(phi_i, psi_i)
    nat = np.asarray(nc["CA"], float)
    print(f"  helix 5.270 A | snapped native {geo.ca_rmsd(co_s['CA'], nat):.3f} A "
          f"| best encodable hairpin {geo.ca_rmsd(co_b['CA'], nat):.3f} A")
    for title, hel, hp in (
        ("ENCODABLE, hairpin = BEST BY ENUMERATION (1.96 A) -- this one governs",
         (co_h, ph_h, ps_h, rg_h), (co_b, ph_b, ps_b, rg_b)),
        ("ENCODABLE, hairpin = SNAPPED NATIVE (2.67 A) -- the misleading one",
         (co_h, ph_h, ps_h, rg_h), (co_s, ph_s, ps_s, rg_s)),
        ("CONTINUOUS (what a perfect search would see)",
         (co_i, phi_i, psi_i, None), (co_n, nphi, npsi, None)),
    ):
        print(f"  {title}")
        print(f"    {'matcher':<18}{'c':>5}{'helix':>10}{'hairpin':>10}"
              f"{'GAP':>9}{'vs base':>9}{'bonds h/p':>11}")
        base = None
        for mode in ("shipped", "optimal-weighted", "cap2"):
            for c in (1.0, 0.5, 0.0):
                eh, nh, _ = total_energy(seq, *hel, c, mode)
                ep, np_, nlr = total_energy(seq, *hp, c, mode)
                gap = ep - eh
                base = gap if base is None else base
                print(f"    {mode:<18}{c:>5}{eh:>10.3f}{ep:>10.3f}"
                      f"{gap:>9.3f}{gap-base:>+9.3f}{f'{nh}/{np_}':>11}")
        print()
    print("  -> the sign depends on WHICH hairpin, and the bonds h/p column is why.")
    print("     The offset is refunded once per bond, so it pays in proportion to bonds")
    print("     held: a hairpin with 3 cross-strand bonds banks 3x3.0 against the helix's")
    print("     6x1.0 and gains; one with a single local bond banks 1.0 and loses.")
    print("     Break-even is two cross-strand bonds.")


def section_cliff(n_trials: int = 60):
    print("\n== the cliff: cross-strand survival vs CA-RMSD ====================")
    seq, nc, nphi, npsi, rep = _load()
    nat = np.asarray(nc["CA"], float)
    rng = np.random.default_rng(0)
    rows = []
    for sig in (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 27.5):
        for _ in range(1 if sig == 0 else n_trials):
            s = math.radians(sig)
            co = geo.build_backbone(nphi + rng.normal(0, s, len(seq)),
                                    npsi + rng.normal(0, s, len(seq)))
            _, lr, pr = et.hbond_terms(co)
            rows.append((sig, geo.ca_rmsd(co["CA"], nat),
                         sum(1 for i, j in pr if abs(i - j) >= SEP),
                         W["coop_sheet"] * et.coop_sheet_term(pr)))
    a = np.array(rows)
    print(f"  {'sigma':>7}{'RMSD':>9}{'#cross':>9}{'coop_sheet(w)':>16}")
    for s in np.unique(a[:, 0]):
        m = a[:, 0] == s
        print(f"  {s:>7.1f}{a[m,1].mean():>9.3f}{a[m,2].mean():>9.2f}"
              f"{a[m,3].mean():>16.3f}")
    print(f"\n  binned by CA-RMSD; 'frac>=2' is the break-even threshold:")
    print(f"  {'bin':>12}{'n':>6}{'mean #cross':>13}{'frac>=1':>10}{'frac>=2':>10}")
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.), (1., 1.5),
                   (1.5, 2.), (2., 3.), (3., 99)):
        m = (a[:, 1] >= lo) & (a[:, 1] < hi)
        if m.sum() < 3:
            continue
        print(f"  {f'{lo}-{hi}':>12}{int(m.sum()):>6}{a[m,2].mean():>13.2f}"
              f"{(a[m,2]>=1).mean():>10.2f}{(a[m,2]>=2).mean():>10.2f}")
    print("  -> 4-state representation ceiling is 0.97 A; at 0.75-1.0 A only ~6% of")
    print("     structures hold the two cross-strand bonds needed to break even")


def section_rings(n_per_sigma: int = 400):
    print("\n== aromatic ring excluded volume vs CA-RMSD =======================")
    seq, nc, nphi, npsi, _ = _load()
    nat = np.asarray(nc["CA"], float)
    for ns in (4, 8):
        rep = reps.TorsionStateRepresentation(len(seq), n_states=ns, sequence=seq)
        rng = np.random.default_rng(1)
        rows, seen = [], set()
        for sig in (0, 2, 5, 8, 12, 18, 25, 35, 50):
            for _ in range(n_per_sigma):
                s = math.radians(sig)
                b = rep.angles_to_bits(nphi + rng.normal(0, s, len(seq)),
                                       npsi + rng.normal(0, s, len(seq)))
                bb = b[:rep.n_backbone_bits]
                if bb in seen:
                    continue
                seen.add(bb)
                st = [int(bb[i*rep.bits_per_residue:(i+1)*rep.bits_per_residue], 2)
                      for i in range(len(seq))]
                best = None
                for k in range(1 << rep.n_chi_bits):
                    cs = [(k >> t) & 1 for t in range(rep.n_chi_bits)]
                    _, _, co, rg = rep.build_all(
                        rep.bitstring_from_states(st, chi_states=cs))
                    sr = W["steric"] * et.steric_term(co, seq, rings=rg)
                    sn = W["steric"] * et.steric_term(co, seq, rings=None)
                    if best is None or sr < best[0]:
                        best = (sr, sn, geo.ca_rmsd(co["CA"], nat))
                rows.append(best[::-1])
        a = np.array(rows)          # (rmsd, steric_no_rings, steric_rings)
        print(f"  --- {ns} states, {len(a)} distinct backbones ---")
        print(f"  {'bin':>12}{'n':>6}{'ring-only steric':>18}{'frac>5':>9}{'frac>20':>9}")
        for lo, hi in ((0, 1.), (1., 1.5), (1.5, 2.), (2., 2.5), (2.5, 3.5),
                       (3.5, 5.), (5., 99)):
            m = (a[:, 0] >= lo) & (a[:, 0] < hi)
            if m.sum() < 3:
                continue
            ro = a[m, 2] - a[m, 1]
            print(f"  {f'{lo}-{hi}':>12}{int(m.sum()):>6}{ro.mean():>18.3f}"
                  f"{(ro>5).mean():>9.2f}{(ro>20).mean():>9.2f}")
    print("  -> monotonic: the closer to native, the larger the penalty")


def section_weights():
    """The result that matters: coop_helix alone decides the ordering.

    Reported on the two structures rather than the full enumeration, so this runs in a
    second. The enumeration figures it summarises -- enrichment -0.274 -> +2.613, global
    minimum 5.27 A -> 1.96 A -- are in `docs/hbond-channel-findings.md`; regenerate them
    with `diagnose_energy_model.py --weights-json`.
    """
    print("\n== coop_helix alone inverts the ordering ==========================")
    seq, nc, nphi, npsi, rep = _load()
    nat = np.asarray(nc["CA"], float)
    _, _, ph_h, ps_h, co_h, rg_h, _ = _best_chi(rep, states=[0] * len(seq))
    ph_b, ps_b, co_b, rg_b = rep.build_all(BEST_ENCODABLE_HAIRPIN)
    print(f"  {'coop_helix':>12}{'helix 5.27A':>14}{'hairpin 1.96A':>15}"
          f"{'gap':>9}   winner")
    for w_ch in (2.0, 1.5, 1.0, 2.0 / 3.0, 0.0):
        w = dict(W)
        w["coop_helix"] = w_ch
        eh = et.total_from_components(
            et.energy_components(seq, co_h, ph_h, ps_h, rings=rg_h), w)
        ep = et.total_from_components(
            et.energy_components(seq, co_b, ph_b, ps_b, rings=rg_b), w)
        tag = "HELIX" if eh < ep else "hairpin"
        note = "  <- calibrator's guardrail floor (base/3)" if abs(w_ch - 2/3) < 1e-9 else ""
        print(f"  {w_ch:>12.3f}{eh:>14.3f}{ep:>15.3f}{ep-eh:>9.3f}   {tag}{note}")
    print("  -> the crossover sits between 1.5 and 1.0, well inside the range")
    print("     calibrate_weights may already move coop_helix through.")
    print(f"  (coop_helix is exactly 0.000 on 98.44% of all 4,194,304 structures --")
    print(f"   a narrow, deep reward switched on only by helical topology)")


def section_amber():
    """`test_amber_native_below_helix` passes only because its helix is the clashing rotamer.

    Needs OpenMM. Skipped with a message when it is absent -- which is itself the point:
    without OpenMM the suite prints `75/75 passed` and this gate never runs at all.
    """
    print("\n== test_amber_native_below_helix passes on the clashing rotamer ===")
    try:
        import amber_hamiltonian as amb
    except ImportError:
        print("  SKIPPED: openmm not installed. Note that `main.py --validate` reports")
        print("  `75/75 passed` in exactly this situation -- the 5 AMBER gates are")
        print("  dropped from BOTH sides of the fraction, not counted as failures.")
        return
    seq, nc, nphi, npsi, rep = _load()
    H = amb.AmberHamiltonian(seq, rep)
    e_nat = H.energy_from_coords(nc, nphi, npsi)
    print(f"  native (real coords)   {e_nat:10.2f} kcal/mol")
    print(f"  {'helix chi':>12}{'energy':>12}{'gap vs native':>16}   test says")
    energies = []
    for k in range(1 << rep.n_chi_bits):
        cs = [(k >> b) & 1 for b in range(rep.n_chi_bits)]
        e = H.energy(rep.bitstring_from_states([0] * len(seq), chi_states=cs))
        energies.append((e, tuple(cs)))
        used = "  <- bitstring_from_states([0]*n)" if k == 0 else ""
        print(f"  {str(tuple(cs)):>12}{e:>12.2f}{e_nat - e:>+16.2f}   "
              f"{'PASS' if e_nat < e else 'FAIL'}{used}")
    lo, hi = min(energies)[0], max(energies)[0]
    best = min(energies)
    print(f"\n  one backbone, {len(energies)} chi assignments, spread {hi - lo:.2f} kcal/mol "
          f"AFTER the restrained minimization")
    print(f"  best helix chi={best[1]} at {best[0]:.2f} -> gap {e_nat - best[0]:+.2f}")
    print("  -> the gate holds ONLY against the clashing rotamer. Built properly the")
    print("     helix wins and the assertion is false, so the AMBER arm is not the")
    print("     independent physics check on helix bias that its name implies.")
    print("  -> it also falsifies energy_from_coords's stated reason for not scanning")
    print("     rotamers ('the minimizer relaxes the sidechains anyway'): a Trp ring")
    print("     inside the backbone is behind a barrier, not a gradient.")


SECTIONS = {"chi": section_chi, "sweep": section_sweep, "matcher": section_matcher,
            "gap": section_gap, "weights": section_weights,
            "cliff": section_cliff, "rings": section_rings, "amber": section_amber}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", choices=sorted(SECTIONS), default=None)
    a = ap.parse_args()
    for name in ([a.section] if a.section else
                 ["chi", "sweep", "matcher", "gap", "weights", "cliff", "rings", "amber"]):
        SECTIONS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
