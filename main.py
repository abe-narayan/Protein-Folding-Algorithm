"""CLI for the global CVaR-VQE peptide structure-prediction pipeline.

  python main.py --validate
  python main.py --scaling
  python main.py --predict --sequence GYDPETGTWG
  python main.py --main-comparison --proteins 1UAO,5AWL --seeds 0,1,2
  python main.py --energy-ablation --proteins 1UAO --seeds 0,1,2
  python main.py --hparams --protein 1UAO --seeds 0,1,2
  python main.py --prior-ablation --proteins 1UAO --seeds 0,1,2

Defaults use NO pretrained model and NO learned weights.
"""
import argparse
import sys
from typing import List

import numpy as np
import numpy as _np
import protein_geometry as geo
import representations as reps
import energy_terms as et
import hamiltonian as ham
import vqe as vqe_mod
import dataset as ds
import evaluation as ev
import experiments as exp
import validation as val


def _parse_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_strs(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip() != ""]


def cmd_predict(args) -> int:
    """Sequence-only prediction. No PDB is read at any point."""
    seq = args.sequence.strip().upper()
    rep = reps.make_representation(args.representation, len(seq),
                                   n_states=args.states)
    H = ham.FoldingHamiltonian(seq, rep)
    cfg = exp.default_vqe_config()
    cfg.update(layers=args.layers, alpha=args.alpha, shots=args.shots,
               maxiter=args.maxiter, restarts=args.restarts)

    print(f"sequence        : {seq}  (N = {len(seq)})")
    d = rep.describe()
    print(f"representation  : {d['name']} ({d['n_states']} states)")
    print(f"qubits          : {rep.n_qubits}")
    print(f"config space    : {d['config_space']:.4g}")
    print(f"statevector mem : {(2 ** rep.n_qubits) * 16 / 1e6:.1f} MB")
    print()

    geo.reset_pdb_log()
    res = vqe_mod.run_global_cvar_vqe(H, seed=args.seed, verbose=True, **cfg)
    assert len(geo.get_pdb_log()) == 0, "LEAKAGE: PDB read during prediction"

    print()
    print(f"VQE solution    : {res['vqe_bitstring']}")
    print(f"  energy        : {res['vqe_energy']:.4f}")
    print(f"VQE modal state : {res['vqe_modal_bitstring']}")
    print(f"  energy        : {res['vqe_modal_energy']:.4f}")
    print(f"best seen       : {res['best_seen_bitstring']}")
    print(f"  energy        : {res['best_seen_energy']:.4f}")
    print(f"  (best-seen is an anytime result, NOT the VQE answer)")
    print()
    print(f"distribution top-1 prob : {res['distribution_top1_prob']:.4f}")
    print(f"distribution entropy    : {res['distribution_entropy_bits']:.2f} "
          f"/ {res['max_entropy_bits']:.0f} bits")
    print(f"energy evaluations      : {res['n_energy_evaluations']}")
    print(f"objective evaluations   : {res['n_objective_evals_total']}")
    print(f"runtime                 : {res['runtime']:.1f} s")

    # ---- per-term energy breakdown (diagnostic) ----
    print()
    print("energy breakdown (weighted contributions):")
    comparisons = [("vqe", res["vqe_bitstring"])]
    if not rep.is_lattice:
        comparisons.append(("helix", rep.bitstring_from_states([0] * len(seq))))
        comparisons.append(("extended", rep.bitstring_from_states([1] * len(seq))))
    breakdowns = {name: H.components(b) for name, b in comparisons}
    header = " ".join(f"{n:>12}" for n in breakdowns)
    print(f"  {'term':<16} {'weight':>7} {header}")
    for term in et.TERM_NAMES:
        w = H.weights.get(term, 0.0)
        vals = " ".join(f"{w * breakdowns[n][term]:>12.3f}" for n in breakdowns)
        print(f"  {term:<16} {w:>7.2f} {vals}")
        totals = " ".join(f"{H.energy(b):>12.3f}" for _, b in comparisons)
    print(f"  {'TOTAL':<16} {'':>7} {totals}")

    # ---- native-structure reference breakdown (DIAGNOSTIC ONLY) ----
    # This reads a PDB and therefore runs strictly AFTER optimization is
    # complete. It never influences the search; it exists so the energy of a
    # known-correct structure can be compared term-by-term against what the
    # optimizer produced.
    import os as _os
    _pdb_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "pdbs")
    _known = {"GYDPETGTWG": "1UAO"}
    _pdb_id = _known.get(seq)
    if _pdb_id and not rep.is_lattice:
        _pdb = _os.path.join(_pdb_dir, f"{_pdb_id}.pdb")
        if _os.path.exists(_pdb):
            try:
                _nseq, _ncoords, _nphi, _npsi = geo.native_coords_from_pdb(_pdb)
                if len(_nseq) == len(seq):
                    _ncomp = et.energy_components(seq, _ncoords, _nphi, _npsi)
                    print()
                    print(f"native ({_pdb_id}) weighted breakdown, for comparison:")
                    print(f"  {'term':<16} {'weight':>7} {'native':>12}")
                    for term in et.TERM_NAMES:
                        w = H.weights.get(term, 0.0)
                        print(f"  {term:<16} {w:>7.2f} "
                              f"{w * _ncomp[term]:>12.3f}")
                    print(f"  {'TOTAL':<16} {'':>7} "
                          f"{et.total_from_components(_ncomp, H.weights):>12.3f}")

                    # Per-residue torsion diagnostic: the native's torsion
                    # term is the largest single penalty in the table, which
                    # should not happen for a real crystal structure. This
                    # shows which residues are being penalized and what their
                    # actual backbone angles are.
                    import math as _math
                    print()
                    print("  native per-residue torsion penalty:")
                    print(f"  {'i':>3} {'aa':>3} {'phi':>8} {'psi':>8} "
                          f"{'penalty':>9}")
                    for _i in range(len(seq)):
                        _p = _math.degrees(_nphi[_i])
                        _q = _math.degrees(_npsi[_i])
                        _pen = et.rama_penalty(seq[_i], _nphi[_i], _npsi[_i])
                        print(f"  {_i+1:>3} {seq[_i]:>3} {_p:>8.1f} "
                              f"{_q:>8.1f} {_pen:>9.3f}")

                    # Contact-term diagnostic. The native's contact energy is
                    # POSITIVE (+0.201) on a structure whose stability comes
                    # partly from Tyr-Trp stacking, which should register as
                    # favourable. This lists every |i-j|>=3 pair with its
                    # CB-CB distance, the switching-function value, the
                    # corrected-MJ energy, and their product, so it is visible
                    # whether the 4.5-8.5 A window is missing real contacts or
                    # whether the potential simply has no signal here.
                    _nCB = _np.asarray(_ncoords["CB"], dtype=float)
                    _kd, _q_, _mj = et.sequence_arrays(seq, True)
                    print()
                    print("  native contact pairs (|i-j| >= 3), "
                          "sorted by CB-CB distance:")
                    print(f"  {'pair':>8} {'d_CB':>7} {'switch':>7} "
                          f"{'MJ':>7} {'product':>8}")
                    _pairs = []
                    for _a in range(len(seq)):
                        for _b in range(_a + 3, len(seq)):
                            _d = float(_np.linalg.norm(_nCB[_a] - _nCB[_b]))
                            _s = float(et.switch(_np.array([_d]), 4.5, 8.5)[0])
                            _m = float(_mj[_a, _b])
                            _pairs.append((_d, _a, _b, _s, _m))
                    _pairs.sort()
                    for _d, _a, _b, _s, _m in _pairs:
                        _tag = f"{seq[_a]}{_a+1}-{seq[_b]}{_b+1}"
                        print(f"  {_tag:>8} {_d:>7.2f} {_s:>7.3f} "
                              f"{_m:>7.3f} {_s * _m:>8.3f}")
                else:
                    print(f"\n  [native breakdown skipped: {_pdb_id} has "
                          f"{len(_nseq)} residues, sequence has {len(seq)}]")
            except Exception as _exc:
                print(f"\n  [native breakdown unavailable: "
                      f"{type(_exc).__name__}: {_exc}]")

    if not rep.is_lattice:
        import os
        out = os.path.join(exp._ensure_results_dir(), "prediction.pdb")
        geo.write_pdb(out, seq, rep.build_coords(res["vqe_bitstring"]),
                      remark="global CVaR-VQE sequence-only prediction")
        print(f"\nstructure written to {out}")
    return 0


def cmd_main_comparison(args) -> int:
    entries = ds.build_dataset(pdb_ids=args.proteins)
    if not entries:
        print("No usable peptides. Check network access to files.rcsb.org.")
        return 1
    cfg = exp.default_vqe_config()
    cfg.update(layers=args.layers, alpha=args.alpha, shots=args.shots,
               maxiter=args.maxiter, restarts=args.restarts)
    exp.experiment_main_comparison(entries, args.seeds, vqe_config=cfg)
    return 0


def cmd_energy_ablation(args) -> int:
    entries = ds.build_dataset(pdb_ids=args.proteins)
    if not entries:
        return 1
    cfg = exp.default_vqe_config()
    cfg.update(layers=args.layers, alpha=args.alpha, shots=args.shots,
               maxiter=args.maxiter, restarts=args.restarts)
    exp.experiment_energy_ablation(entries, args.seeds, vqe_config=cfg)
    return 0


def cmd_hparams(args) -> int:
    entries = ds.build_dataset(pdb_ids=[args.protein])
    if not entries:
        return 1
    exp.experiment_vqe_hyperparameters(entries[0], args.seeds)
    return 0


def cmd_prior_ablation(args) -> int:
    entries = ds.build_dataset(pdb_ids=args.proteins)
    if not entries:
        return 1
    cfg = exp.default_vqe_config()
    cfg.update(layers=args.layers, alpha=args.alpha, shots=args.shots,
               maxiter=args.maxiter, restarts=args.restarts)
    exp.experiment_prior_ablation(entries, args.seeds, vqe_config=cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Global full-system CVaR-VQE peptide structure prediction")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--scaling", action="store_true")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--main-comparison", action="store_true")
    p.add_argument("--energy-ablation", action="store_true")
    p.add_argument("--hparams", action="store_true")
    p.add_argument("--prior-ablation", action="store_true")

    p.add_argument("--sequence", default="GYDPETGTWG")
    p.add_argument("--proteins", default="1UAO,5AWL",
                   help="comma-separated PDB IDs")
    p.add_argument("--protein", default="1UAO")
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated integer seeds")

    p.add_argument("--representation", default="torsion",
                   choices=["torsion", "lattice"])
    p.add_argument("--states", type=int, default=4, choices=[4, 8])
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.seeds = _parse_ints(args.seeds)
    args.proteins = _parse_strs(args.proteins)
    args.protein = args.protein.strip().upper()

    ran = False
    rc = 0

    if args.validate:
        ran = True
        rc |= (0 if val.run_all() else 1)
    if args.scaling:
        ran = True
        exp.experiment_scaling_report()
    if args.predict:
        ran = True
        rc |= cmd_predict(args)
    if args.main_comparison:
        ran = True
        rc |= cmd_main_comparison(args)
    if args.energy_ablation:
        ran = True
        rc |= cmd_energy_ablation(args)
    if args.hparams:
        ran = True
        rc |= cmd_hparams(args)
    if args.prior_ablation:
        ran = True
        rc |= cmd_prior_ablation(args)

    if not ran:
        parser.print_help()
        print("\nSuggested first run:\n  python main.py --validate --scaling")
    return rc


if __name__ == "__main__":
    sys.exit(main())