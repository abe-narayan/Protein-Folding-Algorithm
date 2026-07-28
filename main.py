import argparse
import math
import os
import sys
from typing import List

import numpy as np
import protein_geometry as geo
import representations as reps
import energy_terms as et
import hamiltonian as ham
import vqe as vqe_mod
import dataset as ds
import experiments as exp
import validation as val


def _parse_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_strs(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip() != ""]


def _print_prediction_summary(res) -> None:
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


def _print_legacy_energy_breakdown(seq, rep, H, res) -> None:
    """Weighted 7-term breakdown for the VQE answer vs helix/extended references."""
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


def _print_legacy_native_breakdown(seq, rep, H) -> None:
    """For a sequence with a known native PDB, print the native weighted breakdown,
    per-residue Ramachandran penalties, and contact-pair table (legacy model only)."""
    pdb_id = {"GYDPETGTWG": "1UAO"}.get(seq)
    if not (pdb_id and not rep.is_lattice):
        return
    pdb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdbs",
                       f"{pdb_id}.pdb")
    if not os.path.exists(pdb):
        return
    try:
        nseq, ncoords, nphi, npsi = geo.native_coords_from_pdb(pdb)
        if len(nseq) != len(seq):
            print(f"\n  [native breakdown skipped: {pdb_id} has "
                  f"{len(nseq)} residues, sequence has {len(seq)}]")
            return

        ncomp = et.energy_components(seq, ncoords, nphi, npsi)
        print()
        print(f"native ({pdb_id}) weighted breakdown, for comparison:")
        print(f"  {'term':<16} {'weight':>7} {'native':>12}")
        for term in et.TERM_NAMES:
            w = H.weights.get(term, 0.0)
            print(f"  {term:<16} {w:>7.2f} {w * ncomp[term]:>12.3f}")
        print(f"  {'TOTAL':<16} {'':>7} "
              f"{et.total_from_components(ncomp, H.weights):>12.3f}")

        print()
        print("  native per-residue torsion penalty:")
        print(f"  {'i':>3} {'aa':>3} {'phi':>8} {'psi':>8} {'penalty':>9}")
        for i in range(len(seq)):
            phi_deg = math.degrees(nphi[i])
            psi_deg = math.degrees(npsi[i])
            pen = et.rama_penalty(seq[i], nphi[i], npsi[i])
            print(f"  {i+1:>3} {seq[i]:>3} {phi_deg:>8.1f} "
                  f"{psi_deg:>8.1f} {pen:>9.3f}")

        nCB = np.asarray(ncoords["CB"], dtype=float)
        _, _, mj = et.sequence_arrays(seq, True)
        print()
        print("  native contact pairs (|i-j| >= 3), "
              "sorted by CB-CB distance:")
        print(f"  {'pair':>8} {'d_CB':>7} {'switch':>7} {'MJ':>7} {'product':>8}")
        pairs = []
        for a in range(len(seq)):
            for b in range(a + 3, len(seq)):
                dist = float(np.linalg.norm(nCB[a] - nCB[b]))
                s = float(et.switch(np.array([dist]), 4.5, 8.5)[0])
                m = float(mj[a, b])
                pairs.append((dist, a, b, s, m))
        pairs.sort()
        for dist, a, b, s, m in pairs:
            tag = f"{seq[a]}{a+1}-{seq[b]}{b+1}"
            print(f"  {tag:>8} {dist:>7.2f} {s:>7.3f} {m:>7.3f} {s * m:>8.3f}")
    except Exception as exc:
        print(f"\n  [native breakdown unavailable: "
              f"{type(exc).__name__}: {exc}]")


def cmd_predict(args) -> int:
    """Sequence-only prediction. No PDB is read at any point."""
    seq = args.sequence.strip().upper()
    rep = reps.make_representation(args.representation, len(seq),
                                   n_states=args.states)
    if args.energy_model == "amber":
        import amber_hamiltonian as amber   # lazy: openmm is optional
        H = amber.AmberHamiltonian(seq, rep)
    else:
        H = ham.FoldingHamiltonian(seq, rep)
    cfg = exp.default_vqe_config()
    cfg.update(layers=args.layers, alpha=args.alpha, shots=args.shots,
               maxiter=args.maxiter, restarts=args.restarts)

    d = rep.describe()
    print(f"sequence        : {seq}  (N = {len(seq)})")
    print(f"representation  : {d['name']} ({d['n_states']} states)")
    print(f"qubits          : {rep.n_qubits}")
    print(f"config space    : {d['config_space']:.4g}")
    print(f"statevector mem : {(2 ** rep.n_qubits) * 16 / 1e6:.1f} MB")
    print("")

    geo.reset_pdb_log()
    res = vqe_mod.run_global_cvar_vqe(H, seed=args.seed, verbose=True, **cfg)
    assert len(geo.get_pdb_log()) == 0, "LEAKAGE: PDB read during prediction"

    _print_prediction_summary(res)

    print()
    if args.energy_model == "legacy":
        _print_legacy_energy_breakdown(seq, rep, H, res)
        _print_legacy_native_breakdown(seq, rep, H)
    else:
        print("energy breakdown : skipped "
              "(amber model has no 7-term weighted decomposition)")

    if not rep.is_lattice:
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
    exp.experiment_main_comparison(entries, args.seeds, vqe_config=cfg,
                                   energy_model=args.energy_model,
                                   n_workers=args.workers)
    return 0

def cmd_energy_ablation(args) -> int:
    entries = ds.build_dataset(pdb_ids=args.proteins)
    if not entries:
        return 1
    cfg = exp.default_vqe_config()
    cfg.update(layers=args.layers, alpha=args.alpha, shots=args.shots,
               maxiter=args.maxiter, restarts=args.restarts)
    exp.experiment_energy_ablation(entries, args.seeds, vqe_config=cfg,
                                   n_workers=args.workers)
    return 0


def cmd_hparams(args) -> int:
    entries = ds.build_dataset(pdb_ids=[args.protein])
    if not entries:
        return 1
    exp.experiment_vqe_hyperparameters(entries[0], args.seeds,
                                       n_workers=args.workers)
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

    p.add_argument("--sequence", default="GYDPETGTWG")
    p.add_argument("--proteins", default="1UAO,5AWL",
                   help="comma-separated PDB IDs")
    p.add_argument("--protein", default="1UAO")
    p.add_argument("--seeds", default="0,1,2",
                   help="comma-separated integer seeds")

    p.add_argument("--representation", default="torsion",
                   choices=["torsion", "lattice"])
    p.add_argument("--energy-model", default="legacy",
                   choices=["legacy", "amber"])
    p.add_argument("--states", type=int, default=4, choices=[4, 8])
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.15)
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None,
                   help="processes to fan the experiment loop across "
                        "(default 1 = serial, or $PFA_WORKERS; 0 = cores - 2). "
                        "Each (arm, protein, seed) cell is independent, so this "
                        "changes wall-clock only, not results.")
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

    if not ran:
        parser.print_help()
        print("\nSuggested first run:\n  python main.py --validate --scaling")
    return rc


if __name__ == "__main__":
    sys.exit(main())