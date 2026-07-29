"""CLI entry point.

`amber_hamiltonian` (and therefore `openmm`) is imported lazily, inside the amber branch
only, so every legacy-model command runs without OpenMM installed.
"""
import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

# Pin BLAS/OMP thread pools BEFORE numpy is imported. `experiments.py` does the same, but
# it is imported *after* numpy here, so its pinning arrives too late to take effect in this
# process -- and `optimizations.md` records the pinning as mandatory for determinism, not an
# optimization: `evaluation.representation_ceiling` makes 20,000 sequential accept/reject
# decisions on `cs < cur_s`, and one comparison flipping on a last-ulp difference sends the
# whole annealing trajectory elsewhere.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import protein_geometry as geo
import representations as reps
import energy_terms as et
import hamiltonian as ham
import vqe as vqe_mod
import dataset as ds
import experiments as exp
import evaluation as ev
import energy_quality as eq
import validation as val


def _parse_ints(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_strs(s: str) -> List[str]:
    return [x.strip().upper() for x in s.split(",") if x.strip() != ""]


def _load_weights(path: Optional[str]) -> Optional[Dict[str, float]]:
    """Load weights from a `weight_calibration.json` (or a bare {term: weight} dict)."""
    if not path:
        return None
    with open(path) as fh:
        blob = json.load(fh)
    w = blob.get("weights", blob)
    missing = [t for t in et.TERM_NAMES if t not in w]
    if missing:
        raise ValueError(f"{path} is missing weights for {missing}")
    return {t: float(w[t]) for t in et.TERM_NAMES}


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
    """Weighted breakdown for the VQE answer vs helix/extended references.

    The helix and extended references use library index 0 and 1, which every
    per-residue-class library in `representations` reserves for the helical and extended
    state of its class -- so these remain meaningful for any composition.
    """
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


def _print_native_comparison(seq, rep, H) -> None:
    """Native breakdown and per-residue torsion table, if a matching PDB is cached.

    The PDB is resolved by looking for a cached structure whose deposited sequence equals
    `seq` (`dataset.resolve_pdb_for_sequence`). This replaces a hardcoded
    ``{"GYDPETGTWG": "1UAO"}`` lookup, which meant this diagnostic only ever fired for
    one specific peptide.
    """
    if rep.is_lattice:
        return
    pdb_id = ds.resolve_pdb_for_sequence(seq)
    if not pdb_id:
        print("\n  [no cached PDB matches this sequence; native comparison skipped]")
        return
    pdb = os.path.join(ds.PDB_DIR, f"{pdb_id}.pdb")
    try:
        ensemble = geo.native_ensemble_from_pdb(pdb)
        nseq, ncoords, nphi, npsi = ensemble[0]

        ncomp = et.energy_components(nseq, ncoords, nphi, npsi)
        print()
        print(f"native ({pdb_id}, {len(ensemble)} deposited model(s)) "
              f"weighted breakdown:")
        print(f"  {'term':<16} {'weight':>7} {'native':>12}")
        for term in et.TERM_NAMES:
            w = H.weights.get(term, 0.0)
            print(f"  {term:<16} {w:>7.2f} {w * ncomp[term]:>12.3f}")
        print(f"  {'TOTAL':<16} {'':>7} "
              f"{et.total_from_components(ncomp, H.weights):>12.3f}")

        models = [np.asarray(c["CA"]) for _, c, _, _ in ensemble]
        spread = geo.ensemble_spread(models)
        mask = ev.well_determined_mask(models)
        print()
        print(f"  ensemble spread (mean pairwise CA-RMSD) : {spread:.2f} A")
        print(f"  well-determined residues (RMSF < "
              f"{ev.CORE_RMSF_TOL} A)   : {int(mask.sum())}/{len(mask)}")
        print(f"  -> a prediction closer than {spread:.2f} A is not distinguishable "
              f"from the experiment")

        print()
        print("  native per-residue torsion penalty (model 1):")
        print(f"  {'i':>3} {'aa':>3} {'class':>8} {'phi':>8} {'psi':>8} {'penalty':>9}")
        classes = getattr(rep, "classes", ["GENERAL"] * len(nseq))
        for i in range(len(nseq)):
            pen = et.rama_penalty(nseq[i], nphi[i], npsi[i])
            print(f"  {i+1:>3} {nseq[i]:>3} {classes[i]:>8} "
                  f"{math.degrees(nphi[i]):>8.1f} "
                  f"{math.degrees(npsi[i]):>8.1f} {pen:>9.3f}")
    except Exception as exc:
        print(f"\n  [native comparison unavailable: {type(exc).__name__}: {exc}]")


def cmd_predict(args) -> int:
    """Sequence-only prediction. No PDB is read during the search."""
    seq = args.sequence.strip().upper()
    rep = reps.make_representation(args.representation, len(seq),
                                  n_states=args.states, sequence=seq)
    weights = _load_weights(args.weights_json)
    if args.energy_model == "amber":
        import amber_hamiltonian as amber   # lazy: openmm is optional
        H = amber.AmberHamiltonian(seq, rep, eval_budget=args.eval_budget)
    else:
        # --eval-budget now applies here too. It was previously ignored by --predict, so
        # the run was unbounded and every invocation emitted the no-shared-budget warning.
        H = ham.FoldingHamiltonian(seq, rep, weights=weights,
                                   eval_budget=args.eval_budget)
    cfg = exp.default_vqe_config()
    cfg.update(layers=args.layers, alpha=args.alpha, shots=args.shots,
               maxiter=args.maxiter, restarts=args.restarts,
               optimizer=args.optimizer)

    d = rep.describe()
    print(f"sequence        : {seq}  (N = {len(seq)})")
    print(f"representation  : {d['name']} ({d['n_states']} states)")
    if not rep.is_lattice:
        print(f"residue classes : {' '.join(c[:3] for c in d['residue_classes'])}")
    print(f"qubits          : {rep.n_qubits}")
    print(f"config space    : {d['config_space']:.4g}")
    print(f"statevector mem : {(2 ** rep.n_qubits) * 16 / 1e6:.1f} MB")
    print(f"eval budget     : {args.eval_budget}")
    print("")

    geo.reset_pdb_log()
    res = vqe_mod.run_global_cvar_vqe(H, seed=args.seed, verbose=True, **cfg)
    assert len(geo.get_pdb_log()) == 0, "LEAKAGE: PDB read during prediction"

    H.end_search()          # post-hoc analysis is not billed to the search budget
    _print_prediction_summary(res)

    print()
    if args.energy_model == "legacy":
        _print_legacy_energy_breakdown(seq, rep, H, res)
        _print_native_comparison(seq, rep, H)
    else:
        print("energy breakdown : skipped "
              "(amber model has no weighted term decomposition)")

    if not rep.is_lattice:
        out = os.path.join(exp._ensure_results_dir(), f"{seq}_prediction.pdb")
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
               maxiter=args.maxiter, restarts=args.restarts,
               optimizer=args.optimizer)
    exp.experiment_main_comparison(entries, args.seeds, vqe_config=cfg,
                                   energy_model=args.energy_model,
                                   n_workers=args.workers,
                                   eval_budget=args.eval_budget,
                                   weights=_load_weights(args.weights_json))
    return 0


def cmd_energy_ablation(args) -> int:
    entries = ds.build_dataset(pdb_ids=args.proteins)
    if not entries:
        return 1
    cfg = exp.default_vqe_config()
    cfg.update(layers=args.layers, alpha=args.alpha, shots=args.shots,
               maxiter=args.maxiter, restarts=args.restarts,
               optimizer=args.optimizer)
    exp.experiment_energy_ablation(entries, args.seeds, vqe_config=cfg,
                                   n_workers=args.workers,
                                   eval_budget=args.eval_budget)
    return 0


def cmd_hparams(args) -> int:
    entries = ds.build_dataset(pdb_ids=[args.protein])
    if not entries:
        return 1
    exp.experiment_vqe_hyperparameters(entries[0], args.seeds,
                                       n_workers=args.workers,
                                       eval_budget=args.eval_budget)
    return 0


def cmd_energy_quality(args) -> int:
    """Is the objective worth minimising? Enumerates when the space is small enough."""
    entries = ds.build_dataset(pdb_ids=args.proteins)
    if not entries:
        return 1
    weights = _load_weights(args.weights_json)
    rc = 0
    for entry in entries:
        seq = entry.sequence
        rep = reps.make_representation("torsion", len(seq), n_states=args.states,
                                       sequence=seq)
        H = ham.FoldingHamiltonian(seq, rep, weights=weights)
        _, ncoords, nphi, npsi = ds.load_native(entry)
        print("=" * 72)
        print(f"  {entry.pdb_id}  [{seq}]  N={len(seq)}  "
              f"{rep.n_bits} bits, {2 ** rep.n_bits:,} structures")
        print("=" * 72)
        q = eq.energy_quality(H, rep, ncoords["CA"], nphi, npsi,
                              n_sample=args.sample, seed=args.seed)
        eq.print_quality(q)
        if not (q["spearman"] > 0 and q["enrichment"] > 0):
            print("  !! this objective is not usable for structure prediction: "
                  "optimising it does not move toward the native fold.")
            rc = 1
        print()
    print("  term statistics over random FEASIBLE structures:")
    eq.print_terms(eq.term_statistics([e.sequence for e in entries],
                                      n_states=args.states, seed=args.seed),
                   weights=weights)
    return rc


def cmd_calibrate_weights(args) -> int:
    entries = ds.build_dataset(pdb_ids=args.proteins)
    if not entries:
        return 1
    exp.experiment_weight_calibration(entries, seed=args.seed,
                                      n_states=args.states,
                                      n_test_clusters=args.test_clusters,
                                      n_sample=args.sample)
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
    p.add_argument("--energy-quality", action="store_true",
                   help="measure whether the energy model correlates with "
                        "distance-to-native at all. Run this before trusting any RMSD.")
    p.add_argument("--calibrate-weights", action="store_true",
                   help="variance-balance the free energy weights on train clusters and "
                        "report quality on held-out clusters")

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
    p.add_argument("--maxiter", type=int, default=None,
                   help="optimizer allowance; default None resolves to 50 x n_params "
                        "so the shared --eval-budget is what terminates the search")
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--optimizer", default=None,
                   choices=["SPSA", "COBYLA"],
                   help="default SPSA. The CVaR objective is a stochastic estimate, and "
                        "COBYLA is a trust-region method that requires a reproducible "
                        "one -- see experiments.default_vqe_config. COBYLA is kept "
                        "selectable so the two can be compared; --hparams does that.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-budget", type=int, default=exp.DEFAULT_EVAL_BUDGET,
                   help="unique energy evaluations allowed to EACH arm (default "
                        "20000). This is what makes the arms cost-comparable; "
                        "0 disables it and makes any comparison unsound.")
    p.add_argument("--weights-json", default=None,
                   help="path to results/weight_calibration.json (or any {term: weight} "
                        "JSON) to use instead of energy_terms.DEFAULT_WEIGHTS")
    p.add_argument("--sample", type=int, default=None,
                   help="--energy-quality / --calibrate-weights: sample this many "
                        "structures instead of enumerating (default: enumerate when "
                        "n_bits <= 22)")
    p.add_argument("--test-clusters", type=int, default=3,
                   help="--calibrate-weights: identity clusters to hold out")
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
    if args.eval_budget is not None and args.eval_budget <= 0:
        args.eval_budget = None
    if args.optimizer is None:
        args.optimizer = exp.default_vqe_config()["optimizer"]

    ran = False
    rc = 0

    if args.validate:
        ran = True
        rc |= (0 if val.run_all() else 1)
    if args.scaling:
        ran = True
        exp.experiment_scaling_report()
    if args.energy_quality:
        ran = True
        rc |= cmd_energy_quality(args)
    if args.calibrate_weights:
        ran = True
        rc |= cmd_calibrate_weights(args)
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
        print("\nSuggested first run:\n  python main.py --validate --energy-quality")
    return rc


if __name__ == "__main__":
    sys.exit(main())
