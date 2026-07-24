
import csv
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

import protein_geometry as geo
import representations as reps
import energy_terms as et
import hamiltonian as ham
import vqe as vqe_mod
import classical_baselines as cb
import evaluation as ev
import dataset as ds


BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")

CSV_FIELDS = [
    "protein", "sequence", "length", "method", "representation", "n_states",
    "n_qubits", "config_space", "seed", "layers", "alpha", "shots", "maxiter",
    "restarts", "optimizer",
    "vqe_energy", "vqe_modal_energy", "best_seen_energy", "native_energy",
    "energy_gap", "ca_rmsd_angstrom", "backbone_rmsd_angstrom",
    "ceiling_ca_rmsd", "contact_f1", "longrange_contact_recall",
    "ss_agreement", "ss_predicted", "ss_native",
    "n_energy_evaluations", "n_objective_evals", "runtime",
    "distribution_top1_prob", "distribution_entropy_bits",
    "vqe_bitstring", "best_seen_bitstring",
]


def _ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def append_csv(path: str, row: Dict) -> None:
    _ensure_results_dir()
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    clean = {k: row.get(k, "") for k in CSV_FIELDS}
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(clean)


def default_vqe_config() -> Dict:
    return {
        "layers": 4,
        "alpha": 0.15,
        "shots": 2048,
        "maxiter": 300,
        "restarts": 4,
        "optimizer": "COBYLA",
        "final_shots": 8192,
        "init_scale": 0.25,
    }


def run_one(entry: ds.PeptideEntry, representation: str = "torsion",
            n_states: int = 4, seed: int = 0,
            vqe_config: Optional[Dict] = None,
            method: str = "vqe",
            sa_steps: Optional[int] = None,
            verbose: bool = True) -> Dict:
    """Optimize one peptide with one method and one seed, then evaluate.

    STRICT ORDERING: optimization runs first with the PDB log reset. The
    native structure is loaded ONLY afterwards, for evaluation. An assertion
    verifies zero PDB access during optimization.
    """
    cfg = dict(default_vqe_config() if vqe_config is None else vqe_config)
    seq = entry.sequence
    rep = reps.make_representation(representation, len(seq), n_states=n_states)
    H = ham.FoldingHamiltonian(seq, rep)

    if verbose:
        d = rep.describe()
        print(f"  {entry.pdb_id} [{seq}] len={len(seq)} "
              f"rep={representation} qubits={rep.n_qubits} "
              f"space={d['config_space']:.3g} method={method} seed={seed}")

    geo.reset_pdb_log()
    t0 = time.time()

    if method == "vqe":
        res = vqe_mod.run_global_cvar_vqe(H, seed=seed, verbose=False, **cfg)
        chosen = res["vqe_bitstring"]
        n_obj = res["n_objective_evals_total"]
    elif method == "sa":
        steps = sa_steps if sa_steps is not None else 20000
        res = cb.simulated_annealing(H, n_steps=steps, seed=seed)
        chosen = res["best_bitstring"]
        res["vqe_energy"] = res["best_energy"]
        res["vqe_modal_energy"] = float("nan")
        res["best_seen_energy"] = res["best_energy"]
        res["vqe_bitstring"] = chosen
        res["best_seen_bitstring"] = chosen
        res["distribution_top1_prob"] = float("nan")
        res["distribution_entropy_bits"] = float("nan")
        n_obj = steps
    elif method == "random":
        n = sa_steps if sa_steps is not None else 20000
        res = cb.random_search(H, n_samples=n, seed=seed)
        chosen = res["best_bitstring"]
        res["vqe_energy"] = res["best_energy"]
        res["vqe_modal_energy"] = float("nan")
        res["best_seen_energy"] = res["best_energy"]
        res["vqe_bitstring"] = chosen
        res["best_seen_bitstring"] = chosen
        res["distribution_top1_prob"] = float("nan")
        res["distribution_entropy_bits"] = float("nan")
        n_obj = n
    else:
        raise ValueError(f"unknown method {method!r}")

    opt_runtime = time.time() - t0
    assert len(geo.get_pdb_log()) == 0, \
        "LEAKAGE: a native PDB was read during optimization"

    native_seq, native_coords, native_phi, native_psi = ds.load_native(entry)
    metrics = ev.evaluate_structure(chosen, rep, H, native_seq, native_coords,
                                    native_phi, native_psi)
    ceiling = ev.representation_ceiling(rep, native_coords["CA"],
                                        native_phi, native_psi, seed=seed)
    ss_native = (geo.assign_secondary_structure(native_coords)
                 if not rep.is_lattice else "unavailable")

    d = rep.describe()
    row = {
        "protein": entry.pdb_id, "sequence": seq, "length": len(seq),
        "method": method, "representation": representation,
        "n_states": (n_states if representation == "torsion" else 4),
        "n_qubits": rep.n_qubits, "config_space": d["config_space"],
        "seed": seed,
        "layers": cfg.get("layers", ""), "alpha": cfg.get("alpha", ""),
        "shots": cfg.get("shots", ""), "maxiter": cfg.get("maxiter", ""),
        "restarts": cfg.get("restarts", ""), "optimizer": cfg.get("optimizer", ""),
        "vqe_energy": res.get("vqe_energy"),
        "vqe_modal_energy": res.get("vqe_modal_energy"),
        "best_seen_energy": res.get("best_seen_energy"),
        "native_energy": metrics["native_energy"],
        "energy_gap": metrics["energy_gap_pred_minus_native"],
        "ca_rmsd_angstrom": metrics["ca_rmsd_angstrom"],
        "backbone_rmsd_angstrom": metrics["backbone_rmsd_angstrom"],
        "ceiling_ca_rmsd": ceiling["ceiling_ca_rmsd"],
        "contact_f1": metrics["contact_f1"],
        "longrange_contact_recall": metrics["longrange_contact_recall"],
        "ss_agreement": metrics["ss_agreement"],
        "ss_predicted": metrics["ss_predicted"],
        "ss_native": ss_native,
        "n_energy_evaluations": res.get("n_energy_evaluations"),
        "n_objective_evals": n_obj,
        "runtime": opt_runtime,
        "distribution_top1_prob": res.get("distribution_top1_prob"),
        "distribution_entropy_bits": res.get("distribution_entropy_bits"),
        "vqe_bitstring": res.get("vqe_bitstring"),
        "best_seen_bitstring": res.get("best_seen_bitstring"),
    }

    if verbose:
        print(f"    CA-RMSD {row['ca_rmsd_angstrom']:.2f} A "
              f"(ceiling {row['ceiling_ca_rmsd']:.2f} A) | "
              f"E_pred {row['vqe_energy']:.2f} "
              f"E_nat {row['native_energy']:.2f} "
              f"gap {row['energy_gap']:+.2f} | {opt_runtime:.1f}s")

    return row


def experiment_main_comparison(entries: List[ds.PeptideEntry],
                               seeds: List[int],
                               vqe_config: Optional[Dict] = None,
                               csv_name: str = "main_comparison.csv") -> List[Dict]:

    path = os.path.join(_ensure_results_dir(), csv_name)
    cfg = dict(default_vqe_config() if vqe_config is None else vqe_config)
    rows = []

    arms = [
        ("A_lattice_vqe", "lattice", 4, "vqe"),
        ("B_torsion_vqe", "torsion", 4, "vqe"),
        ("C_torsion_sa", "torsion", 4, "sa"),
        ("D_torsion_random", "torsion", 4, "random"),
    ]

    print("=" * 72)
    print("EXPERIMENT 1: MAIN COMPARISON")
    print("=" * 72)

    for arm_name, rep_kind, n_states, method in arms:
        print(f"\n--- arm {arm_name} ---")
        for entry in entries:
            for seed in seeds:
                try:
                    row = run_one(entry, representation=rep_kind,
                                  n_states=n_states, seed=seed,
                                  vqe_config=cfg, method=method)
                    row["method"] = arm_name
                    append_csv(path, row)
                    rows.append(row)
                except MemoryError as exc:
                    print(f"    SKIPPED {entry.pdb_id} seed {seed}: {exc}")
                except Exception as exc:
                    print(f"    FAILED {entry.pdb_id} seed {seed}: "
                          f"{type(exc).__name__}: {exc}")

    _print_summary(rows)
    print(f"\n  results written to {path}")
    return rows


def _print_summary(rows: List[Dict]) -> None:
    if not rows:
        print("  (no rows)")
        return
    print()
    print("  SUMMARY: mean +/- std CA-RMSD (A) across seeds")
    print(f"  {'arm':<20} {'protein':<8} {'RMSD':>14} {'ceiling':>9} {'gap':>9}")
    print("  " + "-" * 64)
    arms = sorted({r["method"] for r in rows})
    proteins = sorted({r["protein"] for r in rows})
    for arm in arms:
        arm_rows = [r for r in rows if r["method"] == arm]
        for p in proteins:
            pr = [r for r in arm_rows if r["protein"] == p]
            if not pr:
                continue
            s = ev.summarize_seeds(pr, "ca_rmsd_angstrom")
            c = np.nanmean([r["ceiling_ca_rmsd"] for r in pr])
            g = ev.summarize_seeds(pr, "energy_gap")
            print(f"  {arm:<20} {p:<8} "
                  f"{s['mean']:>6.2f} +/- {s['std']:<5.2f} "
                  f"{c:>9.2f} {g['mean']:>9.2f}")
        overall = ev.summarize_seeds(arm_rows, "ca_rmsd_angstrom")
        print(f"  {arm:<20} {'ALL':<8} {overall['mean']:>6.2f} "
              f"+/- {overall['std']:<5.2f}")
        print()


def experiment_energy_ablation(entries: List[ds.PeptideEntry],
                               seeds: List[int],
                               vqe_config: Optional[Dict] = None,
                               csv_name: str = "energy_ablation.csv") -> List[Dict]:

    path = os.path.join(_ensure_results_dir(), csv_name)
    cfg = dict(default_vqe_config() if vqe_config is None else vqe_config)
    rows = []

    print("=" * 72)
    print("EXPERIMENT 2: ENERGY-TERM ABLATION")
    print("=" * 72)

    variants: List[tuple] = [("full", dict(et.DEFAULT_WEIGHTS), True)]
    for term in et.TERM_NAMES:
        w = dict(et.DEFAULT_WEIGHTS)
        w[term] = 0.0
        variants.append((f"no_{term}", w, True))
    variants.append(("raw_mj", dict(et.DEFAULT_WEIGHTS), False))

    for name, weights, corrected in variants:
        print(f"\n--- variant {name} ---")
        for entry in entries:
            for seed in seeds:
                seq = entry.sequence
                rep = reps.TorsionStateRepresentation(len(seq), n_states=4)
                H = ham.FoldingHamiltonian(seq, rep, weights=weights,
                                           use_corrected_mj=corrected)
                geo.reset_pdb_log()
                t0 = time.time()
                try:
                    res = vqe_mod.run_global_cvar_vqe(H, seed=seed, **cfg)
                except MemoryError as exc:
                    print(f"    SKIPPED {entry.pdb_id}: {exc}")
                    continue
                rt = time.time() - t0
                assert len(geo.get_pdb_log()) == 0, "LEAKAGE during ablation"

                nseq, ncoords, nphi, npsi = ds.load_native(entry)
                m = ev.evaluate_structure(res["vqe_bitstring"], rep, H,
                                          nseq, ncoords, nphi, npsi)
                ceil_ = ev.representation_ceiling(rep, ncoords["CA"], nphi, npsi)
                row = {
                    "protein": entry.pdb_id, "sequence": seq,
                    "length": len(seq), "method": name,
                    "representation": "torsion", "n_states": 4,
                    "n_qubits": rep.n_qubits,
                    "config_space": 4.0 ** len(seq), "seed": seed,
                    "layers": cfg["layers"], "alpha": cfg["alpha"],
                    "shots": cfg["shots"], "maxiter": cfg["maxiter"],
                    "restarts": cfg["restarts"], "optimizer": cfg["optimizer"],
                    "vqe_energy": res["vqe_energy"],
                    "vqe_modal_energy": res["vqe_modal_energy"],
                    "best_seen_energy": res["best_seen_energy"],
                    "native_energy": m["native_energy"],
                    "energy_gap": m["energy_gap_pred_minus_native"],
                    "ca_rmsd_angstrom": m["ca_rmsd_angstrom"],
                    "backbone_rmsd_angstrom": m["backbone_rmsd_angstrom"],
                    "ceiling_ca_rmsd": ceil_["ceiling_ca_rmsd"],
                    "contact_f1": m["contact_f1"],
                    "longrange_contact_recall": m["longrange_contact_recall"],
                    "ss_agreement": m["ss_agreement"],
                    "ss_predicted": m["ss_predicted"],
                    "ss_native": geo.assign_secondary_structure(ncoords),
                    "n_energy_evaluations": res["n_energy_evaluations"],
                    "n_objective_evals": res["n_objective_evals_total"],
                    "runtime": rt,
                    "distribution_top1_prob": res["distribution_top1_prob"],
                    "distribution_entropy_bits": res["distribution_entropy_bits"],
                    "vqe_bitstring": res["vqe_bitstring"],
                    "best_seen_bitstring": res["best_seen_bitstring"],
                }
                append_csv(path, row)
                rows.append(row)
                print(f"    {entry.pdb_id} seed {seed}: "
                      f"RMSD {row['ca_rmsd_angstrom']:.2f} A")

    _print_summary(rows)
    print(f"\n  results written to {path}")
    return rows


def experiment_vqe_hyperparameters(entry: ds.PeptideEntry, seeds: List[int],
                                   csv_name: str = "vqe_hparams.csv") -> List[Dict]:
 
    path = os.path.join(_ensure_results_dir(), csv_name)
    rows = []
    print("=" * 72)
    print(f"EXPERIMENT 3: VQE HYPERPARAMETERS on {entry.pdb_id}")
    print("=" * 72)

    grid = []
    for alpha in (0.05, 0.15, 0.35, 1.0):
        grid.append(dict(default_vqe_config(), alpha=alpha))
    for layers in (1, 2, 6, 8):
        grid.append(dict(default_vqe_config(), layers=layers))
    grid.append(dict(default_vqe_config(), optimizer="SPSA", maxiter=150))

    for cfg in grid:
        tag = (f"a{cfg['alpha']}_L{cfg['layers']}_{cfg['optimizer']}")
        print(f"\n--- {tag} ---")
        for seed in seeds:
            try:
                row = run_one(entry, representation="torsion", n_states=4,
                              seed=seed, vqe_config=cfg, method="vqe")
                row["method"] = tag
                append_csv(path, row)
                rows.append(row)
            except Exception as exc:
                print(f"    FAILED seed {seed}: {type(exc).__name__}: {exc}")

    _print_summary(rows)
    print(f"\n  results written to {path}")
    return rows


# ==========================================================================
def experiment_scaling_report(max_length: int = 22,
                              json_name: str = "scaling.json") -> Dict:
    """Report qubit / memory / config-space scaling. No optimization run."""
    print("=" * 72)
    print("EXPERIMENT 4: SCALING REPORT (analytic)")
    print("=" * 72)
    print(f"  {'N':>3} {'lat.q':>6} {'t4.q':>6} {'t8.q':>6} "
          f"{'t4 space':>12} {'t4 SV mem':>12} {'simulable':>10}")
    print("  " + "-" * 62)
    table = []
    for n in range(8, max_length + 1):
        lat_q = 2 * (n - 1)
        t4_q = 2 * n
        t8_q = 3 * n
        space = 4.0 ** n
        mem_bytes = (2 ** t4_q) * 16
        simulable = t4_q <= 30
        mem_str = (f"{mem_bytes / 1e9:.1f} GB" if mem_bytes >= 1e9
                   else f"{mem_bytes / 1e6:.0f} MB")
        print(f"  {n:>3} {lat_q:>6} {t4_q:>6} {t8_q:>6} "
              f"{space:>12.3g} {mem_str:>12} {str(simulable):>10}")
        table.append({"n_residues": n, "lattice_qubits": lat_q,
                      "torsion4_qubits": t4_q, "torsion8_qubits": t8_q,
                      "torsion4_config_space": space,
                      "torsion4_statevector_bytes": mem_bytes,
                      "simulable_at_30_qubit_limit": simulable})
    out = {"table": table, "statevector_limit_qubits": 30,
           "max_simulable_length_torsion4": 15,
           "max_simulable_length_torsion8": 10,
           "max_simulable_length_lattice": 16}
    path = os.path.join(_ensure_results_dir(), json_name)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n  written to {path}")
    print("\n  HONEST LIMIT: a genuine full-system VQE with the 4-state")
    print("  torsion representation is simulable to N ~ 15 residues on a")
    print("  large-memory machine and N ~ 12 on a laptop. N = 20 requires")
    print("  40 qubits (~17 TB) and is NOT simulable. There is no way to")
    print("  reach N = 20 globally; block decomposition would reach it only")
    print("  by abandoning the global-VQE requirement.")
    return out


def experiment_prior_ablation(entries: List[ds.PeptideEntry], seeds: List[int],
                              vqe_config: Optional[Dict] = None,
                              csv_name: str = "prior_ablation.csv") -> List[Dict]:

    import priors

    path = os.path.join(_ensure_results_dir(), csv_name)
    cfg = dict(default_vqe_config() if vqe_config is None else vqe_config)
    rows = []
    prior_model = priors.SequencePrior(prefer_esm=True)
    print("=" * 72)
    print("EXPERIMENT 5: SEQUENCE-PRIOR ABLATION (non-default)")
    print(f"  prior mode: {prior_model.info()['mode']} "
          f"-- {prior_model.info()['description']}")
    print("=" * 72)

    class PriorHamiltonian(ham.FoldingHamiltonian):
        def __init__(self, sequence, representation, contact_prob,
                     prior_weight=2.0, **kw):
            super().__init__(sequence, representation, **kw)
            self.contact_prob = contact_prob
            self.prior_weight = float(prior_weight)

        def energy(self, bitstring):
            hit = self._cache.get(bitstring)
            if hit is not None:
                return hit
            base_comp = self.components(bitstring)
            e = (et.total_from_components(base_comp, self.weights)
                 + base_comp["backtracking"])
            cb_ = self.rep.build_coords(bitstring).get("CB")
            e += self.prior_weight * priors.contact_violation_energy(
                self.contact_prob, np.asarray(cb_, dtype=float))
            if len(self._cache) < self._cache_limit:
                self._cache[bitstring] = e
            self.n_energy_evaluations += 1
            return e

    for entry in entries:
        cp = prior_model.contact_probabilities(entry.sequence)
        for seed in seeds:
            rep = reps.TorsionStateRepresentation(len(entry.sequence), 4)
            H = PriorHamiltonian(entry.sequence, rep, cp)
            geo.reset_pdb_log()
            t0 = time.time()
            res = vqe_mod.run_global_cvar_vqe(H, seed=seed, **cfg)
            rt = time.time() - t0
            assert len(geo.get_pdb_log()) == 0, "LEAKAGE during prior ablation"
            nseq, ncoords, nphi, npsi = ds.load_native(entry)
            m = ev.evaluate_structure(res["vqe_bitstring"], rep, H,
                                      nseq, ncoords, nphi, npsi)
            ceil_ = ev.representation_ceiling(rep, ncoords["CA"], nphi, npsi)
            row = {
                "protein": entry.pdb_id, "sequence": entry.sequence,
                "length": len(entry.sequence),
                "method": f"prior_{prior_model.info()['mode']}",
                "representation": "torsion", "n_states": 4,
                "n_qubits": rep.n_qubits,
                "config_space": 4.0 ** len(entry.sequence), "seed": seed,
                "layers": cfg["layers"], "alpha": cfg["alpha"],
                "shots": cfg["shots"], "maxiter": cfg["maxiter"],
                "restarts": cfg["restarts"], "optimizer": cfg["optimizer"],
                "vqe_energy": res["vqe_energy"],
                "vqe_modal_energy": res["vqe_modal_energy"],
                "best_seen_energy": res["best_seen_energy"],
                "native_energy": m["native_energy"],
                "energy_gap": m["energy_gap_pred_minus_native"],
                "ca_rmsd_angstrom": m["ca_rmsd_angstrom"],
                "backbone_rmsd_angstrom": m["backbone_rmsd_angstrom"],
                "ceiling_ca_rmsd": ceil_["ceiling_ca_rmsd"],
                "contact_f1": m["contact_f1"],
                "longrange_contact_recall": m["longrange_contact_recall"],
                "ss_agreement": m["ss_agreement"],
                "ss_predicted": m["ss_predicted"],
                "ss_native": geo.assign_secondary_structure(ncoords),
                "n_energy_evaluations": res["n_energy_evaluations"],
                "n_objective_evals": res["n_objective_evals_total"],
                "runtime": rt,
                "distribution_top1_prob": res["distribution_top1_prob"],
                "distribution_entropy_bits": res["distribution_entropy_bits"],
                "vqe_bitstring": res["vqe_bitstring"],
                "best_seen_bitstring": res["best_seen_bitstring"],
            }
            append_csv(path, row)
            rows.append(row)
            print(f"    {entry.pdb_id} seed {seed}: "
                  f"RMSD {row['ca_rmsd_angstrom']:.2f} A")

    _print_summary(rows)
    print(f"\n  results written to {path}")
    print("  NOTE: these numbers are prior-assisted and must be reported")
    print("  separately from the default (no-prior) results.")
    return rows


def save_prediction(entry: ds.PeptideEntry, bitstring: str, rep,
                    filename: Optional[str] = None) -> str:
    coords = rep.build_coords(bitstring)
    if "N" not in coords:
        raise ValueError("cannot write a PDB for a CA-only lattice structure")
    path = os.path.join(_ensure_results_dir(),
                        filename or f"{entry.pdb_id}_predicted.pdb")
    geo.write_pdb(path, entry.sequence, coords,
                  remark=f"global CVaR-VQE prediction for {entry.pdb_id}")
    return path