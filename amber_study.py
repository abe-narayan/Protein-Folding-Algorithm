"""Run the cost-matched AMBER study used by the poster figures.

This is intentionally separate from ``experiments.py``: the historical CSV contract and
legacy experiments remain untouched, while every file here is AMBER-only and belongs to
one immutable, manifested run directory.

Examples
--------
Fast development pilot::

    python amber_study.py --split development --seeds 0,1,2 --budget 5000

Locked held-out run::

    python amber_study.py --split test --seeds 100,101,102,103,104,105,106,107,108,109
"""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from typing import Dict, List, Optional


# Prevent worker/process oversubscription before importing numpy/OpenMM.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

import amber_hamiltonian as amber
import classical_baselines as cb
import dataset as ds
import energy_quality as eq
import evaluation as ev
import protein_geometry as geo
import representations as reps
import vqe as vqe_mod


BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
# Sampling/proposal attempts are not the matched cost: cached duplicates are free.  Keep
# the administrative cap far above the unique-energy budget so a concentrated CEM or SA
# run does not stop early merely because it revisits structures.
SEARCH_DRAW_MULTIPLIER = 200

# Four identity clusters. Hyperparameters are developed on clusters 0/1 and locked before
# clusters 2/3 are used for the poster accuracy result.
TARGET_SPLIT = {
    "1UAO": "development", "5AWL": "development",
    "1LE0": "development", "1LE1": "development",
    "1LE3": "test", "2EVQ": "test",
}

LOCKED_ALGORITHM_KEYS = (
    "states", "layers", "shots", "alpha", "restarts", "final_shots",
    "init_scale", "spsa_a", "spsa_c", "backend", "mps_threshold",
    "mps_max_bond", "mps_cutoff", "minim_workers", "checkpoint",
    "readout_reserve_frac",
    "sa_restarts", "sa_t_start", "sa_t_end", "cem_elite",
    "cem_learning_rate", "search_draw_multiplier", "restraint_k", "minimization_steps",
    "minimization_tolerance", "collapse_floor",
)

RUN_FIELDS = [
    "run_id", "config_id", "protein", "split", "cluster", "sequence", "length",
    "method", "energy_model", "backend", "n_states", "n_qubits", "config_space",
    "seed", "eval_budget", "n_energy_evaluations", "n_objective_evals",
    "setup_s", "search_s", "total_s", "parallel_energy_s", "minim_workers",
    "selected_energy", "best_seen_energy", "native_energy", "energy_gap",
    "ca_rmsd_angstrom", "ca_rmsd_ensemble_min", "ca_rmsd_ensemble_mean",
    "best_seen_ca_rmsd_ensemble_min", "backbone_rmsd_angstrom", "core_ca_rmsd",
    "ensemble_spread", "ceiling_ca_rmsd", "contact_f1",
    "longrange_contact_recall", "ss_agreement", "n_collapsed",
    "layers", "alpha", "shots", "restarts", "optimizer", "spsa_a", "spsa_c",
    "distribution_top1_prob", "distribution_entropy_bits", "terminated_by",
    "restarts_completed", "selected_bitstring", "best_seen_bitstring", "pdb_path",
]

TRACE_FIELDS = [
    "run_id", "config_id", "protein", "split", "method", "seed", "backend",
    "n_energy_evaluations", "elapsed_s", "method_step", "objective_eval", "restart",
    "best_energy", "best_bitstring", "cvar", "ca_rmsd_angstrom",
    "ca_rmsd_ensemble_min",
]

QUALITY_FIELDS = [
    "protein", "split", "sequence", "n_qubits", "n_structures", "exhaustive",
    "spearman", "enrichment", "mean_rmsd_all", "mean_rmsd_low_energy",
    "native_pct", "native_snapped_rmsd", "e_native_snapped", "e_global_min",
    "rmsd_at_global_min", "best_rmsd_anywhere", "e_at_best_rmsd",
]


def _csv_append(path: str, fields: List[str], row: Dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def _parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("numpy", "scipy", "pennylane", "pennylane-lightning",
                    "openmm", "quimb", "matplotlib"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "missing"
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE, text=True).strip()
    except Exception:
        return "unknown"


def _config_id(config: Dict) -> str:
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _prepare_output(path: Optional[str], config_id: str) -> str:
    if path is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(RESULTS, f"amber_study_{stamp}_{config_id}")
    path = os.path.abspath(path)
    if os.path.exists(path) and os.listdir(path):
        raise FileExistsError(
            f"{path} is not empty; AMBER studies never append to an existing run")
    os.makedirs(os.path.join(path, "structures"), exist_ok=True)
    return path


def _rmsd_for_bits(bitstring: Optional[str], rep, native_ca,
                   ensemble_ca) -> Dict[str, float]:
    if not bitstring:
        return {"ca_rmsd_angstrom": float("nan"),
                "ca_rmsd_ensemble_min": float("nan")}
    ca = np.asarray(rep.build_coords(bitstring)["CA"], dtype=float)
    return {
        "ca_rmsd_angstrom": float(geo.ca_rmsd(ca, native_ca)),
        "ca_rmsd_ensemble_min": float(geo.ca_rmsd_to_ensemble(ca, ensemble_ca)["min"]),
    }


def _backend(rep, requested: str, layers: int, mps_threshold: int) -> str:
    if requested != "auto":
        return requested
    if layers == 1:
        return "direct"
    return "mps" if rep.n_qubits >= mps_threshold else "statevector"


def _make_batch(H, sequence: str, args, method: str):
    if args.minim_workers <= 1 or method == "sa":
        return None
    from mps.parallel import AmberBuilder, BudgetedEnergyBatch
    builder = AmberBuilder(
        sequence, n_states=args.states,
        restraint_k=args.restraint_k,
        minimization_steps=args.minimization_steps,
        minimization_tolerance=args.minimization_tolerance,
        collapse_floor=args.collapse_floor)
    return BudgetedEnergyBatch(H, builder, args.minim_workers)


def _run_method(entry, method: str, seed: int, args, output: str,
                config_id: str) -> Dict:
    task_t0 = time.perf_counter()
    rep = reps.TorsionStateRepresentation(
        len(entry.sequence), n_states=args.states, sequence=entry.sequence)
    H = amber.AmberHamiltonian(
        entry.sequence, rep, restraint_k=args.restraint_k,
        minimization_steps=args.minimization_steps,
        minimization_tolerance=args.minimization_tolerance,
        collapse_floor=args.collapse_floor, eval_budget=args.budget)
    setup_s = time.perf_counter() - task_t0
    backend = _backend(rep, args.backend, args.layers, args.mps_threshold)
    # Include worker-context startup and teardown in measured search time. Excluding it
    # would make the parallel VQE/CEM paths look artificially faster on short runs.
    search_t0 = time.perf_counter()
    batch = _make_batch(H, entry.sequence, args, method)
    energy_batch = None if batch is None else batch.compute
    try:
        if method == "vqe":
            sampler = None
            if backend == "direct":
                from direct_sampler import OneLayerDirectSampler
                sampler = OneLayerDirectSampler(
                    rep.n_qubits, layers=args.layers, ring=True)
            elif backend == "mps":
                from mps.sampler import MPSSampler
                sampler = MPSSampler(rep.n_qubits, args.layers,
                                     max_bond=args.mps_max_bond,
                                     cutoff=args.mps_cutoff)
            result = vqe_mod.run_global_cvar_vqe(
                H, layers=args.layers, alpha=args.alpha, shots=args.shots,
                maxiter=None, restarts=args.restarts, seed=seed,
                optimizer="SPSA", final_shots=args.final_shots,
                init_scale=args.init_scale, sampler=sampler,
                energy_batch=energy_batch, spsa_a=args.spsa_a,
                spsa_c=args.spsa_c, trace_interval=args.checkpoint,
                readout_reserve_frac=args.readout_reserve_frac,
                verbose=args.verbose)
            selected = result["vqe_bitstring"] or result["best_seen_bitstring"]
            best_bits = result["best_seen_bitstring"]
            best_e = result["best_seen_energy"]
            selected_e = (result["vqe_energy"] if result["vqe_bitstring"]
                          else best_e)
            trace = result["objective_trace"]
            n_obj = result["n_objective_evals_total"]
        elif method == "sa":
            result = cb.simulated_annealing(
                H, n_steps=args.search_draw_multiplier * args.budget,
                t_start=args.sa_t_start,
                t_end=args.sa_t_end, seed=seed, restarts=args.sa_restarts,
                trace_interval=args.checkpoint)
            selected = best_bits = result["best_bitstring"]
            selected_e = best_e = result["best_energy"]
            trace, n_obj = result["trace"], result["n_objective_evals"]
        elif method == "cem":
            result = cb.cross_entropy_search(
                H, n_samples=args.search_draw_multiplier * args.budget,
                batch_size=args.shots,
                elite_frac=args.cem_elite, learning_rate=args.cem_learning_rate,
                seed=seed, trace_interval=args.checkpoint,
                energy_batch=energy_batch)
            selected = best_bits = result["best_bitstring"]
            selected_e = best_e = result["best_energy"]
            trace, n_obj = result["trace"], result["n_objective_evals"]
        elif method == "random":
            result = cb.random_search(
                H, n_samples=2 * args.budget, seed=seed,
                trace_interval=args.checkpoint, batch_size=args.shots,
                energy_batch=energy_batch)
            selected = best_bits = result["best_bitstring"]
            selected_e = best_e = result["best_energy"]
            trace, n_obj = result["trace"], result["n_objective_evals"]
        else:
            raise ValueError(f"unknown method {method!r}")
    finally:
        if batch is not None:
            batch.close()
    search_s = time.perf_counter() - search_t0
    spent = H.n_energy_evaluations
    parallel_energy_s = 0.0 if batch is None else batch.runtime
    H.end_search()

    native_seq, native_coords, native_phi, native_psi = ds.load_native(entry)
    ensemble = ds.load_ensemble(entry)
    ensemble_ca = [np.asarray(c["CA"], dtype=float) for _, c, _, _ in ensemble]
    metrics = ev.evaluate_structure(selected, rep, H, native_seq, native_coords,
                                    native_phi, native_psi, ensemble=ensemble)
    ceiling = ev.representation_ceiling(
        rep, native_coords["CA"], native_phi, native_psi, seed=seed)
    best_rmsd = _rmsd_for_bits(best_bits, rep, native_coords["CA"], ensemble_ca)

    run_key = f"{config_id}:{entry.pdb_id}:{method}:{seed}"
    run_id = hashlib.sha256(run_key.encode()).hexdigest()[:16]
    structure_name = f"{entry.pdb_id}_{method}_seed{seed}.pdb"
    structure_path = os.path.join(output, "structures", structure_name)
    geo.write_pdb(structure_path, entry.sequence, rep.build_coords(selected),
                  remark=(f"AMBER-study {method} encoded backbone; "
                          f"run {run_id}, seed {seed}"))

    row = {
        "run_id": run_id, "config_id": config_id, "protein": entry.pdb_id,
        "split": TARGET_SPLIT.get(entry.pdb_id, "custom"), "cluster": entry.cluster,
        "sequence": entry.sequence, "length": len(entry.sequence), "method": method,
        "energy_model": "amber", "backend": backend if method == "vqe" else "classical",
        "n_states": args.states, "n_qubits": rep.n_qubits,
        "config_space": rep.describe()["config_space"], "seed": seed,
        "eval_budget": args.budget, "n_energy_evaluations": spent,
        "n_objective_evals": n_obj, "setup_s": setup_s, "search_s": search_s,
        "total_s": time.perf_counter() - task_t0,
        "parallel_energy_s": parallel_energy_s,
        "minim_workers": args.minim_workers if batch is not None else 1,
        "selected_energy": selected_e, "best_seen_energy": best_e,
        "native_energy": metrics["native_energy"],
        "energy_gap": metrics["energy_gap_pred_minus_native"],
        "ca_rmsd_angstrom": metrics["ca_rmsd_angstrom"],
        "ca_rmsd_ensemble_min": metrics.get("ca_rmsd_ensemble_min"),
        "ca_rmsd_ensemble_mean": metrics.get("ca_rmsd_ensemble_mean"),
        "best_seen_ca_rmsd_ensemble_min": best_rmsd["ca_rmsd_ensemble_min"],
        "backbone_rmsd_angstrom": metrics["backbone_rmsd_angstrom"],
        "core_ca_rmsd": metrics.get("core_ca_rmsd"),
        "ensemble_spread": metrics.get("ensemble_spread"),
        "ceiling_ca_rmsd": ceiling["ceiling_ca_rmsd"],
        "contact_f1": metrics["contact_f1"],
        "longrange_contact_recall": metrics["longrange_contact_recall"],
        "ss_agreement": metrics["ss_agreement"],
        "n_collapsed": sum(1 for e in H._cache.values() if not np.isfinite(e)),
        "layers": args.layers if method == "vqe" else "",
        "alpha": args.alpha if method == "vqe" else "",
        "shots": args.shots if method in ("vqe", "cem") else "",
        "restarts": args.restarts if method == "vqe" else args.sa_restarts if method == "sa" else "",
        "optimizer": "SPSA" if method == "vqe" else "",
        "spsa_a": args.spsa_a if method == "vqe" else "",
        "spsa_c": args.spsa_c if method == "vqe" else "",
        "distribution_top1_prob": result.get("distribution_top1_prob"),
        "distribution_entropy_bits": result.get("distribution_entropy_bits"),
        "terminated_by": result.get("terminated_by"),
        "restarts_completed": result.get("restarts_completed"),
        "selected_bitstring": selected, "best_seen_bitstring": best_bits,
        "pdb_path": os.path.relpath(structure_path, output),
    }

    for point in trace:
        tr = dict(point)
        tr.update(run_id=run_id, config_id=config_id, protein=entry.pdb_id,
                  split=row["split"], method=method, seed=seed,
                  backend=row["backend"])
        if "elapsed_s" not in tr:
            tr["elapsed_s"] = ""
        rr = _rmsd_for_bits(tr.get("best_bitstring"), rep,
                            native_coords["CA"], ensemble_ca)
        tr.update(rr)
        _csv_append(os.path.join(output, "checkpoints.csv"), TRACE_FIELDS, tr)
    return row


def _quality(entries, args, output: str) -> None:
    if args.quality_samples <= 0:
        return
    for entry in entries:
        rep = reps.TorsionStateRepresentation(
            len(entry.sequence), n_states=args.states, sequence=entry.sequence)
        H = amber.AmberHamiltonian(
            entry.sequence, rep, restraint_k=args.restraint_k,
            minimization_steps=args.minimization_steps,
            minimization_tolerance=args.minimization_tolerance,
            collapse_floor=args.collapse_floor)
        _, coords, phi, psi = ds.load_native(entry)
        q = eq.energy_quality(H, rep, coords["CA"], phi, psi,
                              n_sample=args.quality_samples, seed=args.quality_seed)
        q.update(protein=entry.pdb_id, split=TARGET_SPLIT.get(entry.pdb_id, "custom"),
                 sequence=entry.sequence, n_qubits=rep.n_qubits)
        _csv_append(os.path.join(output, "quality.csv"), QUALITY_FIELDS, q)
        print(f"quality {entry.pdb_id}: Spearman {q['spearman']:+.3f}, "
              f"enrichment {q['enrichment']:+.2f} A", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AMBER-only, poster-ready search benchmark")
    p.add_argument("--split", choices=["development", "test", "all"],
                   default="development")
    p.add_argument("--proteins", default=None,
                   help="explicit comma-separated PDB IDs; overrides --split")
    p.add_argument("--methods", default="vqe,sa,cem,random")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--config-from", default=None,
                   help="development manifest whose algorithm settings are locked here")
    p.add_argument("--budget", type=int, default=20_000)
    p.add_argument("--states", type=int, choices=[4, 8], default=4)
    # Accuracy-first defaults: enough optimizer updates to learn under a fixed AMBER
    # budget, with an exact direct sampler for the one-layer circuit.
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--shots", type=int, default=8)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--final-shots", type=int, default=8192)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--spsa-a", type=float, default=0.25)
    p.add_argument("--spsa-c", type=float, default=0.15)
    p.add_argument("--backend", choices=["auto", "direct", "statevector", "mps"],
                   default="auto")
    p.add_argument("--mps-threshold", type=int, default=26)
    p.add_argument("--mps-max-bond", type=int, default=4096)
    p.add_argument("--mps-cutoff", type=float, default=1e-12)
    p.add_argument("--minim-workers", type=int, default=4,
                   help="parallel AMBER minimizers for batch methods; use 1 for serial")
    p.add_argument("--checkpoint", type=int, default=100)
    p.add_argument("--readout-reserve-frac", type=float, default=0.05,
                   help="fraction of VQE's total matched budget reserved for final "
                        "circuit readout (default 0.05)")
    p.add_argument("--sa-restarts", type=int, default=2)
    p.add_argument("--sa-t-start", type=float, default=4.0)
    p.add_argument("--sa-t-end", type=float, default=1e-3)
    p.add_argument("--cem-elite", type=float, default=0.25)
    p.add_argument("--cem-learning-rate", type=float, default=0.5)
    p.add_argument(
        "--search-draw-multiplier", type=int, default=SEARCH_DRAW_MULTIPLIER,
        help="maximum proposal attempts per unique-energy budget unit; cached "
             "duplicates are free and this cap should not terminate a headline run")
    p.add_argument("--quality-samples", type=int, default=1000)
    p.add_argument("--quality-seed", type=int, default=314159)
    p.add_argument("--restraint-k", type=float, default=100.0)
    p.add_argument("--minimization-steps", type=int, default=50)
    p.add_argument("--minimization-tolerance", type=float, default=2.0)
    p.add_argument("--collapse-floor", type=float, default=-600.0)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.search_draw_multiplier < 1:
        raise ValueError("--search-draw-multiplier must be at least 1")
    locked_from = None
    if args.config_from:
        with open(args.config_from, encoding="utf-8") as fh:
            locked_manifest = json.load(fh)
        locked = locked_manifest.get("config", {})
        source_splits = {
            TARGET_SPLIT.get(p, "custom") for p in locked.get("proteins", [])
        }
        if source_splits != {"development"}:
            raise ValueError(
                "--config-from must point to a development-only manifest; "
                f"found target splits {sorted(source_splits)}")
        missing = [k for k in LOCKED_ALGORITHM_KEYS if k not in locked]
        if missing:
            raise ValueError(f"locked manifest is missing algorithm settings: {missing}")
        for key in LOCKED_ALGORITHM_KEYS:
            setattr(args, key, locked[key])
        locked_from = locked_manifest.get("config_id", "unknown")
    methods = [m.lower() for m in _parse_csv(args.methods)]
    allowed = {"vqe", "sa", "cem", "random"}
    if not methods or not set(methods) <= allowed:
        raise ValueError(f"--methods must be from {sorted(allowed)}")
    seeds = [int(x) for x in _parse_csv(args.seeds)]
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.proteins:
        proteins = [x.upper() for x in _parse_csv(args.proteins)]
    elif args.split == "all":
        proteins = list(TARGET_SPLIT)
    else:
        proteins = [p for p, split in TARGET_SPLIT.items() if split == args.split]

    config = vars(args).copy()
    # Paths and console verbosity do not define the scientific configuration. Keeping
    # them out makes an identical rerun retain the same configuration fingerprint.
    config.pop("output_dir", None)
    config.pop("config_from", None)
    config.pop("verbose", None)
    config.update(proteins=proteins, methods=methods, seeds=seeds,
                  energy_model="amber")
    config_id = _config_id(config)
    output = _prepare_output(args.output_dir, config_id)
    manifest = {
        "config_id": config_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(), "versions": _versions(), "config": config,
        "algorithm_config_locked_from": locked_from,
        "claim_boundary": (
            "Accuracy is primary. Runtime is secondary and may be called an advantage "
            "only if measured under the recorded worker count. Simulated VQE is not "
            "evidence of quantum speedup."),
    }
    with open(os.path.join(output, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # Cluster against the complete study catalogue even when running only one split.
    # Otherwise both development and test runs would independently number their clusters
    # 0/1, obscuring the fact that the fixed catalogue contains four distinct clusters.
    catalogue = list(dict.fromkeys([*TARGET_SPLIT, *proteins]))
    all_entries = ds.build_dataset(pdb_ids=catalogue)
    entries = [e for e in all_entries if e.pdb_id in proteins]
    found = {e.pdb_id for e in entries}
    missing = sorted(set(proteins) - found)
    if missing:
        raise RuntimeError(f"requested targets were not usable: {missing}")
    entries.sort(key=lambda e: proteins.index(e.pdb_id))
    _quality(entries, args, output)

    failures = 0
    for entry in entries:
        for method in methods:
            for seed in seeds:
                print(f"{entry.pdb_id} {method} seed {seed}", flush=True)
                try:
                    row = _run_method(entry, method, seed, args, output, config_id)
                    _csv_append(os.path.join(output, "runs.csv"), RUN_FIELDS, row)
                    print(f"  E {row['selected_energy']:.2f} | "
                          f"ens-RMSD {row['ca_rmsd_ensemble_min']:.2f} A | "
                          f"{row['search_s']:.1f}s", flush=True)
                except Exception as exc:
                    failures += 1
                    _csv_append(os.path.join(output, "failures.csv"),
                                ["protein", "method", "seed", "error", "traceback"],
                                {"protein": entry.pdb_id, "method": method, "seed": seed,
                                 "error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()})
                    print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(f"results: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
