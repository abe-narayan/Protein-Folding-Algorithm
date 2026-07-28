import csv
import json
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

# Pin BLAS/OMP thread pools to 1 BEFORE numpy is imported. Two reasons, both required
# for the worker pool below to be result-identical to the serial path:
#   1. Determinism. Multi-threaded BLAS reductions associate in a nondeterministic
#      order, so a dot product or SVD can differ in the last ulp between runs. That is
#      normally invisible, but `evaluation.representation_ceiling` and the lattice
#      `native_bitstring` make 20,000 sequential accept/reject decisions on `cs < cur_s`,
#      and a single flipped comparison sends the whole annealing trajectory elsewhere.
#   2. Oversubscription. N worker processes each spawning a full thread pool thrashes.
# The arrays here are tiny (n_residues ~ 10-20), so nothing was gaining from threading.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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
    "protein", "sequence", "length", "method", "representation", "energy_model","n_states",
    "n_qubits", "config_space", "seed", "layers", "alpha", "shots", "maxiter",
    "restarts", "optimizer",
    "vqe_energy", "vqe_modal_energy", "best_seen_energy", "native_energy",
    "energy_gap", "ca_rmsd_angstrom", "backbone_rmsd_angstrom",
    "ceiling_ca_rmsd", "contact_f1", "longrange_contact_recall",
    "ss_agreement", "ss_predicted", "ss_native",
    "n_energy_evaluations", "n_objective_evals",
    "eval_budget", "budget_exhausted", "terminated_by", "restarts_completed",
    "n_parameters", "maxiter_requested", "maxiter_resolved",
    "maxiter_over_n_params", "runtime",
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


def resolve_workers(n_workers: Optional[int] = None) -> int:
    """How many processes to fan the experiment loop across.

    ``None`` reads ``PFA_WORKERS`` and otherwise stays serial, so the default behaviour
    is unchanged. ``0`` means "one per core, less two".
    """
    if n_workers is None:
        n_workers = int(os.environ.get("PFA_WORKERS", "1"))
    if n_workers == 0:
        n_workers = max(1, (os.cpu_count() or 2) - 2)
    return max(1, int(n_workers))


def _run_one_task(task: Dict):
    """Pool worker: one ``run_one`` call. Top-level so it is picklable under spawn.

    Returns ``(row, None)`` or ``(None, error_string)`` -- exceptions are returned rather
    than raised so one bad cell cannot take down the sweep, matching the serial path's
    per-iteration try/except.
    """
    label = task.pop("_label")
    try:
        row = run_one(**task)
        row["method"] = label
        return row, None
    except MemoryError as exc:
        return None, f"SKIPPED {task['entry'].pdb_id} seed {task['seed']}: {exc}"
    except Exception as exc:
        return None, (f"FAILED {task['entry'].pdb_id} seed {task['seed']}: "
                      f"{type(exc).__name__}: {exc}")


def _dispatch(tasks: List[Dict], n_workers: int, worker=_run_one_task):
    """Run ``tasks`` serially or across a process pool, always yielding in task order.

    Order matters: the parent appends to the CSV in the order results arrive, so
    preserving it keeps the output file identical to the serial run rather than merely
    equivalent. ``Pool.imap`` is ordered, so this holds regardless of completion order.
    """
    if n_workers <= 1:
        return [worker(dict(t)) for t in tasks]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        return list(pool.imap(worker, [dict(t) for t in tasks]))


#: Unique energy evaluations every arm is allowed. This is the currency the comparison
#: is denominated in: under the amber model one evaluation is one OpenMM
#: backbone-restrained minimization, and it dominates wall-clock for every arm.
DEFAULT_EVAL_BUDGET = 20_000


def default_vqe_config() -> Dict:
    return {
        "layers": 4,
        "alpha": 0.15,
        "shots": 2048,
        "maxiter": None,        # resolves to 50 x n_params; the budget is what binds
        "restarts": 4,
        "optimizer": "COBYLA",
        "final_shots": 8192,
        "init_scale": 0.25,
    }


def run_one(entry: ds.PeptideEntry, representation: str = "torsion",
            n_states: int = 4, seed: int = 0,
            vqe_config: Optional[Dict] = None,
            method: str = "vqe",
            energy_model: str = "legacy",
            sa_steps: Optional[int] = None,
            eval_budget: Optional[int] = DEFAULT_EVAL_BUDGET,
            verbose: bool = True) -> Dict:

    if energy_model not in ("legacy", "amber"):
        raise ValueError(f"unknown energy_model {energy_model!r}")
    cfg = dict(default_vqe_config() if vqe_config is None else vqe_config)
    seq = entry.sequence
    rep = reps.make_representation(representation, len(seq), n_states=n_states)
    # The SAME budget goes to every arm through the Hamiltonian each one shares.
    if energy_model == "amber":
        # Imported lazily: it pulls in openmm, which is an optional dependency. At
        # module scope it made every legacy-model run fail at import on a machine
        # without OpenMM.
        import amber_hamiltonian as amber
        H = amber.AmberHamiltonian(seq, rep, eval_budget=eval_budget)
    else:
        H = ham.FoldingHamiltonian(seq, rep, eval_budget=eval_budget)

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
        # Enough steps that the shared BUDGET is what stops it, not the step count.
        # Annealing revisits heavily -- 20,000 SA steps spent only ~11-14k unique
        # evaluations while 20,000 random draws spent ~19.8k, so a matched step count
        # is a ~1.7x cost mismatch. Revisits are cache hits, so the headroom is cheap.
        steps = sa_steps if sa_steps is not None else (
            20 * eval_budget if eval_budget else 20000)
        res = cb.simulated_annealing(H, n_steps=steps, seed=seed)
        chosen = res["best_bitstring"]
        res["vqe_energy"] = res["best_energy"]
        res["vqe_modal_energy"] = float("nan")
        res["best_seen_energy"] = res["best_energy"]
        res["vqe_bitstring"] = chosen
        res["best_seen_bitstring"] = chosen
        res["distribution_top1_prob"] = float("nan")
        res["distribution_entropy_bits"] = float("nan")
        n_obj = res["n_objective_evals"]
    elif method == "random":
        n = sa_steps if sa_steps is not None else (
            2 * eval_budget if eval_budget else 20000)
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

    # Everything below is post-hoc analysis -- metrics, native comparisons, energy
    # breakdowns. It must not be billed to the search budget, and must not be able to
    # fail once the budget is spent. The spent count is preserved for reporting.
    spent = H.n_energy_evaluations
    H.end_search()

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
        "method": method, "representation": representation, "energy_model": energy_model,
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
        "n_energy_evaluations": spent,
        "n_objective_evals": n_obj,
        "eval_budget": eval_budget,
        "budget_exhausted": (eval_budget is not None and spent >= eval_budget),
        "terminated_by": res.get("terminated_by", ""),
        "restarts_completed": res.get("restarts_completed", ""),
        "n_parameters": res.get("n_parameters", ""),
        "maxiter_requested": cfg.get("maxiter"),
        "maxiter_resolved": res.get("maxiter_resolved", ""),
        "maxiter_over_n_params": res.get("maxiter_over_n_params", ""),
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
                               energy_model: str = "legacy",
                               csv_name: str = "main_comparison.csv",
                               n_workers: Optional[int] = None,
                               eval_budget: Optional[int] = DEFAULT_EVAL_BUDGET) -> List[Dict]:

    path = os.path.join(_ensure_results_dir(), csv_name)
    cfg = dict(default_vqe_config() if vqe_config is None else vqe_config)
    if energy_model not in ("legacy", "amber"):
        raise ValueError(f"unknown energy_model {energy_model!r}")
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

    # Every (arm, entry, seed) cell is independent: its own Hamiltonian, its own RNG
    # stream seeded from `seed`, and no state shared with any other cell. Building the
    # full task list first lets them run concurrently without changing any one of them.
    tasks = []
    for arm_name, rep_kind, n_states, method in arms:
        if energy_model == "amber" and rep_kind == "lattice":
            print(f"\n--- arm {arm_name} SKIPPED "
                  f"(amber energy model has no lattice topology) ---")
            continue
        for entry in entries:
            for seed in seeds:
                tasks.append(dict(_label=arm_name, entry=entry,
                                  representation=rep_kind, n_states=n_states,
                                  seed=seed, vqe_config=cfg, method=method,
                                  energy_model=energy_model,
                                  eval_budget=eval_budget))

    n_workers = resolve_workers(n_workers)
    print(f"\n  {len(tasks)} runs over {n_workers} worker(s), "
          f"shared budget {eval_budget} unique energy evaluations per arm")
    for row, err in _dispatch(tasks, n_workers):
        if err is not None:
            print(f"    {err}")
            continue
        append_csv(path, row)
        rows.append(row)

    _print_summary(rows)
    _check_cost_matched(rows)
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


def _check_cost_matched(rows: List[Dict]) -> bool:
    """Refuse to present arms that were not given the same allowance.

    The soundness condition is an equal *allowance*, not equal *spend*: an arm that
    converges and stops early has gained no advantage. So unequal or absent
    ``eval_budget`` is an error, while spend spread is reported as information --
    under-spending arms terminated before exhausting the budget, which is itself a
    finding about optimizer choice.
    """
    if not rows:
        return True
    print("  COST MATCHING")
    budgets = {r.get("eval_budget") for r in rows}
    if None in budgets or "" in budgets:
        print("    !! at least one arm ran with NO shared budget -- cost is configured")
        print("       per-arm and the comparison is UNSOUND.")
        return False
    if len(budgets) > 1:
        print(f"    !! arms were given DIFFERENT budgets {sorted(budgets)} -- "
              "the comparison is UNSOUND.")
        return False
    budget = budgets.pop()
    spend = {r["method"]: int(r["n_energy_evaluations"]) for r in rows}
    lo, hi = min(spend.values()), max(spend.values())
    print(f"    equal allowance: {budget} unique energy evaluations to every arm  [OK]")
    print(f"    actual spend   : {lo}-{hi}"
          + (f" ({hi / max(1, lo):.2f}x spread)" if lo else ""))
    under = {m: v for m, v in spend.items() if v < 0.95 * budget}
    if under:
        print("    under-spending arms terminated before exhausting the budget: "
              + ", ".join(f"{m} ({v})" for m, v in sorted(under.items())))
    return True


def experiment_energy_ablation(entries: List[ds.PeptideEntry],
                               seeds: List[int],
                               vqe_config: Optional[Dict] = None,
                               csv_name: str = "energy_ablation.csv",
                               n_workers: Optional[int] = None,
                               eval_budget: Optional[int] = DEFAULT_EVAL_BUDGET) -> List[Dict]:

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

    tasks = [dict(_label=name, weights=weights, corrected=corrected,
                  entry=entry, seed=seed, cfg=cfg, eval_budget=eval_budget)
             for name, weights, corrected in variants
             for entry in entries for seed in seeds]

    n_workers = resolve_workers(n_workers)
    print(f"\n  {len(tasks)} runs over {n_workers} worker(s)")
    for row, err in _dispatch(tasks, n_workers, worker=_run_ablation_task):
        if err is not None:
            print(f"    {err}")
            continue
        append_csv(path, row)
        rows.append(row)
        print(f"    {row['protein']} {row['method']} seed {row['seed']}: "
              f"RMSD {row['ca_rmsd_angstrom']:.2f} A")

    _print_summary(rows)
    print(f"\n  results written to {path}")
    return rows


def _run_ablation_task(task: Dict):
    """Pool worker for the energy-term ablation. Top-level so it is picklable.

    Body is the serial loop's body verbatim -- same Hamiltonian construction, same
    `seed` into the VQE, same metric calls -- so a cell computes what it always did.
    """
    name = task["_label"]
    weights, corrected = task["weights"], task["corrected"]
    entry, seed, cfg = task["entry"], task["seed"], task["cfg"]
    try:
        seq = entry.sequence
        rep = reps.TorsionStateRepresentation(len(seq), n_states=4)
        H = ham.FoldingHamiltonian(seq, rep, weights=weights,
                                   use_corrected_mj=corrected,
                                   eval_budget=task["eval_budget"])
        geo.reset_pdb_log()
        t0 = time.time()
        res = vqe_mod.run_global_cvar_vqe(H, seed=seed, **cfg)
        rt = time.time() - t0
        assert len(geo.get_pdb_log()) == 0, "LEAKAGE during ablation"
        spent = H.n_energy_evaluations
        H.end_search()

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
                    "n_energy_evaluations": spent,
                    "n_objective_evals": res["n_objective_evals_total"],
                    "eval_budget": task["eval_budget"],
                    "terminated_by": res.get("terminated_by", ""),
                    "restarts_completed": res.get("restarts_completed", ""),
                    "runtime": rt,
                    "distribution_top1_prob": res["distribution_top1_prob"],
                    "distribution_entropy_bits": res["distribution_entropy_bits"],
                    "vqe_bitstring": res["vqe_bitstring"],
                    "best_seen_bitstring": res["best_seen_bitstring"],
        }
        return row, None
    except MemoryError as exc:
        return None, f"SKIPPED {entry.pdb_id} {name}: {exc}"
    except Exception as exc:
        return None, (f"FAILED {entry.pdb_id} {name} seed {seed}: "
                      f"{type(exc).__name__}: {exc}")


def experiment_vqe_hyperparameters(entry: ds.PeptideEntry, seeds: List[int],
                                   csv_name: str = "vqe_hparams.csv",
                                   n_workers: Optional[int] = None,
                                   eval_budget: Optional[int] = DEFAULT_EVAL_BUDGET) -> List[Dict]:

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

    tasks = []
    for cfg in grid:
        tag = f"a{cfg['alpha']}_L{cfg['layers']}_{cfg['optimizer']}"
        for seed in seeds:
            tasks.append(dict(_label=tag, entry=entry, representation="torsion",
                              n_states=4, seed=seed, vqe_config=cfg, method="vqe",
                              eval_budget=eval_budget))

    n_workers = resolve_workers(n_workers)
    print(f"\n  {len(tasks)} runs over {n_workers} worker(s), "
          f"shared budget {eval_budget} unique energy evaluations per arm")
    for row, err in _dispatch(tasks, n_workers):
        if err is not None:
            print(f"    {err}")
            continue
        append_csv(path, row)
        rows.append(row)

    _print_summary(rows)
    _check_cost_matched(rows)
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