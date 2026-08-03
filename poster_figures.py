"""Create defensible, poster-ready figures from an ``amber_study.py`` run.

The figures deliberately keep four different questions separate:

* structural accuracy: ensemble-minimum CA-RMSD and backbone RMSD;
* structural fidelity beyond RMSD: contacts and secondary structure;
* search efficiency: accuracy versus equal energy evaluations and wall time;
* objective validity: whether lower model energy actually tracks lower RMSD.

Every plot shows individual seeds.  With fewer than five seeds the summary bar is the
observed range; with five or more it is a deterministic bootstrap 95% confidence interval
for the median.  PDF and SVG are the preferred poster assets; a 600-DPI PNG is also saved.

Example
-------
    python poster_figures.py results/amber_study_... --structure-protein 1UAO
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import dataset as ds
import protein_geometry as geo


METHOD_ORDER = ("vqe", "sa", "cem", "random")
LABELS = {
    "vqe": "CVaR-VQE",
    "sa": "Simulated annealing",
    "cem": "Cross-entropy method",
    "random": "Random search",
}
SHORT_LABELS = {
    "vqe": "CVaR-VQE", "sa": "Sim. annealing",
    "cem": "CEM", "random": "Random",
}
# Okabe-Ito-derived, color-vision-deficiency-safe palette.
COLORS = {
    "vqe": "#0072B2",
    "sa": "#D55E00",
    "cem": "#009E73",
    "random": "#707070",
}
MARKERS = {"vqe": "o", "sa": "s", "cem": "D", "random": "X"}


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: Dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _finite(rows: Iterable[Dict[str, str]], key: str) -> np.ndarray:
    values = np.asarray([_number(row, key) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _methods(runs: Sequence[Dict[str, str]]) -> List[str]:
    present = {row.get("method") for row in runs}
    return [method for method in METHOD_ORDER if method in present]


def _validate(runs: Sequence[Dict[str, str]], manifest: Dict) -> None:
    if not runs:
        raise ValueError("runs.csv has no result rows")
    models = {row.get("energy_model") for row in runs}
    if models != {"amber"}:
        raise ValueError(f"expected one AMBER-only study, found {sorted(models)}")
    config_ids = {row.get("config_id") for row in runs}
    if config_ids != {manifest.get("config_id")}:
        raise ValueError("runs.csv and manifest.json do not describe one configuration")
    budgets = {_number(row, "eval_budget") for row in runs}
    if len(budgets) != 1:
        raise ValueError("methods do not share one energy-evaluation budget")
    run_ids = [row.get("run_id") for row in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("runs.csv contains duplicate run IDs")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, output: str, stem: str) -> None:
    os.makedirs(output, exist_ok=True)
    fig.savefig(os.path.join(output, stem + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output, stem + ".svg"), bbox_inches="tight")
    fig.savefig(
        os.path.join(output, stem + ".png"), dpi=600,
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def _summary_interval(values: Sequence[float], seed: int = 1937
                      ) -> Tuple[float, float, float, str]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan, np.nan, np.nan, "no data"
    median = float(np.median(clean))
    if clean.size < 5:
        return median, float(clean.min()), float(clean.max()), "range"
    rng = np.random.default_rng(seed)
    boot = np.median(
        rng.choice(clean, size=(5000, clean.size), replace=True), axis=1)
    low, high = np.quantile(boot, (0.025, 0.975))
    return median, float(low), float(high), "bootstrap 95% CI"


def _seed_strip(ax, runs: Sequence[Dict[str, str]], key: str,
                methods: Sequence[str], seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    interval_label = ""
    for x, method in enumerate(methods):
        values = _finite((row for row in runs if row["method"] == method), key)
        if not values.size:
            continue
        jitter = rng.uniform(-0.10, 0.10, values.size)
        ax.scatter(
            x + jitter, values, s=64, marker=MARKERS[method],
            color=COLORS[method], edgecolor="white", linewidth=0.8,
            alpha=0.88, zorder=3,
        )
        median, low, high, interval_label = _summary_interval(values, 1000 + x)
        ax.errorbar(
            x, median, yerr=[[median - low], [high - median]], fmt="_",
            markersize=23, markeredgewidth=3.4, capsize=5, color="#202020",
            zorder=4,
        )
        ax.annotate(
            f"{median:.2f}", (x, high), xytext=(0, 8),
            textcoords="offset points", ha="center", va="bottom", fontsize=10,
        )
    ax.set_xticks(range(len(methods)), [SHORT_LABELS[m] for m in methods],
                  rotation=18, ha="right")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.75)
    return interval_label


def figure_accuracy(runs: Sequence[Dict[str, str]], output: str) -> None:
    """Primary RMSD figure; never replaces RMSD with model energy."""
    for protein in sorted({row["protein"] for row in runs}):
        protein_runs = [row for row in runs if row["protein"] == protein]
        methods = _methods(protein_runs)
        fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.4))
        specs = (
            ("ca_rmsd_ensemble_min", "Ensemble-minimum CA-RMSD (Å) ↓",
             "Primary fold accuracy"),
            ("backbone_rmsd_angstrom", "Backbone RMSD (Å) ↓",
             "All-backbone agreement"),
            ("excess", "CA-RMSD above encoding ceiling (Å) ↓",
             "Search error after representation limit"),
        )
        augmented = []
        for row in protein_runs:
            copy = dict(row)
            copy["excess"] = str(
                _number(row, "ca_rmsd_ensemble_min")
                - _number(row, "ceiling_ca_rmsd"))
            augmented.append(copy)
        interval_label = ""
        for i, (key, ylabel, title) in enumerate(specs):
            use = augmented if key == "excess" else protein_runs
            interval_label = _seed_strip(axes[i], use, key, methods, seed=31 + i)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(title)
        spread = np.nanmedian(_finite(protein_runs, "ensemble_spread"))
        ceiling = np.nanmedian(_finite(protein_runs, "ceiling_ca_rmsd"))
        if np.isfinite(spread):
            axes[0].axhspan(0, spread, color="#CC79A7", alpha=0.12, zorder=0)
            axes[0].annotate(
                f"native ensemble spread: {spread:.2f} Å", (0.02, spread),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", fontsize=10, color="#6A3158")
        if np.isfinite(ceiling):
            axes[0].axhline(ceiling, color="#3B3B3B", lw=1.4, ls=(0, (4, 3)))
            axes[0].annotate(
                f"encoding ceiling: {ceiling:.2f} Å", (0.98, ceiling),
                xycoords=("axes fraction", "data"), xytext=(0, 5),
                textcoords="offset points", ha="right", fontsize=10)
            axes[2].axhline(0, color="#3B3B3B", lw=1.2, ls=(0, (4, 3)))
        n_seeds = min(sum(row["method"] == m for row in protein_runs)
                      for m in methods)
        fig.suptitle(
            f"Structural accuracy under an equal AMBER evaluation budget — {protein}",
            fontsize=20, fontweight="bold")
        fig.text(
            0.5, 0.005,
            f"Points are independent seeds (minimum n={n_seeds}/method); "
            f"black bar = median, whisker = {interval_label}.",
            ha="center", fontsize=11, color="#444444")
        fig.tight_layout(rect=(0, 0.045, 1, 0.93))
        _save(fig, output, f"poster_accuracy_{protein}")


def figure_fidelity(runs: Sequence[Dict[str, str]], output: str) -> None:
    """Complement RMSD with topology-sensitive and local-structure metrics."""
    specs = (
        ("contact_f1", "Contact F1 ↑", "Native contacts"),
        ("longrange_contact_recall", "Long-range contact recall ↑",
         "Tertiary topology"),
        ("ss_agreement", "Secondary-structure agreement ↑", "Local structure"),
    )
    for protein in sorted({row["protein"] for row in runs}):
        protein_runs = [row for row in runs if row["protein"] == protein]
        methods = _methods(protein_runs)
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), sharey=True)
        for i, (key, ylabel, title) in enumerate(specs):
            _seed_strip(axes[i], protein_runs, key, methods, seed=83 + i)
            axes[i].set_ylim(-0.04, 1.07)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(title)
        fig.suptitle(f"Structural fidelity beyond RMSD — {protein}",
                     fontsize=20, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        _save(fig, output, f"poster_structural_fidelity_{protein}")


def _group_traces(traces: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for point in traces:
        grouped[point["run_id"]].append(point)
    for points in grouped.values():
        points.sort(key=lambda row: _number(row, "n_energy_evaluations"))
    return grouped


def _step(points: Sequence[Dict[str, str]], x: float, x_key: str,
          y_key: str) -> float:
    eligible = [row for row in points if _number(row, x_key) <= x]
    return _number(eligible[-1], y_key) if eligible else float("nan")


def _band(curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = np.full(curves.shape[1], np.nan)
    low = np.full(curves.shape[1], np.nan)
    high = np.full(curves.shape[1], np.nan)
    for i in range(curves.shape[1]):
        column = curves[:, i]
        column = column[np.isfinite(column)]
        if column.size:
            median[i] = np.median(column)
            low[i], high[i] = np.quantile(column, (0.25, 0.75))
    return median, low, high


def _convergence_panel(ax, protein_runs: Sequence[Dict[str, str]], grouped,
                       methods: Sequence[str], x_key: str, xlabel: str) -> None:
    if x_key == "elapsed_s":
        max_x = max(_number(row, "search_s") for row in protein_runs)
    else:
        max_x = max(
            _number(point, x_key)
            for row in protein_runs for point in grouped.get(row["run_id"], [])
            if np.isfinite(_number(point, x_key)))
    grid = np.linspace(max_x / 45.0, max_x, 45)
    for method in methods:
        curves = []
        for row in protein_runs:
            if row["method"] != method:
                continue
            points = grouped[row["run_id"]]
            if x_key == "elapsed_s":
                # A checkpoint timestamp cannot exceed its run's measured search time.
                # Discard malformed legacy points rather than stretching the time axis.
                limit = _number(row, "search_s") * 1.05
                points = [point for point in points
                          if _number(point, "elapsed_s") <= limit]
            curves.append([
                _step(points, x, x_key, "ca_rmsd_ensemble_min") for x in grid])
        curves = np.asarray(curves, dtype=float)
        if not curves.size or not np.isfinite(curves).any():
            continue
        median, low, high = _band(curves)
        ax.plot(grid, median, color=COLORS[method], lw=2.8, label=LABELS[method])
        ax.fill_between(grid, low, high, color=COLORS[method], alpha=0.14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CA-RMSD of lowest-energy candidate (Å) ↓")
    ax.grid(color="#D8D8D8", linewidth=0.8, alpha=0.75)


def figure_efficiency(runs: Sequence[Dict[str, str]], traces: Sequence[Dict[str, str]],
                      output: str) -> None:
    """Separate budget efficiency from implementation wall-clock performance."""
    methods = _methods(runs)
    grouped = _group_traces(traces)
    for protein in sorted({row["protein"] for row in runs}):
        rows = [row for row in runs if row["protein"] == protein]
        fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
        for method in methods:
            mr = [row for row in rows if row["method"] == method]
            pairs = [(_number(row, "total_s"),
                      _number(row, "ca_rmsd_ensemble_min")) for row in mr]
            pairs = [(x, y) for x, y in pairs if np.isfinite(x) and np.isfinite(y)]
            if not pairs:
                continue
            x, y = map(np.asarray, zip(*pairs))
            axes[0].scatter(
                x, y, s=68, marker=MARKERS[method], color=COLORS[method],
                edgecolor="white", linewidth=0.8, alpha=0.78)
            mx, xlo, xhi, _ = _summary_interval(x, 500 + len(method))
            my, ylo, yhi, _ = _summary_interval(y, 600 + len(method))
            workers = sorted({int(_number(row, "minim_workers")) for row in mr})
            worker_label = "/".join(map(str, workers))
            worker_word = "worker" if workers == [1] else "workers"
            axes[0].errorbar(
                mx, my, xerr=[[mx - xlo], [xhi - mx]],
                yerr=[[my - ylo], [yhi - my]], fmt=MARKERS[method],
                markersize=9, color=COLORS[method], markeredgecolor="#202020",
                capsize=4, label=f"{LABELS[method]} ({worker_label} {worker_word})")
        axes[0].set_xlabel("End-to-end wall time (s) ↓")
        axes[0].set_ylabel("Ensemble-minimum CA-RMSD (Å) ↓")
        axes[0].set_title("Measured speed–accuracy tradeoff")
        axes[0].grid(color="#D8D8D8", linewidth=0.8, alpha=0.75)
        axes[0].legend(frameon=False, fontsize=9)
        _convergence_panel(
            axes[1], rows, grouped, methods, "n_energy_evaluations",
            "Unique AMBER evaluations")
        axes[1].set_title("Algorithmic efficiency (equal budget)")
        _convergence_panel(
            axes[2], rows, grouped, methods, "elapsed_s", "Elapsed search time (s)")
        axes[2].set_title("Observed implementation efficiency")
        fig.suptitle(f"Search efficiency and accuracy — {protein}",
                     fontsize=20, fontweight="bold")
        fig.text(
            0.5, 0.005,
            "Trajectory is the RMSD of the lowest-energy structure seen, not the "
            "best-RMSD structure selected after looking at the native.",
            ha="center", fontsize=11, color="#444444")
        fig.tight_layout(rect=(0, 0.045, 1, 0.92))
        _save(fig, output, f"poster_efficiency_{protein}")


def figure_objective_alignment(runs: Sequence[Dict[str, str]], output: str) -> None:
    """Show whether minimizing the reported energy improves actual structure."""
    methods = _methods(runs)
    proteins = sorted({row["protein"] for row in runs})
    fig, axes = plt.subplots(1, len(proteins), figsize=(6.2 * len(proteins), 5.4),
                             squeeze=False)
    for ax, protein in zip(axes[0], proteins):
        rows = [row for row in runs if row["protein"] == protein]
        for method in methods:
            mr = [row for row in rows if row["method"] == method]
            x = np.asarray([_number(row, "energy_gap") for row in mr])
            y = np.asarray([_number(row, "ca_rmsd_ensemble_min") for row in mr])
            keep = np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[keep], y[keep], s=75, marker=MARKERS[method],
                color=COLORS[method], edgecolor="white", linewidth=0.8,
                alpha=0.86, label=LABELS[method])
            for row, xx, yy in zip(np.asarray(mr, dtype=object)[keep], x[keep], y[keep]):
                ax.annotate(f"s{row['seed']}", (xx, yy), xytext=(4, 4),
                            textcoords="offset points", fontsize=8, color="#444444")
        ax.axvline(0, color="#333333", lw=1.2, ls=(0, (4, 3)))
        ax.set_xlabel("Predicted − native AMBER energy (kcal/mol)")
        ax.set_ylabel("Ensemble-minimum CA-RMSD (Å) ↓")
        ax.set_title(protein)
        ax.grid(color="#D8D8D8", linewidth=0.8, alpha=0.75)
    axes[0][-1].legend(frameon=False)
    fig.suptitle("Energy optimization and structural accuracy are different outcomes",
                 fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, output, "poster_energy_vs_structure")


def _representative(rows: Sequence[Dict[str, str]], method: str) -> Dict[str, str]:
    candidates = [row for row in rows if row["method"] == method
                  and np.isfinite(_number(row, "ca_rmsd_ensemble_min"))]
    if not candidates:
        raise ValueError(f"no finite {method} result available")
    values = np.asarray([_number(row, "ca_rmsd_ensemble_min") for row in candidates])
    median = np.median(values)
    return candidates[int(np.argmin(np.abs(values - median)))]


def _ensemble_medoid(models: Sequence[np.ndarray]) -> int:
    if len(models) == 1:
        return 0
    distance = np.zeros((len(models), len(models)), dtype=float)
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            distance[i, j] = distance[j, i] = geo.ca_rmsd(models[i], models[j])
    return int(np.argmin(distance.mean(axis=1)))


def figure_structure_comparison(run_dir: str, runs: Sequence[Dict[str, str]],
                                output: str, protein: str) -> None:
    """Use one native conformer, one camera, and one scale for every method panel."""
    rows = [row for row in runs if row["protein"] == protein]
    methods = _methods(rows)
    entry = ds.build_dataset([protein], verbose=False)[0]
    ensemble = ds.load_ensemble(entry)
    native_models = [np.asarray(coords["CA"], dtype=float)
                     for _, coords, _, _ in ensemble]
    medoid_i = _ensemble_medoid(native_models)
    native = native_models[medoid_i]

    aligned = {}
    representatives = {}
    for method in methods:
        row = _representative(rows, method)
        sequence, _, ca, _ = geo.parse_pdb(os.path.join(run_dir, row["pdb_path"]))
        ca = np.asarray(ca, dtype=float)
        if len(ca) != len(native):
            raise ValueError(f"{method} prediction length does not match {protein}")
        aligned[method] = (sequence, geo.kabsch_superpose(ca, native))
        representatives[method] = row

    all_points = np.vstack([native] + [aligned[m][1] for m in methods])
    center = all_points.mean(axis=0)
    radius = max(1.0, float(np.abs(all_points - center).max()) * 1.10)
    ncols = 2
    nrows = int(math.ceil(len(methods) / ncols))
    fig = plt.figure(figsize=(13.0, 5.8 * nrows))
    for index, method in enumerate(methods, start=1):
        ax = fig.add_subplot(nrows, ncols, index, projection="3d")
        sequence, predicted = aligned[method]
        row = representatives[method]
        ax.plot(*native.T, color="#303030", lw=3.2, alpha=0.86)
        ax.scatter(*native.T, color="#303030", s=34, depthshade=False)
        ax.plot(*predicted.T, color=COLORS[method], lw=3.0)
        ax.scatter(*predicted.T, color=COLORS[method], s=37, depthshade=False)
        ax.text(*native[0], " N", color="#303030", fontsize=10)
        ax.text(*native[-1], " C", color="#303030", fontsize=10)
        medoid_rmsd = geo.ca_rmsd(predicted, native)
        ax.set_title(f"{LABELS[method]} — representative seed {row['seed']}", pad=2)
        ax.text2D(
            0.5, 0.015,
            f"ensemble-min RMSD {_number(row, 'ca_rmsd_ensemble_min'):.2f} Å   •   "
            f"medoid RMSD {medoid_rmsd:.2f} Å   •   "
            f"contact F1 {_number(row, 'contact_f1'):.2f}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#D0D0D0", "alpha": 0.92})
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=20, azim=42)
        ax.set_proj_type("ortho")
        ax.set_axis_off()
    legend = [
        Line2D([0], [0], color="#303030", lw=3, label=f"Native ensemble medoid (model {medoid_i + 1})"),
        Line2D([0], [0], color=COLORS["vqe"], lw=3,
               label="Predicted backbone (method color)"),
    ]
    fig.legend(handles=legend, frameon=False, loc="lower center", ncol=2)
    fig.suptitle(
        f"Fair structure comparison — {protein}: same native, alignment, camera, and scale",
        fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    _save(fig, output, f"poster_structure_comparison_{protein}")


def _paired_vqe_gap(runs: Sequence[Dict[str, str]]) -> List[Dict[str, object]]:
    cells: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in runs:
        cells[(row["protein"], row["seed"])].append(row)
    pairs = []
    for (protein, seed), rows in sorted(cells.items()):
        vqe = [row for row in rows if row["method"] == "vqe"]
        classical = [row for row in rows if row["method"] != "vqe"
                     and np.isfinite(_number(row, "ca_rmsd_ensemble_min"))]
        if len(vqe) != 1 or not classical:
            continue
        best = min(classical, key=lambda row: _number(row, "ca_rmsd_ensemble_min"))
        gap = (_number(vqe[0], "ca_rmsd_ensemble_min")
               - _number(best, "ca_rmsd_ensemble_min"))
        pairs.append({
            "protein": protein, "seed": seed, "gap": gap,
            "best_classical": best["method"], "vqe_wins": gap < 0,
        })
    return pairs


def figure_paired_vqe_gap(runs: Sequence[Dict[str, str]], output: str) -> None:
    """Paired comparison prevents target difficulty from being pooled away."""
    pairs = _paired_vqe_gap(runs)
    if not pairs:
        return
    proteins = sorted({str(pair["protein"]) for pair in pairs})
    fig, ax = plt.subplots(figsize=(max(7.2, 2.2 * len(proteins)), 5.0))
    rng = np.random.default_rng(911)
    for x, protein in enumerate(proteins):
        values = np.asarray([float(pair["gap"]) for pair in pairs
                             if pair["protein"] == protein])
        jitter = rng.uniform(-0.09, 0.09, values.size)
        colors = [COLORS["vqe"] if value < 0 else COLORS["sa"] for value in values]
        ax.scatter(x + jitter, values, color=colors, s=72, edgecolor="white",
                   linewidth=0.8, zorder=3)
        median, low, high, interval = _summary_interval(values, 1400 + x)
        ax.errorbar(x, median, yerr=[[median - low], [high - median]], fmt="_",
                    markersize=24, markeredgewidth=3.4, capsize=5,
                    color="#202020", zorder=4)
        wins = int(np.sum(values < 0))
        ax.annotate(f"VQE wins {wins}/{values.size}\nmedian {median:+.2f} Å",
                    (x, median), xytext=(16, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                          "edgecolor": "#D0D0D0", "alpha": 0.9})
    ax.axhline(0, color="#303030", lw=1.4)
    ax.axhspan(ax.get_ylim()[0], 0, color=COLORS["vqe"], alpha=0.055)
    ax.set_xticks(range(len(proteins)), proteins)
    ax.set_ylabel("VQE − best classical CA-RMSD (Å) ↓")
    ax.set_title("Paired structural-accuracy difference by target and seed")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.8, alpha=0.75)
    fig.text(0.5, 0.01,
             "Negative values favor VQE; the classical comparator is the lowest-RMSD "
             "classical method for the same target and seed.",
             ha="center", fontsize=11, color="#444444")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    _save(fig, output, "poster_vqe_vs_best_classical")


def write_audit(runs: Sequence[Dict[str, str]], manifest: Dict, output: str) -> None:
    methods = _methods(runs)
    stats = {}
    for method in methods:
        rows = [row for row in runs if row["method"] == method]
        stats[method] = {
            "n": len(rows),
            "median_ensemble_min_ca_rmsd_angstrom": float(np.median(
                _finite(rows, "ca_rmsd_ensemble_min"))),
            "median_total_seconds": float(np.median(_finite(rows, "total_s"))),
            "median_contact_f1": float(np.median(_finite(rows, "contact_f1"))),
            "median_longrange_contact_recall": float(np.median(
                _finite(rows, "longrange_contact_recall"))),
            "median_secondary_structure_agreement": float(np.median(
                _finite(rows, "ss_agreement"))),
        }
    pairs = _paired_vqe_gap(runs)
    proteins = sorted({row["protein"] for row in runs})
    splits = sorted({row.get("split", "") for row in runs})
    minimum_cell_n = min(
        sum(row["protein"] == protein and row["method"] == method for row in runs)
        for protein in proteins for method in methods)
    held_out_scope = splits == ["test"] and len(proteins) >= 2 and minimum_cell_n >= 5
    audit = {
        "config_id": manifest.get("config_id"),
        "scope": {"proteins": proteins, "splits": splits,
                  "minimum_seeds_per_protein_method": minimum_cell_n},
        "method_statistics": stats,
        "paired_vqe_vs_best_classical": {
            "wins": sum(bool(pair["vqe_wins"]) for pair in pairs),
            "comparisons": len(pairs),
            "gaps_angstrom": pairs,
        },
        "poster_claims": {
            "held_out_multi_target_accuracy_scope": held_out_scope,
            "quantum_accuracy_advantage_supported": bool(
                held_out_scope and pairs
                and sum(bool(pair["vqe_wins"]) for pair in pairs) > len(pairs) / 2),
            "quantum_speedup_supported": False,
            "runtime_wording": (
                "Wall-clock measurements describe this classical simulator and the "
                "recorded worker counts; they are not evidence of quantum speedup."),
        },
    }
    with open(os.path.join(output, "poster_claim_audit.json"), "w",
              encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--structure-protein", action="append", default=[])
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output = os.path.abspath(
        args.output_dir or os.path.join(run_dir, "poster_figures"))
    with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    runs = _read_csv(os.path.join(run_dir, "runs.csv"))
    traces = _read_csv(os.path.join(run_dir, "checkpoints.csv"))
    _validate(runs, manifest)
    _style()
    figure_accuracy(runs, output)
    figure_fidelity(runs, output)
    figure_efficiency(runs, traces, output)
    figure_objective_alignment(runs, output)
    figure_paired_vqe_gap(runs, output)
    proteins = args.structure_protein or sorted({row["protein"] for row in runs})
    for protein in proteins:
        figure_structure_comparison(run_dir, runs, output, protein)
    write_audit(runs, manifest, output)
    print(f"poster-ready figures: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
