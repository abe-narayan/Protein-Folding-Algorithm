"""Exhaustive energy-landscape diagnosis, with a per-term variance decomposition.

The core measurement -- Spearman(energy, RMSD), enrichment, native percentile -- now
lives in `energy_quality`, where `validation.py` can gate on it. This script remains the
*exhaustive*, multiprocess version: it enumerates the whole configuration space rather
than sampling, and adds the per-term attribution that explains a bad result.

Run:  python diagnose_energy_model.py [--protein 1UAO] [--workers 8]
      python main.py --energy-quality        # the fast, sampled equivalent
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from typing import Dict, List, Optional

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import protein_geometry as geo
import representations as reps
import energy_terms as et
import hamiltonian as ham
import evaluation as ev
import energy_quality as eq

BASE = os.path.dirname(os.path.abspath(__file__))
_W: Dict = {}


def _init(seq: str, n_states: int, native_ca: np.ndarray,
          weights: Optional[Dict[str, float]]) -> None:
    rep = reps.TorsionStateRepresentation(len(seq), n_states=n_states, sequence=seq)
    _W["seq"] = seq
    _W["rep"] = rep
    _W["H"] = ham.FoldingHamiltonian(seq, rep, weights=weights, cache_limit=0)
    _W["nat"] = np.asarray(native_ca, dtype=float)
    _W["fmt"] = f"0{rep.n_bits}b"


def _chunk(args) -> tuple:
    lo, hi = args
    H, rep, fmt, nat = _W["H"], _W["rep"], _W["fmt"], _W["nat"]
    n = hi - lo
    e = np.empty(n, dtype=np.float32)
    r = np.empty(n, dtype=np.float32)
    for k in range(n):
        bs = format(lo + k, fmt)
        e[k] = H.energy(bs)
        r[k] = geo.ca_rmsd(rep.build_coords(bs)["CA"], nat)
    return lo, e, r


def diagnose(pdb_id: str, n_states: int = 4, workers: int = 8,
             weights: Optional[Dict[str, float]] = None) -> Dict:
    pdb = os.path.join(BASE, "pdbs", f"{pdb_id}.pdb")
    ensemble = geo.native_ensemble_from_pdb(pdb)
    seq, ncoords, nphi, npsi = ensemble[0]
    rep = reps.TorsionStateRepresentation(len(seq), n_states=n_states, sequence=seq)
    H = ham.FoldingHamiltonian(seq, rep, weights=weights)
    nat_ca = np.asarray(ncoords["CA"], dtype=float)
    total = 1 << rep.n_bits

    print(f"  protein {pdb_id}  seq {seq}  N={len(seq)}  "
          f"{n_states} states -> {rep.n_bits} bits, {total:,} structures")
    print(f"  residue classes: {' '.join(c[:3] for c in rep.classes)}")

    e_native_real = H.energy_from_coords(ncoords, nphi, npsi)
    snap_bits = rep.angles_to_bits(nphi, npsi)
    e_native_snap = H.energy(snap_bits)
    snap_rmsd = geo.ca_rmsd(rep.build_coords(snap_bits)["CA"], nat_ca)
    ceiling = ev.representation_ceiling(rep, nat_ca, nphi, npsi, seed=0)

    models = [np.asarray(c["CA"]) for _, c, _, _ in ensemble]
    spread = geo.ensemble_spread(models)

    print(f"  native (real backbone)   E = {e_native_real:9.4f}")
    print(f"  native (snapped)         E = {e_native_snap:9.4f}   "
          f"RMSD {snap_rmsd:.2f} A")
    print(f"  representation ceiling   RMSD {ceiling['ceiling_ca_rmsd']:.2f} A")
    print(f"  NMR ensemble spread      RMSD {spread:.2f} A  "
          f"({len(models)} models) <- resolution floor")
    print(f"  enumerating {total:,} structures on {workers} worker(s) ...", flush=True)

    t0 = time.time()
    bounds = [(i, min(i + 8192, total)) for i in range(0, total, 8192)]
    energies = np.empty(total, dtype=np.float32)
    rmsds = np.empty(total, dtype=np.float32)
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init,
                      initargs=(seq, n_states, nat_ca, weights)) as pool:
            for lo, e, r in pool.imap_unordered(_chunk, bounds, chunksize=4):
                energies[lo:lo + e.size] = e
                rmsds[lo:lo + r.size] = r
    else:
        _init(seq, n_states, nat_ca, weights)
        for b in bounds:
            lo, e, r = _chunk(b)
            energies[lo:lo + e.size] = e
            rmsds[lo:lo + r.size] = r
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    n_better = int((energies < e_native_snap).sum())
    pct = 100.0 * n_better / total
    gmin = int(np.argmin(energies))
    gmin_bits = format(gmin, f"0{rep.n_bits}b")

    order = np.argsort(energies)
    top100 = order[:100]
    top1k = order[:1000]

    out = {
        "protein": pdb_id, "sequence": seq, "n_residues": len(seq),
        "n_states": n_states, "n_structures": total,
        "residue_classes": list(rep.classes),
        "e_native_real": float(e_native_real),
        "e_native_snapped": float(e_native_snap),
        "native_snapped_rmsd": float(snap_rmsd),
        "ceiling_rmsd": float(ceiling["ceiling_ca_rmsd"]),
        "nmr_ensemble_spread": float(spread),
        "n_native_models": len(models),
        "n_structures_better_than_native": n_better,
        "native_energy_percentile": pct,
        "e_global_min": float(energies[gmin]),
        "global_min_rmsd": float(rmsds[gmin]),
        "global_min_bitstring": gmin_bits,
        "spearman_energy_vs_rmsd": eq.spearman(energies, rmsds),
        "mean_rmsd_all": float(rmsds.mean()),
        "mean_rmsd_top100_by_energy": float(rmsds[top100].mean()),
        "mean_rmsd_top1000_by_energy": float(rmsds[top1k].mean()),
        "best_rmsd_anywhere": float(rmsds.min()),
        "e_at_best_rmsd": float(energies[int(np.argmin(rmsds))]),
        "weights": dict(H.weights),
    }

    comp_nat = et.energy_components(seq, ncoords, nphi, npsi)
    comp_min = H.components(gmin_bits)
    out["terms"] = {
        t: {"weight": H.weights.get(t, 0.0),
            "native": float(H.weights.get(t, 0.0) * comp_nat.get(t, 0.0)),
            "global_min": float(H.weights.get(t, 0.0) * comp_min.get(t, 0.0))}
        for t in et.TERM_NAMES}
    return out


def report(d: Dict) -> None:
    print()
    print("=" * 74)
    print(f"  ENERGY-MODEL DIAGNOSIS -- {d['protein']} ({d['sequence']})")
    print("=" * 74)
    print(f"  native ranks #{d['n_structures_better_than_native']:,} of "
          f"{d['n_structures']:,} by energy")
    print(f"  -> {d['native_energy_percentile']:.2f}% of all representable structures "
          f"score BETTER than native")
    print()
    print(f"  {'':<34}{'energy':>10}  {'CA-RMSD':>9}")
    print(f"  {'native (real backbone)':<34}{d['e_native_real']:>10.4f}  {'-':>9}")
    print(f"  {'native (snapped to encoding)':<34}{d['e_native_snapped']:>10.4f}  "
          f"{d['native_snapped_rmsd']:>9.2f}")
    print(f"  {'global energy minimum':<34}{d['e_global_min']:>10.4f}  "
          f"{d['global_min_rmsd']:>9.2f}")
    print(f"  {'closest structure to native':<34}{d['e_at_best_rmsd']:>10.4f}  "
          f"{d['best_rmsd_anywhere']:>9.2f}")
    print()
    print(f"  Spearman(energy, RMSD)        : {d['spearman_energy_vs_rmsd']:+.4f}")
    print(f"     (+1 = lower energy means closer to native; 0 = no information)")
    print(f"  mean RMSD, all structures     : {d['mean_rmsd_all']:.2f} A")
    print(f"  mean RMSD, 1000 lowest-energy : {d['mean_rmsd_top1000_by_energy']:.2f} A")
    print(f"  mean RMSD, 100 lowest-energy  : {d['mean_rmsd_top100_by_energy']:.2f} A")
    enr = d['mean_rmsd_all'] - d['mean_rmsd_top100_by_energy']
    print(f"     enrichment                 : {enr:+.2f} A "
          f"({'helps' if enr > 0.25 else 'no useful signal' if enr > -0.25 else 'ANTI-correlated'})")
    print(f"  NMR ensemble spread           : {d['nmr_ensemble_spread']:.2f} A "
          f"over {d['n_native_models']} models")
    print()
    print(f"  per-term, native vs global minimum (weighted):")
    print(f"  {'term':<16}{'weight':>8}{'native':>12}{'global min':>12}{'delta':>12}")
    for t, v in d["terms"].items():
        delta = v["native"] - v["global_min"]
        print(f"  {t:<16}{v['weight']:>8.2f}{v['native']:>12.3f}"
              f"{v['global_min']:>12.3f}{delta:>+12.3f}")
    tot_n = sum(v["native"] for v in d["terms"].values())
    tot_m = sum(v["global_min"] for v in d["terms"].values())
    print(f"  {'TOTAL':<16}{'':>8}{tot_n:>12.3f}{tot_m:>12.3f}{tot_n - tot_m:>+12.3f}")
    worst = max(d["terms"].items(), key=lambda kv: kv[1]["native"] - kv[1]["global_min"])
    print(f"\n  largest single contribution to native's penalty: "
          f"{worst[0]} ({worst[1]['native'] - worst[1]['global_min']:+.3f})")
    if d["spearman_energy_vs_rmsd"] <= 0 or enr <= 0:
        print("\n  !! This objective is NOT usable for structure prediction. Optimising")
        print("     it does not move toward the native fold, so no RMSD from it should")
        print("     be quoted in either direction. Run `main.py --calibrate-weights`.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protein", default="1UAO")
    ap.add_argument("--states", type=int, default=4, choices=[4, 8])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--weights-json", default=None,
                    help="use calibrated weights from results/weight_calibration.json")
    ap.add_argument("--out", default="results/energy_model_diagnosis.json")
    a = ap.parse_args()

    weights = None
    if a.weights_json:
        with open(a.weights_json) as fh:
            blob = json.load(fh)
        weights = blob.get("weights", blob)

    d = diagnose(a.protein, n_states=a.states, workers=a.workers, weights=weights)
    report(d)
    path = os.path.join(BASE, a.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)
    print(f"\n  written to {path}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
