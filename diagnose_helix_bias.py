"""Measure the helix bias DIRECTLY, on every peptide, instead of via RMSD proxies.

Everything else in this repo measures the bias indirectly: RMSD enrichment, Spearman,
per-term energies on one target. Those are proxies, and every proxy-based conclusion in
this investigation that rested on a single protein turned out to be an artefact. This
asks the question the `energy_quality` gate structurally cannot (see
`docs/hbond-channel-findings.md`): *what secondary structure does the objective actually
prefer, per peptide, compared with the native?*

Method. For each peptide, sample structures, DISCARD THE CLASHING ONES, take the
lowest-energy `TOP_N` of what remains, assign secondary structure with
`geo.assign_secondary_structure`, and report mean helix and strand fraction against the
native's. `bias = predicted helix fraction - native helix fraction`; positive means the
objective invents helix that is not there.

The feasibility filter is not optional and the first version of this script omitted it,
which produced a badly wrong conclusion. Without it, the beta-rich structures in a random
pool are overwhelmingly interpenetrating ones -- comparing weighted per-term means of
beta-rich against low-energy structures gives steric +151.8 out of a +150.5 total gap,
while every topology term mildly FAVOURS the beta set. Read uncontrolled, that looks like
"the objective rejects beta"; it is really "the objective rejects clashes, and random beta
is nearly all clash". `energy_quality` filters on steric for the same reason
(`FEASIBLE_FRACTION`), and any measurement that skips it is measuring the steric term.

This is the measurement a topology gate would assert on, and the number any proposed fix
for the helix bias has to move. It is deliberately reported per peptide as well as in
aggregate, because a mean over 12 peptides can hide the same single-target artefact that
misled the earlier passes.

Run:  python diagnose_helix_bias.py
      python diagnose_helix_bias.py --weights-json results/weight_calibration.json
"""
import argparse
import json
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import dataset as ds
import energy_terms as et
import hamiltonian as ham
import protein_geometry as geo
import representations as reps

N_SAMPLE = 4000
TOP_N = 100
SEED = 0


def ss_fractions(ss: str):
    n = max(len(ss), 1)
    return ss.count("H") / n, ss.count("E") / n


def analyse(entry, weights, n_sample=N_SAMPLE, top_n=TOP_N, seed=SEED):
    seq, ncoords, nphi, npsi = ds.load_native(entry)
    rep = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
    H = ham.FoldingHamiltonian(seq, rep, weights=weights, cache_limit=0)

    nat_h, nat_e = ss_fractions(geo.assign_secondary_structure(ncoords))

    rng = np.random.default_rng(seed)
    built = [rep.build_all(rep.random_bitstring(rng)) for _ in range(n_sample)]
    comps = [et.energy_components(seq, co, ph, ps, rings=rg)
             for ph, ps, co, rg in built]

    # Clash-free only -- see the module docstring for what happens without this.
    feasible = [i for i, c in enumerate(comps) if c["steric"] <= 1e-9]
    if len(feasible) < top_n:
        top_n = max(1, len(feasible))
    energies = np.array([et.total_from_components(comps[i], H.weights)
                         for i in feasible])
    order = [feasible[k] for k in np.argsort(energies, kind="stable")[:top_n]]

    hs, es = [], []
    pool_e = []
    for i in feasible:
        pool_e.append(ss_fractions(geo.assign_secondary_structure(built[i][2]))[1])
    for k in order:
        h, e = ss_fractions(geo.assign_secondary_structure(built[k][2]))
        hs.append(h)
        es.append(e)
    return {
        "pdb": entry.pdb_id, "seq": seq, "n": len(seq),
        "native_helix": nat_h, "native_strand": nat_e,
        "n_feasible": len(feasible),
        "pool_strand": float(np.mean(pool_e)) if pool_e else 0.0,
        "pool_strand_max": float(np.max(pool_e)) if pool_e else 0.0,
        "pred_helix": float(np.mean(hs)), "pred_strand": float(np.mean(es)),
        "helix_bias": float(np.mean(hs)) - nat_h,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights-json", default=None)
    ap.add_argument("--sample", type=int, default=N_SAMPLE)
    ap.add_argument("--top", type=int, default=TOP_N)
    a = ap.parse_args()

    weights = None
    if a.weights_json:
        with open(a.weights_json) as fh:
            blob = json.load(fh)
        weights = blob.get("weights", blob)
    label = a.weights_json or "DEFAULT_WEIGHTS"

    entries = ds.build_dataset(verbose=False)
    print(f"weights: {label}   {a.sample} sampled, top {a.top} by energy\n")
    print(f"{'pdb':<7}{'seq':<22}{'nat H':>7}{'nat E':>7}"
          f"{'pred H':>8}{'pred E':>8}{'BIAS':>8}{'feas':>7}"
          f"{'poolE':>8}{'poolEmax':>9}")
    rows = []
    for e in entries:
        r = analyse(e, weights, n_sample=a.sample, top_n=a.top)
        rows.append(r)
        print(f"{r['pdb']:<7}{r['seq']:<22}{r['native_helix']:>7.2f}"
              f"{r['native_strand']:>7.2f}{r['pred_helix']:>8.2f}"
              f"{r['pred_strand']:>8.2f}{r['helix_bias']:>+8.2f}"
              f"{r['n_feasible']:>7}{r['pool_strand']:>8.3f}"
              f"{r['pool_strand_max']:>9.3f}")

    mb = float(np.mean([r["helix_bias"] for r in rows]))
    n_pos = sum(1 for r in rows if r["helix_bias"] > 0.05)
    print(f"\n  mean helix bias {mb:+.3f} over {len(rows)} peptides")
    print(f"  {n_pos}/{len(rows)} peptides show the objective inventing helix "
          f"(bias > +0.05)")
    print(f"  mean predicted helix fraction {np.mean([r['pred_helix'] for r in rows]):.3f}"
          f"  vs native {np.mean([r['native_helix'] for r in rows]):.3f}")
    print(f"  mean predicted strand fraction {np.mean([r['pred_strand'] for r in rows]):.3f}"
          f" vs native {np.mean([r['native_strand'] for r in rows]):.3f}")
    print(f"  mean strand in the FEASIBLE pool "
          f"{np.mean([r['pool_strand'] for r in rows]):.3f} -- if this is ~0 the sampler,"
          f" not the objective, is what has no beta in it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
