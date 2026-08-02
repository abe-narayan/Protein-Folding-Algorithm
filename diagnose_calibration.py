"""Does `calibrate_weights` improve the objective on peptides it did not see?

Measured answer: no. It makes it worse on every held-out target, and the shipped
`DEFAULT_WEIGHTS` are the best setting tested.

    max_ratio  solvation  compact   hb_lr  coop_h  HELD-OUT sp     enr   verdict
    (default)      0.500    0.400    3.00    2.00      +0.0552   +0.94   usable
            3      0.167    0.133    9.00    3.74      -0.1000   +0.54   NOT usable
           10      0.078    0.072   13.36    3.74      -0.1675   +0.42   NOT usable
           30      0.078    0.072   13.36    3.74      -0.1675   +0.40   NOT usable
          100      0.078    0.072   13.36    3.74      -0.1675   +0.40   NOT usable
         1000      0.078    0.072   13.36    3.74      -0.1675   +0.40   NOT usable

Per held-out target, default vs calibrated Spearman:

    1L2Y  +0.1019 / -0.0568      2EVQ  +0.0768 / -0.0969
    2JOF  +0.0822 / -0.0713      1J4M  -0.0400 / -0.1752

Default wins on all four, on both Spearman and enrichment, and is positive on
three of four. So this is not one outlier moving a mean.

WHY, and why `max_ratio` is not the answer. `calibrate_weights` balances
*variance*: each free weight is set so `weight * std(term)` matches the median
of the pinned terms. No native structure enters it -- by design, and
`validation.test_calibration_is_sequence_set_derived` enforces that. The
criterion is therefore orthogonal to "does low energy mean low RMSD", and on
this term set it turns out to be anti-correlated with it.

The mechanism is visible in the term statistics over random feasible structures:

    solvation      std 8.637   -> 54.6 % of discriminating variance
    compactness    std 9.333   -> 40.8 %
    ------------------------------------------  95.4 % combined
    hbond_longrange  std 0.050 ->  0.1 %
    aromatic         std 0.051 ->  0.0 %
    coop_sheet       std 0.000 ->  0.0 %

Random torsions essentially never form a hydrogen bond, so the structural terms
are numerically zero across the sampled ensemble. Variance balancing reads their
tiny std and boosts them to their ceilings -- amplifying rare, mostly spurious
values -- while suppressing solvation and compactness, which despite being crude
packing measures carry what real signal the objective has. Widening `max_ratio`
just does more of the same, and saturates at 10 once the balance target is met:
the guardrail was never the binding constraint, the criterion is.

CONSEQUENCE FOR USERS: do not run `--calibrate-weights` and then predict with the
result. `DEFAULT_WEIGHTS` is the better model on held-out clusters. The honest fix
is a calibration objective that optimises held-out energy/RMSD agreement rather
than variance balance, which is a different function and a larger change.

Run:  python diagnose_calibration.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import dataset as ds
import energy_quality as eq
import energy_terms as et
import hamiltonian as ham
import representations as reps

N_SAMPLE = 4000
SEED = 0


def evaluate(entries, weights):
    """Per-peptide Spearman / enrichment at fixed weights."""
    sp, en = [], []
    for e in entries:
        seq, nc, nphi, npsi = ds.load_native(e)
        rep = reps.TorsionStateRepresentation(len(seq), n_states=4, sequence=seq)
        H = ham.FoldingHamiltonian(seq, rep, weights=weights, cache_limit=0)
        q = eq.energy_quality(H, rep, np.asarray(nc["CA"], float),
                              nphi, npsi, n_sample=N_SAMPLE, seed=SEED)
        sp.append(q["spearman"])
        en.append(q["enrichment"])
    return sp, en


def main() -> int:
    entries = ds.build_dataset(verbose=False)
    train, val, test = ds.split_by_cluster(entries, seed=SEED)
    assert ds.check_no_cluster_leak(train, val, test)
    print(f"train {[e.pdb_id for e in train]}")
    print(f"test  {[e.pdb_id for e in test]}  (held out)\n")

    train_seqs = [e.sequence for e in train]
    default = dict(et.DEFAULT_WEIGHTS)

    print(f"{'max_ratio':>10}{'solvation':>11}{'compact':>9}{'hb_lr':>8}"
          f"{'coop_h':>8}{'HELD-OUT sp':>13}{'enr':>8}   verdict")
    sp, en = evaluate(test, default)
    print(f"{'(default)':>10}{default['solvation']:>11.3f}"
          f"{default['compactness']:>9.3f}{default['hbond_longrange']:>8.2f}"
          f"{default['coop_helix']:>8.2f}{np.mean(sp):>13.4f}{np.mean(en):>+8.2f}"
          f"   {'usable' if np.mean(sp) > 0 else 'NOT usable'}")
    per_target = {"default": sp}

    for mr in (3.0, 10.0, 30.0, 100.0, 1000.0):
        w = eq.calibrate_weights(train_seqs, n_states=4, n_sample=2000,
                                 seed=SEED, max_ratio=mr)
        sp, en = evaluate(test, w)
        if mr == 3.0:
            per_target["calibrated"] = sp
        print(f"{mr:>10.0f}{w['solvation']:>11.3f}{w['compactness']:>9.3f}"
              f"{w['hbond_longrange']:>8.2f}{w['coop_helix']:>8.2f}"
              f"{np.mean(sp):>13.4f}{np.mean(en):>+8.2f}   "
              f"{'usable' if np.mean(sp) > 0 else 'NOT usable'}")

    print("\n  per held-out target, Spearman "
          "(default vs calibrated at the shipped max_ratio=3):")
    for e, d, c in zip(test, per_target["default"], per_target["calibrated"]):
        print(f"    {e.pdb_id:<7}{d:>+9.4f}{c:>+9.4f}   "
              f"{'default wins' if d > c else 'calibrated wins'}")
    n_better = sum(1 for d, c in zip(per_target["default"],
                                     per_target["calibrated"]) if d > c)
    print(f"\n  default beats calibrated on {n_better}/{len(test)} held-out targets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
