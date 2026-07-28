"""PROFILE 4: EXACT cache-effectiveness trajectory.
Replays the real seed-0 driver (real MPS sampler + real COBYLA + real CVaR) but serves
energies from the persisted golden CSV via O(1) lookup instead of OpenMM. Because the
served energies are byte-identical to the real run, COBYLA follows the EXACT same
trajectory and the sampled index-sets per eval are identical -> the cache hit/miss
dynamics reproduce the real run exactly. Checkpoints cumulative stats every eval so a
partial run is still informative.
"""
import os, sys, time, csv, math, json
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.insert(0, REPO)
from vqe import cvar_from_samples, BestSeenTracker, n_parameters
from mps.sampler import MPSSampler
from scipy.optimize import minimize

GOLDEN = r"C:\Users\abena\AppData\Local\Temp\claude\C--Abhyuth-claude-protein-build\04a0f21c-acaa-469d-99b6-8775f3298c2d\scratchpad\seed0_full_eval_set.csv"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out")
CKPT = os.path.join(OUT, "prof4_cache_ckpt.json")
N_QUBITS, LAYERS, ALPHA, SHOTS, MAXITER, RESTARTS, SEED = 30, 4, 0.15, 1024, 200, 3, 0
INIT_SCALE = 0.25

def main():
    os.makedirs(OUT, exist_ok=True)
    gold = {}
    with open(GOLDEN, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            try: gold[row[0]] = float(row[1])
            except Exception: gold[row[0]] = float("inf")
    fmt = f"0{N_QUBITS}b"

    cache = {}
    stats = {"total_energy_calls": 0, "misses": 0, "hits": 0,
             "unique_per_eval": [], "new_per_eval": [], "n_evals": 0,
             "missing_from_gold": 0}

    def energy(bs):
        stats["total_energy_calls"] += 1
        v = cache.get(bs)
        if v is not None:
            stats["hits"] += 1
            return v
        stats["misses"] += 1
        g = gold.get(bs)
        if g is None:
            stats["missing_from_gold"] += 1
            g = 0.0  # unseen by real run; shouldn't happen if trajectory matches
        cache[bs] = g
        return g

    samp = MPSSampler(N_QUBITS, layers=LAYERS)
    tracker = BestSeenTracker()
    restart_seeds = [int(s.generate_state(1)[0])
                     for s in np.random.SeedSequence(SEED).spawn(RESTARTS)]
    t0 = time.time()

    def run_single(rseed):
        init_rng = np.random.default_rng(np.random.SeedSequence([rseed, 0xC0FFEE]))
        params0 = (math.pi/2.0) + init_rng.normal(0.0, INIT_SCALE, size=LAYERS*N_QUBITS)
        ev = {"n": 0}
        def objective(params):
            k = ev["n"]; ev["n"] = k+1
            rng = np.random.default_rng(np.random.SeedSequence([rseed, k]))
            idx = samp.draw_indices(params, SHOTS, rng)
            uniq, inverse = np.unique(idx, return_inverse=True)
            before_miss = stats["misses"]
            ue = np.empty(uniq.size)
            for m, u in enumerate(uniq):
                bs = format(int(u), fmt)
                e = energy(bs); ue[m] = e; tracker.offer(bs, e)
            new = stats["misses"] - before_miss
            stats["unique_per_eval"].append(int(uniq.size))
            stats["new_per_eval"].append(int(new))
            stats["n_evals"] += 1
            # checkpoint every eval
            tc = stats["total_energy_calls"]
            snap = {"n_evals": stats["n_evals"], "cumulative_unique": len(cache),
                    "total_energy_calls": tc, "hits": stats["hits"],
                    "hit_rate": stats["hits"]/tc if tc else 0.0,
                    "elapsed_s": time.time()-t0,
                    "missing_from_gold": stats["missing_from_gold"],
                    "mean_unique_per_eval": float(np.mean(stats["unique_per_eval"])),
                    "mean_new_per_eval": float(np.mean(stats["new_per_eval"]))}
            with open(CKPT, "w") as f:
                json.dump(snap, f, indent=2)
            val = cvar_from_samples(ue[inverse], ALPHA)
            return val
        minimize(objective, params0, method="COBYLA",
                 options={"maxiter": MAXITER, "rhobeg": 0.4})

    for rs in restart_seeds:
        run_single(rs)

    final = {"n_evals": stats["n_evals"], "cumulative_unique": len(cache),
             "total_energy_calls": stats["total_energy_calls"],
             "hits": stats["hits"], "misses": stats["misses"],
             "hit_rate": stats["hits"]/stats["total_energy_calls"],
             "genuine_minimizations": stats["misses"],
             "reduction_factor_vs_no_cache": stats["total_energy_calls"]/stats["misses"],
             "mean_unique_per_eval": float(np.mean(stats["unique_per_eval"])),
             "mean_new_per_eval": float(np.mean(stats["new_per_eval"])),
             "missing_from_gold": stats["missing_from_gold"],
             "elapsed_s": time.time()-t0}
    with open(os.path.join(OUT, "prof4_cache_final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(json.dumps(final, indent=2), flush=True)

if __name__ == "__main__":
    main()
