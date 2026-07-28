"""PROFILE 1 (read-only instrumentation on a scratchpad COPY).
Per-energy-evaluation OpenMM cost breakdown + MPS sampler cost + golden verify.
Chignolin GYDPETGTWG, 8-state rep, OBC2-native — the exact seed-0 construction.
"""
import os, sys, time, csv, json, math
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.insert(0, REPO)

import amber_hamiltonian as amb
import representations as reps
from amber_obc2 import build_obc2_native
from mps.sampler import MPSSampler

GOLDEN = r"C:\Users\abena\AppData\Local\Temp\claude\C--Abhyuth-claude-protein-build\04a0f21c-acaa-469d-99b6-8775f3298c2d\scratchpad\seed0_full_eval_set.csv"
SEQ = "GYDPETGTWG"
N_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out")
os.makedirs(OUT, exist_ok=True)

def main():
    t0 = time.time()
    rep = reps.TorsionStateRepresentation(len(SEQ), n_states=8)
    H = build_obc2_native(SEQ, rep)
    t_setup = time.time() - t0
    desc = H.describe()
    print("SETUP", json.dumps({k: desc[k] for k in ("n_atoms","n_hydrogens","platform","platform_properties","minimization_steps")}), flush=True)
    print(f"n_qubits={H.n_qubits} n_bits={H.n_bits} setup_time_s={t_setup:.3f}", flush=True)

    # load golden pairs
    gold = {}
    with open(GOLDEN, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            try: gold[row[0]] = float(row[1])
            except Exception: gold[row[0]] = float("inf")
    keys = list(gold.keys())
    print(f"golden rows={len(keys)}", flush=True)

    rng = np.random.default_rng(12345)
    sample_keys = [keys[i] for i in rng.choice(len(keys), size=min(N_EVAL, len(keys)), replace=False)]

    # ---- per-energy timing (fresh cache) ----
    H.reset_counters()
    for a in ("_t_setpos","_t_restraint","_t_minim_only"):
        setattr(H, a, 0.0)
    per_call = np.empty(len(sample_keys))
    mism = []
    t_wall0 = time.time()
    for i, bs in enumerate(sample_keys):
        c0 = time.perf_counter()
        e = H.energy(bs)
        per_call[i] = time.perf_counter() - c0
        g = gold[bs]
        if math.isfinite(g) and math.isfinite(e):
            if abs(e - g) > 1e-6:
                mism.append((bs, e, g, abs(e-g)))
        elif math.isfinite(g) != math.isfinite(e):
            mism.append((bs, e, g, float("inf")))
    wall = time.time() - t_wall0

    n = len(sample_keys)
    res = {
        "n_eval": n,
        "wall_total_s": wall,
        "per_call_ms_mean": float(per_call.mean()*1e3),
        "per_call_ms_median": float(np.median(per_call)*1e3),
        "per_call_ms_p90": float(np.percentile(per_call,90)*1e3),
        "per_call_ms_min": float(per_call.min()*1e3),
        "per_call_ms_max": float(per_call.max()*1e3),
        "split_ms_per_eval": {
            "assemble_H(t_build)": H.t_build/n*1e3,
            "setPositions": H._t_setpos/n*1e3,
            "restraint_update": H._t_restraint/n*1e3,
            "minimize_only": H._t_minim_only/n*1e3,
            "getState_energy(t_energy)": H.t_energy/n*1e3,
        },
        "split_pct": {},
        "golden_checked": sum(1 for bs in sample_keys if math.isfinite(gold[bs])),
        "golden_mismatches": len(mism),
        "golden_max_abs_err": max((m[3] for m in mism), default=0.0),
    }
    tot = sum(res["split_ms_per_eval"].values())
    for k, v in res["split_ms_per_eval"].items():
        res["split_pct"][k] = 100.0*v/tot
    res["sum_split_ms"] = tot

    # ---- MPS sampler cost (per objective eval) ----
    samp = MPSSampler(H.n_qubits, layers=4)
    n_par = 4 * H.n_qubits
    def time_sampler(params, shots, reps_=3):
        ts = []
        for _ in range(reps_):
            rr = np.random.default_rng(7)
            c0 = time.perf_counter()
            samp.draw_indices(params, shots, rr)
            ts.append(time.perf_counter()-c0)
        return min(ts)
    init_rng = np.random.default_rng(np.random.SeedSequence([0, 0xC0FFEE]))
    p_init = (math.pi/2.0) + init_rng.normal(0.0, 0.25, size=n_par)
    p_rand = np.random.default_rng(1).normal(math.pi/2, 0.6, size=n_par)
    t_samp_init = time_sampler(p_init, 1024)
    t_samp_rand = time_sampler(p_rand, 1024)
    # unique-per-1024 for these param sets
    def uniq_count(params):
        rr = np.random.default_rng(99)
        idx = samp.draw_indices(params, 1024, rr)
        return int(np.unique(idx).size)
    res["mps_sampler"] = {
        "per_eval_s_init_params": t_samp_init,
        "per_eval_s_rand_params": t_samp_rand,
        "unique_per_1024_init": uniq_count(p_init),
        "unique_per_1024_rand": uniq_count(p_rand),
        "bond_dim_init": samp.bond_dim(p_init),
    }

    print(json.dumps(res, indent=2), flush=True)
    with open(os.path.join(OUT, "prof1_chignolin.json"), "w") as f:
        json.dump(res, f, indent=2)
    if mism:
        print("FIRST MISMATCHES:", mism[:5], flush=True)
    print("WROTE", os.path.join(OUT, "prof1_chignolin.json"), flush=True)

if __name__ == "__main__":
    main()
