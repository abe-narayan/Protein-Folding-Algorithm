"""PROFILE 2: decompose the MPS sampler cost (build vs sample), shots scaling,
and qubit scaling (30 vs 36). Read-only; no repo edits."""
import os, sys, time, math, json
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.insert(0, REPO)
from mps.sampler import MPSSampler
import quimb.tensor as qtn

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out")
os.makedirs(OUT, exist_ok=True)

def bench(n_qubits, layers=4, shots_list=(256,512,1024,2048), reps=3):
    s = MPSSampler(n_qubits, layers=layers)
    n_par = layers * n_qubits
    p = (math.pi/2.0) + np.random.default_rng(0).normal(0.0, 0.6, size=n_par)
    out = {"n_qubits": n_qubits, "layers": layers}
    # build-only (apply gates, no sampling)
    tb = []
    for _ in range(reps):
        c0 = time.perf_counter()
        circ = s._circuit(p)
        _ = circ.psi  # force realize
        tb.append(time.perf_counter()-c0)
    out["build_only_s"] = min(tb)
    out["bond_dim"] = int(s.bond_dim(p))
    # build+sample for each shot count
    out["by_shots"] = {}
    for sh in shots_list:
        ts = []
        for _ in range(reps):
            rr = np.random.default_rng(3)
            c0 = time.perf_counter()
            s.draw_indices(p, sh, rr)
            ts.append(time.perf_counter()-c0)
        out["by_shots"][sh] = {"build+sample_s": min(ts),
                                "sample_only_s": min(ts)-out["build_only_s"]}
    # sample-only per-shot at 1024
    t1024 = out["by_shots"][1024]["build+sample_s"]
    out["per_shot_ms_at_1024"] = (t1024 - out["build_only_s"])/1024*1e3
    out["est_per_objeval_s_1024"] = t1024
    return out

def main():
    res = {}
    for nq in (30, 36):
        res[str(nq)] = bench(nq)
        print(json.dumps(res[str(nq)], indent=2), flush=True)
    # extrapolate to full runs
    res["extrap"] = {
        "chignolin_30q": {"objevals": 600, "mps_hours": 600*res["30"]["by_shots"][1024]["build+sample_s"]/3600},
        "trpzip_36q":    {"objevals": 600, "mps_hours": 600*res["36"]["by_shots"][1024]["build+sample_s"]/3600},
    }
    print(json.dumps(res["extrap"], indent=2), flush=True)
    with open(os.path.join(OUT, "prof2_mps.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE prof2_mps.json", flush=True)

if __name__ == "__main__":
    main()
