"""PROFILE 3: OpenMM minimization parallelism headroom.
The per-structure minimizations inside a CVaR step are independent. Measure serial
vs K-process throughput (each worker = its own OBC2 Hamiltonian, Threads=1).
Behavior check: energies from workers must match the golden CSV exactly."""
import os, sys, time, csv, math, json
import numpy as np
import multiprocessing as mp

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
GOLDEN = r"C:\Users\abena\AppData\Local\Temp\claude\C--Abhyuth-claude-protein-build\04a0f21c-acaa-469d-99b6-8775f3298c2d\scratchpad\seed0_full_eval_set.csv"
SEQ = "GYDPETGTWG"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out")

_H = None
def _init():
    global _H
    sys.path.insert(0, REPO)
    import representations as reps
    from amber_obc2 import build_obc2_native
    rep = reps.TorsionStateRepresentation(len(SEQ), n_states=8)
    _H = build_obc2_native(SEQ, rep)

def _work(bslist):
    global _H
    out = []
    for bs in bslist:
        out.append((bs, _H.energy(bs)))
    return out

def _chunks(lst, k):
    # round-robin into k chunks (balanced)
    return [lst[i::k] for i in range(k)]

def serial_baseline(keys):
    _init()
    t0 = time.time()
    res = _work(keys)
    return time.time()-t0, dict(res)

def pool_run(keys, k):
    ch = _chunks(keys, k)
    t0 = time.time()
    with mp.Pool(processes=k, initializer=_init) as pool:
        parts = pool.map(_work, ch)
    dt = time.time()-t0
    merged = {}
    for p in parts:
        merged.update(dict(p))
    return dt, merged

def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, REPO)
    gold = {}
    with open(GOLDEN, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            try: gold[row[0]] = float(row[1])
            except Exception: gold[row[0]] = float("inf")
    allkeys = list(gold.keys())
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2400
    ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1,2,4,8]
    rng = np.random.default_rng(2024)
    keys = [allkeys[i] for i in rng.choice(len(allkeys), size=N, replace=False)]

    def verify(d):
        bad = 0
        for bs, e in d.items():
            g = gold[bs]
            if math.isfinite(g) and math.isfinite(e) and abs(e-g) > 1e-6: bad += 1
            elif math.isfinite(g) != math.isfinite(e): bad += 1
        return bad

    results = {"N": N, "cpu_count": os.cpu_count(), "runs": {}}
    # serial baseline (in-process)
    t_ser, d_ser = serial_baseline(keys)
    results["runs"]["serial_inproc"] = {"wall_s": t_ser, "per_min_ms": t_ser/N*1e3,
                                        "throughput_per_s": N/t_ser, "mismatches": verify(d_ser)}
    print("serial", results["runs"]["serial_inproc"], flush=True)
    base = t_ser
    for k in ks:
        dt, d = pool_run(keys, k)
        results["runs"][f"pool_{k}"] = {
            "wall_s": dt, "per_min_ms": dt/N*1e3, "throughput_per_s": N/dt,
            "speedup_vs_serial": base/dt, "mismatches": verify(d)}
        print(f"pool k={k}", results["runs"][f"pool_{k}"], flush=True)
    with open(os.path.join(OUT, "prof3_parallel.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("WROTE prof3_parallel.json", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()
