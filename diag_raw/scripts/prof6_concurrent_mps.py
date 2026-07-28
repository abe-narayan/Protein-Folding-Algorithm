"""PROFILE 6: does running independent restarts in parallel actually speed up the
MPS-sampling bottleneck, or do concurrent MPS processes contend (BLAS/memory)?
Measures throughput of 1 process doing K MPS build+sample ops vs 3 concurrent
processes each doing K ops. This is the payoff test for restart-level parallelism."""
import os, sys, time, math, json
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import multiprocessing as mp

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
NQ = 30
K = int(sys.argv[1]) if len(sys.argv) > 1 else 12

def worker(args):
    k, tag = args
    sys.path.insert(0, REPO)
    from mps.sampler import MPSSampler
    s = MPSSampler(NQ, layers=4)
    npar = 4*NQ
    rng = np.random.default_rng(tag)
    t0 = time.time()
    for i in range(k):
        p = (math.pi/2) + rng.normal(0, 0.3, size=npar)
        s.draw_indices(p, 1024, np.random.default_rng(i+tag))
    return time.time()-t0

def main():
    # 1-process baseline
    t0 = time.time()
    solo = worker((K, 0))
    solo_wall = time.time()-t0
    # 3 concurrent processes each doing K
    t0 = time.time()
    with mp.Pool(3) as pool:
        parts = pool.map(worker, [(K, 100), (K, 200), (K, 300)])
    par_wall = time.time()-t0
    total_ops_par = 3*K
    res = {
        "n_qubits": NQ, "ops_per_proc": K,
        "solo_wall_s": solo_wall, "solo_per_op_s": solo_wall/K,
        "par3_wall_s": par_wall, "par3_total_ops": total_ops_par,
        "par3_per_op_s": par_wall/K,  # each proc did K, wall is max
        "concurrent_throughput_speedup": (3*K/par_wall)/(K/solo_wall),
        "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    print(json.dumps(res, indent=2), flush=True)
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out")
    with open(os.path.join(OUT, "prof6_concurrent_mps.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("WROTE prof6_concurrent_mps.json", flush=True)

if __name__ == "__main__":
    mp.freeze_support()
    main()
