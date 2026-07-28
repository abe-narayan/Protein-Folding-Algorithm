import os, sys, time, math, json
import numpy as np
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.insert(0, REPO)
from mps.sampler import MPSSampler

nq = int(sys.argv[1]) if len(sys.argv) > 1 else 30
shots = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
layers = 4
s = MPSSampler(nq, layers=layers)
p = (math.pi/2.0) + np.random.default_rng(0).normal(0.0, 0.6, size=layers*nq)

# build only (apply gates)
tb = []
for _ in range(2):
    c0 = time.perf_counter(); circ = s._circuit(p); _ = circ.psi.norm(); tb.append(time.perf_counter()-c0)
build = min(tb)
bond = int(s.bond_dim(p))
print(f"nq={nq} build_only_s={build:.3f} bond_dim={bond}", flush=True)

# build + sample
tbs = []
for _ in range(2):
    rr = np.random.default_rng(3)
    c0 = time.perf_counter(); s.draw_indices(p, shots, rr); tbs.append(time.perf_counter()-c0)
bs = min(tbs)
print(f"nq={nq} shots={shots} build+sample_s={bs:.3f} sample_only_s={bs-build:.3f} per_shot_ms={(bs-build)/shots*1e3:.3f}", flush=True)

# sample-only at fixed built circuit, direct psi.sample to see if CircuitMPS.sample adds overhead
circ = s._circuit(p)
psi = circ.psi
t2 = []
for _ in range(2):
    c0 = time.perf_counter()
    cfgs = list(psi.sample(shots, seed=3))
    t2.append(time.perf_counter()-c0)
print(f"nq={nq} psi.sample_only_s={min(t2):.3f} (direct MPS.sample of {shots})", flush=True)

out = {"nq": nq, "shots": shots, "layers": layers, "build_only_s": build,
       "build+sample_s": bs, "sample_only_s": bs-build, "bond_dim": bond,
       "psi_sample_only_s": min(t2), "per_shot_ms": (bs-build)/shots*1e3}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out")
with open(os.path.join(OUT, f"prof2b_{nq}q.json"), "w") as f:
    json.dump(out, f, indent=2)
print("WROTE", f"prof2b_{nq}q.json", flush=True)
