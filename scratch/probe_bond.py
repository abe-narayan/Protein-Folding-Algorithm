"""Measure EXACT (cutoff=0) MPS bond dimension + per-sample cost of the layers-4
ring ansatz at n=20,26,30. This decides MPS viability for the 30q VQE."""
import sys, time
import numpy as np
import quimb.tensor as qtn
print("interp:", sys.executable, flush=True); assert "WindowsApps" in sys.executable
LAYERS=4
def build(n, max_bond=None, cutoff=0.0, seed=0):
    p=((np.pi/2)+np.random.default_rng(seed).normal(0,0.25,size=LAYERS*n)).reshape(LAYERS,n)
    circ=qtn.CircuitMPS(N=n, max_bond=max_bond, cutoff=cutoff)
    for l in range(LAYERS):
        for q in range(n): circ.apply_gate('RY', float(p[l][q]), q)
        for q in range(n-1): circ.apply_gate('CNOT', q, q+1)
        if n>2: circ.apply_gate('CNOT', n-1, 0)
    return circ

# seed reproducibility (cheap)
c=build(10);
r=list(c.sample(16,seed=42))==list(c.sample(16,seed=42))
print("seed reproducible (same seed -> same draws):", r, flush=True)

for n in (20, 26, 30):
    t0=time.time(); c=build(n, max_bond=None, cutoff=0.0); tb=time.time()-t0
    mb=c.psi.max_bond()
    t0=time.time(); one=list(c.sample(1, seed=0)); t1=time.time()-t0
    t0=time.time(); k=64; some=list(c.sample(k, seed=0)); tk=time.time()-t0
    print(f"n={n}: build {tb:6.2f}s  EXACT bond={mb:5d}  "
          f"sample1 {t1:5.2f}s  sample{k} {tk:6.2f}s  -> est 2048: {tk/k*2048:6.1f}s/eval",
          flush=True)
