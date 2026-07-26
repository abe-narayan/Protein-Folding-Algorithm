"""Re-probe with cutoff=1e-12 (keep only physically-significant singular values).
Effective bond + build + sample cost at n=20,26,30, and accuracy vs double-precision
lightning-equivalent (default.qubit) at n=20. This decides MPS viability."""
import sys, time
import numpy as np
import quimb.tensor as qtn
import pennylane as qml
print("interp:", sys.executable, flush=True); assert "WindowsApps" in sys.executable
LAYERS=4; CUT=1e-12
def build(n, seed=0):
    p=((np.pi/2)+np.random.default_rng(seed).normal(0,0.25,size=LAYERS*n)).reshape(LAYERS,n)
    circ=qtn.CircuitMPS(N=n, max_bond=8192, cutoff=CUT)
    for l in range(LAYERS):
        for q in range(n): circ.apply_gate('RY', float(p[l][q]), q)
        for q in range(n-1): circ.apply_gate('CNOT', q, q+1)
        if n>2: circ.apply_gate('CNOT', n-1, 0)
    return circ

# accuracy at n=20 vs double-precision statevector (default.qubit)
n=20
p=((np.pi/2)+np.random.default_rng(0).normal(0,0.25,size=LAYERS*n))
dev=qml.device("default.qubit",wires=n)
@qml.qnode(dev)
def st(pp):
    pp=pp.reshape(LAYERS,n)
    for l in range(LAYERS):
        for q in range(n): qml.RY(float(pp[l][q]),wires=q)
        for q in range(n-1): qml.CNOT(wires=[q,q+1])
        qml.CNOT(wires=[n-1,0])
    return qml.probs(wires=range(n))
pl=np.asarray(st(p))
c=build(n); pq=np.abs(np.asarray(c.psi.to_dense()).flatten())**2
print(f"n=20 accuracy vs double statevector: max|dp|={np.abs(pl-pq).max():.2e} "
      f"TVD={0.5*np.abs(pl-pq).sum():.2e} effbond={c.psi.max_bond()}", flush=True)

for n in (20, 26, 30):
    t0=time.time(); c=build(n); tb=time.time()-t0
    mb=c.psi.max_bond()
    t0=time.time(); _=list(c.sample(256, seed=0)); ts=time.time()-t0
    print(f"n={n}: build {tb:6.2f}s  eff bond={mb:5d}  sample256 {ts:6.2f}s  "
          f"-> ~{tb+ts/256*2048:6.1f}s/eval (build+2048 samples)", flush=True)
