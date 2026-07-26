"""Validation A+B: state exactness (MPS probs vs double-precision statevector) and
sampler distribution match, at n=8 (bit-exact), n=20, n=24."""
import sys, time
import numpy as np
import pennylane as qml
from mps_sampler import MPSSampler
LAYERS=4
def double_probs(n, params):
    dev=qml.device("default.qubit",wires=n)
    @qml.qnode(dev)
    def st(pp):
        pp=pp.reshape(LAYERS,n)
        for l in range(LAYERS):
            for q in range(n): qml.RY(float(pp[l][q]),wires=q)
            for q in range(n-1): qml.CNOT(wires=[q,q+1])
            qml.CNOT(wires=[n-1,0])
        return qml.probs(wires=range(n))
    return np.asarray(st(params))

print("interp:", sys.executable, flush=True); assert "WindowsApps" in sys.executable
for n in (8, 20, 24):
    params=(np.pi/2)+np.random.default_rng(0).normal(0,0.25,size=LAYERS*n)
    pex=double_probs(n, params)
    smp=MPSSampler(n, LAYERS)
    pmps=smp.probs(params)
    maxdp=np.abs(pex-pmps).max(); tvd=0.5*np.abs(pex-pmps).sum()
    print(f"\nn={n}: STATE exactness  max|dp|={maxdp:.2e}  TVD={tvd:.2e}  "
          f"argmax_match={int(np.argmax(pex))==int(np.argmax(pmps))}  bond={smp.bond_dim(params)}",flush=True)
    N=100000 if n==8 else 30000
    rng=np.random.default_rng(123)
    t0=time.time(); idx=smp.draw_indices(params, N, rng); ts=time.time()-t0
    emp=np.bincount(idx, minlength=pex.size).astype(float)/N
    stvd=0.5*np.abs(emp-pex).sum()
    supp=int((pex>1e-10).sum())
    print(f"      SAMPLER dist match ({N} shots, {ts:.1f}s): TVD={stvd:.4f} "
          f"expected_noise~{np.sqrt(supp/N):.4f}  effective_support={supp}",flush=True)
    # reproducibility: same seed -> identical draws
    a=smp.draw_indices(params, 256, np.random.default_rng(7))
    b=smp.draw_indices(params, 256, np.random.default_rng(7))
    print(f"      seed-reproducible draws: {np.array_equal(a,b)}",flush=True)
