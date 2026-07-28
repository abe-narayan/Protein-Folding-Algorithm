"""PROFILE 5: trpzip per-energy OpenMM cost + realistic MPS per-eval, for scaling.
Seq SWTWEGNKWTWK (12 res, 36q). Golden = trpzip_seed0_full_eval_set.csv."""
import os, sys, time, csv, math, json
import numpy as np
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repo")
sys.path.insert(0, REPO)
import representations as reps
from amber_obc2 import build_obc2_native
from mps.sampler import MPSSampler

GOLDEN = r"C:\Users\abena\AppData\Local\Temp\claude\C--Abhyuth-claude-protein-build\84b6b99e-497b-494f-8516-ca4c535644c6\scratchpad\trpzip_seed0_full_eval_set.csv"
SEQ = "SWTWEGNKWTWK"
N_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 400
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out")

def main():
    os.makedirs(OUT, exist_ok=True)
    rep = reps.TorsionStateRepresentation(len(SEQ), n_states=8)
    H = build_obc2_native(SEQ, rep)
    d = H.describe()
    print("SETUP", {k: d[k] for k in ("n_atoms","n_hydrogens")}, "n_qubits", H.n_qubits, flush=True)
    gold = {}
    with open(GOLDEN, newline="") as f:
        r = csv.reader(f); next(r)
        for row in r:
            try: gold[row[0]] = float(row[1])
            except Exception: gold[row[0]] = float("inf")
    keys = list(gold.keys())
    rng = np.random.default_rng(777)
    sk = [keys[i] for i in rng.choice(len(keys), size=min(N_EVAL,len(keys)), replace=False)]
    H.reset_counters()
    for a in ("_t_setpos","_t_restraint","_t_minim_only"): setattr(H,a,0.0)
    per = np.empty(len(sk)); mism = 0
    for i,bs in enumerate(sk):
        c0=time.perf_counter(); e=H.energy(bs); per[i]=time.perf_counter()-c0
        g=gold[bs]
        if math.isfinite(g) and math.isfinite(e) and abs(e-g)>1e-6: mism+=1
        elif math.isfinite(g)!=math.isfinite(e): mism+=1
    n=len(sk)
    res={"seq":SEQ,"n_atoms":d["n_atoms"],"n_qubits":H.n_qubits,"n_eval":n,
         "per_call_ms_mean":float(per.mean()*1e3),"per_call_ms_median":float(np.median(per)*1e3),
         "split_ms":{"assemble":H.t_build/n*1e3,"setpos":H._t_setpos/n*1e3,
                     "restraint":H._t_restraint/n*1e3,"minimize":H._t_minim_only/n*1e3,
                     "getstate":H.t_energy/n*1e3},
         "golden_mismatches":mism,"golden_checked":sum(1 for b in sk if math.isfinite(gold[b]))}
    # realistic MPS per-eval at true init params (init_scale 0.25) and one spread set
    samp=MPSSampler(H.n_qubits,layers=4); npar=4*H.n_qubits
    init_rng=np.random.default_rng(np.random.SeedSequence([0,0xC0FFEE]))
    p_init=(math.pi/2)+init_rng.normal(0,0.25,size=npar)
    def tmin(p,sh=1024):
        ts=[]
        for _ in range(2):
            rr=np.random.default_rng(7); c0=time.perf_counter(); samp.draw_indices(p,sh,rr); ts.append(time.perf_counter()-c0)
        return min(ts)
    res["mps_per_eval_s_init025"]=tmin(p_init)
    res["mps_bond_init025"]=int(samp.bond_dim(p_init))
    print(json.dumps(res,indent=2),flush=True)
    with open(os.path.join(OUT,"prof5_trpzip.json"),"w") as f: json.dump(res,f,indent=2)
    print("WROTE prof5_trpzip.json",flush=True)

if __name__=="__main__": main()
