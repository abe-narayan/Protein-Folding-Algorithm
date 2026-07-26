"""8-state chignolin VQE, seed 0, config A (layers4/maxiter200/restarts3/shots1024),
default STATES_8, MPS backend, OBC2-native tier. Collapse audit is pure observation
(flag total < -450 in CVaR low-tail). Reports selected energy, audit, CA-RMSD vs
4.80 (4-state) and vs 8-state ceiling, and native-vs-prediction energy gap."""
import os, sys, time, json
import numpy as np
import openmm
from openmm import app, unit
import amber_hamiltonian as amb
import representations as reps
import protein_geometry as geo
import mps_vqe
from mps_sampler import MPSSampler
print("interp:", sys.executable, flush=True); assert "WindowsApps" in sys.executable

_orig=app.ForceField
def _obc(*a): return _orig(*tuple("implicit/obc2.xml" if x=="implicit/gbn2.xml" else x for x in a))
def build_native(seq, rep):
    amb.app.ForceField=_obc
    try: H=amb.AmberHamiltonian(seq,rep)
    finally: amb.app.ForceField=_orig
    for i,f in enumerate(H.system.getForces()):
        if isinstance(f,openmm.CustomGBForce): cgb,gi=f,i;break
    nb=openmm.GBSAOBCForce();nb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
    nb.setSolventDielectric(78.5);nb.setSoluteDielectric(1.0);nb.setSurfaceAreaEnergy(2.25936)
    for p in range(cgb.getNumParticles()):
        c,or_,sr=cgb.getParticleParameters(p);nb.addParticle(c,or_+0.009,sr/or_)
    nb.setForceGroup(5);H.system.removeForce(gi);H.system.addForce(nb)
    H.context=openmm.Context(H.system,openmm.VerletIntegrator(0.001*unit.picoseconds),H.platform,H.platform_properties)
    return H

seq="GYDPETGTWG"
rep=reps.TorsionStateRepresentation(len(seq), n_states=8)   # default STATES_8
H=build_native(seq, rep)
_,nc,nphi,npsi=geo.native_coords_from_pdb(os.path.join(os.path.dirname(amb.__file__),"pdbs","1UAO.pdb"))
nat_ca=np.asarray(nc["CA"]); rmsd=lambda b: geo.ca_rmsd(rep.build_coords(b)["CA"], nat_ca)

sampler=MPSSampler(H.n_qubits, layers=4)
print(f"n_qubits={H.n_qubits}  default STATES_8  MPS backend  config A", flush=True)
t0=time.time()
res=mps_vqe.run_global_cvar_vqe(H, sampler=sampler, layers=4, alpha=0.15, shots=1024,
                                maxiter=200, restarts=3, seed=0, final_shots=8192,
                                init_scale=0.25, verbose=True)
wall=time.time()-t0
a=res["audit"]

print("\n==================== 8-STATE CHIGNOLIN, SEED 0 (config A) ====================")
print(f"  wall {wall/3600:.2f} h | evals {res['n_objective_evals_total']} | "
      f"unique {res['n_unique_structures_cached']}")
print(f"\n  --- COLLAPSE AUDIT (flag total < -450 in CVaR low-tail) ---")
print(f"  (i)   selected vqe_bitstring energy : {a['final_selected_energy']:.2f} kcal  "
      f"-> {'PHYSICAL (-350..-390)' if -390<=a['final_selected_energy']<=-350 else 'OUT OF RANGE'}"
      f"{'  *** COLLAPSE ***' if a['final_selected_is_collapse'] else ''}")
print(f"  (ii)  evals whose low-tail had a collapse: {a['n_evals_lowtail_collapse_best_run']}"
      f"/{a['n_evals_best_run']} (best run) | {a['n_evals_lowtail_collapse_all']} (all runs)")
print(f"  (iii) final selected tail collapse-free : "
      f"{'YES' if a['final_sample_n_collapse']==0 else 'NO ('+str(a['final_sample_n_collapse'])+' collapses)'}"
      f"  | final-sample lowest-5 energies: {[round(x,1) for x in a['final_sample_lowest5']]}")

clean = (not a['final_selected_is_collapse']) and (a['final_sample_n_collapse']==0) \
        and (-390 <= a['final_selected_energy'] <= -350)
print(f"\n  AUDIT VERDICT: {'CLEAN' if clean else 'CONTAMINATED -> STOP, clamp at -450, re-run'}")

if clean:
    r=rmsd(res['vqe_bitstring'])
    print(f"\n  --- RESULT ---")
    print(f"  CA-RMSD (vqe_bitstring) : {r:.2f} A")
    print(f"     vs 4-state 4.80 A    : {'improved' if r<4.80 else 'worse'} by {4.80-r:+.2f} A")
    print(f"     vs 8-state ceiling 0.919 A : +{r-0.919:.2f} A above ceiling")
    e_nat_real=H.energy_from_coords(nc,nphi,npsi)
    e_nat_snap8=H.energy(rep.angles_to_bits(nphi,npsi))
    print(f"\n  --- native-vs-prediction energy gap at 8 states ---")
    print(f"  native (real backbone)   {e_nat_real:.2f}")
    print(f"  native (snapped 8-state) {e_nat_snap8:.2f}")
    print(f"  prediction (vqe)         {res['vqe_energy']:.2f}")
    print(f"  gap real-native - pred     : {e_nat_real-res['vqe_energy']:+.2f} kcal")
    print(f"  gap snap8-native - pred    : {e_nat_snap8-res['vqe_energy']:+.2f} kcal  (4-state snap was ~+33)")
print(f"\n  best_seen (blowup-unreliable): {res['best_seen_energy']:.1f}  "
      f"RMSD {rmsd(res['best_seen_bitstring']):.2f} A")
print("  vqe_bitstring:", res['vqe_bitstring'], flush=True)
