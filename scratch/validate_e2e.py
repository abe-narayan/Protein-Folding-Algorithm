"""Validation C+D+E: end-to-end CVaR-VQE equivalence on chignolin 4-state (n=20).
(E) mps_vqe lightning-path == real vqe (byte-identical refactor, req #4).
(C) mps_vqe MPS-path vs lightning: same energies/argmin, CVaR & RMSD within noise (#2).
(D) determinism: MPS path twice, same seed -> identical (#6)."""
import os, sys
import numpy as np
import openmm
from openmm import app, unit
import amber_hamiltonian as amb
import representations as reps
import protein_geometry as geo
import vqe as real_vqe
import mps_vqe
from mps_sampler import MPSSampler
print("interp:", sys.executable, flush=True); assert "WindowsApps" in sys.executable

_orig=app.ForceField
def _obc(*a): return _orig(*tuple("implicit/obc2.xml" if x=="implicit/gbn2.xml" else x for x in a))
def build(seq):
    rep=reps.TorsionStateRepresentation(len(seq),4)
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
    return H,rep

seq="GYDPETGTWG"; H,rep=build(seq)
_,nc,_,_=geo.native_coords_from_pdb(os.path.join(os.path.dirname(amb.__file__),"pdbs","1UAO.pdb"))
nat_ca=np.asarray(nc["CA"])
def rmsd(bits): return geo.ca_rmsd(rep.build_coords(bits)["CA"], nat_ca)
CFG=dict(layers=2, alpha=0.15, shots=512, maxiter=50, restarts=2, seed=0)

print("\n(E) byte-identical refactor: mps_vqe(lightning) vs real vqe")
H._cache.clear(); r_real=real_vqe.run_global_cvar_vqe(H, **CFG)
H._cache.clear(); r_lite=mps_vqe.run_global_cvar_vqe(H, sampler=None, **CFG)
for k in ("best_seen_energy","vqe_energy","final_objective"):
    print(f"    {k}: real={r_real[k]:.6f} refactor={r_lite[k]:.6f} identical={r_real[k]==r_lite[k]}")

print("\n(C) MPS vs lightning (same exact distribution, different RNG -> within noise)")
smp=MPSSampler(H.n_qubits, CFG["layers"])
H._cache.clear(); r_mps=mps_vqe.run_global_cvar_vqe(H, sampler=smp, **CFG)
print(f"    best_seen: lightning {r_real['best_seen_energy']:.3f} | MPS {r_mps['best_seen_energy']:.3f} "
      f"(d={abs(r_real['best_seen_energy']-r_mps['best_seen_energy']):.3f})")
print(f"    vqe_energy: lightning {r_real['vqe_energy']:.3f} | MPS {r_mps['vqe_energy']:.3f}")
print(f"    final CVaR: lightning {r_real['final_objective']:.3f} | MPS {r_mps['final_objective']:.3f}")
print(f"    vqe RMSD:  lightning {rmsd(r_real['vqe_bitstring']):.3f} | MPS {rmsd(r_mps['vqe_bitstring']):.3f} A")
print(f"    best RMSD: lightning {rmsd(r_real['best_seen_bitstring']):.3f} | MPS {rmsd(r_mps['best_seen_bitstring']):.3f} A")

print("\n(D) determinism: MPS path twice, same seed")
H._cache.clear(); a=mps_vqe.run_global_cvar_vqe(H, sampler=MPSSampler(H.n_qubits,CFG["layers"]), **CFG)
H._cache.clear(); b=mps_vqe.run_global_cvar_vqe(H, sampler=MPSSampler(H.n_qubits,CFG["layers"]), **CFG)
print(f"    best_seen identical={a['best_seen_energy']==b['best_seen_energy']} "
      f"vqe_energy identical={a['vqe_energy']==b['vqe_energy']} "
      f"spread={abs(a['best_seen_energy']-b['best_seen_energy']):.2e}")
