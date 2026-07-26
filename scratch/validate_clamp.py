"""Diagnostic: is the best_seen/CVaR disagreement the OBC2 GB blowup pathology
(not an MPS defect)? Re-run (C) with energies clamped at a floor (blowups removed).
If clamped best_seen/CVaR agree within noise, the disagreement is 100% the blowup.
Clamp is DIAGNOSTIC ONLY (scratch) — the production energy/protocol is untouched."""
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

class ClampH:
    """Delegates to H but clamps energies at a floor (removes GB blowups). Diagnostic."""
    def __init__(self, H, floor=-800.0): self._H=H; self._floor=floor
    def __getattr__(self,k): return getattr(self._H,k)
    def energy(self,bs): return max(self._H.energy(bs), self._floor)

seq="GYDPETGTWG"; H,rep=build(seq)
_,nc,_,_=geo.native_coords_from_pdb(os.path.join(os.path.dirname(amb.__file__),"pdbs","1UAO.pdb"))
nat_ca=np.asarray(nc["CA"]); rmsd=lambda b: geo.ca_rmsd(rep.build_coords(b)["CA"],nat_ca)
CFG=dict(layers=2, alpha=0.15, shots=512, maxiter=50, restarts=2, seed=0)

Hc=ClampH(H, floor=-800.0)
H._cache.clear(); rl=real_vqe.run_global_cvar_vqe(Hc, **CFG)
H._cache.clear(); rm=mps_vqe.run_global_cvar_vqe(Hc, sampler=MPSSampler(H.n_qubits,CFG["layers"]), **CFG)
print("\nCLAMPED (floor -800, blowups removed) — MPS vs lightning:")
print(f"  best_seen : lightning {rl['best_seen_energy']:.3f} | MPS {rm['best_seen_energy']:.3f} "
      f"(d={abs(rl['best_seen_energy']-rm['best_seen_energy']):.3f})")
print(f"  vqe_energy: lightning {rl['vqe_energy']:.3f} | MPS {rm['vqe_energy']:.3f} "
      f"(d={abs(rl['vqe_energy']-rm['vqe_energy']):.3f})")
print(f"  final CVaR: lightning {rl['final_objective']:.3f} | MPS {rm['final_objective']:.3f} "
      f"(d={abs(rl['final_objective']-rm['final_objective']):.3f})")
print(f"  vqe RMSD  : lightning {rmsd(rl['vqe_bitstring']):.3f} | MPS {rmsd(rm['vqe_bitstring']):.3f} A")
print(f"  best RMSD : lightning {rmsd(rl['best_seen_bitstring']):.3f} | MPS {rmsd(rm['best_seen_bitstring']):.3f} A")
