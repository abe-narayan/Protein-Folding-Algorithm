"""Characterize the OBC2 GB blowups on 8-state chignolin (default STATES_8):
cause, frequency, and — critically — whether they sit in the CVaR LOW-tail (the
lowest energies CVaR selects) and whether the sane energy minimum is near-native.
Fast random-sampling preview; no VQE."""
import os, sys, time
import numpy as np
import openmm
from openmm import app, unit
import amber_hamiltonian as amb
import representations as reps
import protein_geometry as geo
print("interp:", sys.executable, flush=True); assert "WindowsApps" in sys.executable

_orig=app.ForceField
def _obc(*a): return _orig(*tuple("implicit/obc2.xml" if x=="implicit/gbn2.xml" else x for x in a))
def build(seq, nstates):
    rep=reps.TorsionStateRepresentation(len(seq), nstates)
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

def min_ca(coords):
    ca=coords["CA"];n=len(ca);b=1e9
    for i in range(n):
        for j in range(i+2,n): b=min(b,float(np.linalg.norm(ca[i]-ca[j])))
    return b

seq="GYDPETGTWG"; H,rep=build(seq,8)
_,nc,nphi,npsi=geo.native_coords_from_pdb(os.path.join(os.path.dirname(amb.__file__),"pdbs","1UAO.pdb"))
nat_ca=np.asarray(nc["CA"]); rmsd=lambda b: geo.ca_rmsd(rep.build_coords(b)["CA"],nat_ca)
e_native=H.energy_from_coords(nc,nphi,npsi)
print(f"native (real backbone) OBC2 energy = {e_native:.1f} kcal  ceiling 0.919 A\n", flush=True)

N=2500; rng=np.random.default_rng(0)
bits=[rep.random_bitstring(rng) for _ in range(N)]
E=np.array([H.energy(b) for b in bits])
comp_solv=[]  # for a few extremes
order=np.argsort(E)
print(f"energy over {N} random 8-state structures:")
for q in (0,1,5,25,50):
    print(f"  {q:>2}th pct: {np.percentile(E,q):9.1f}")
print(f"  median {np.median(E):.1f}  max {E.max():.1f}")
for thr in (-800,-600,-500,-400, e_native):
    print(f"  count < {thr:8.1f}: {(E<thr).sum():4d}  ({100*(E<thr).mean():.1f}%)")

print("\nLOWEST-15 by energy (what CVaR's low-tail and the answer select):")
print(f"  {'E':>10}{'solv':>10}{'minCACA':>9}{'RMSD':>8}  bits")
for k in order[:15]:
    cm=H.components(bits[k]); mc=min_ca(rep.build_coords(bits[k]))
    print(f"  {E[k]:10.1f}{cm['solvation']:10.1f}{mc:9.2f}{rmsd(bits[k]):8.2f}  {bits[k]}", flush=True)

# lowest SANE (non-blowup, E>-600) structures -> what the energy points to without blowups
sane=[k for k in order if E[k]>-600][:10]
print("\nLOWEST-10 SANE (E>-600) structures:")
print(f"  {'E':>10}{'minCACA':>9}{'RMSD':>8}")
for k in sane:
    print(f"  {E[k]:10.1f}{min_ca(rep.build_coords(bits[k])):9.2f}{rmsd(bits[k]):8.2f}", flush=True)
print(f"\nbest RMSD among all {N}: {min(rmsd(b) for b in bits):.2f} A  "
      f"| best RMSD among sane: {min(rmsd(bits[k]) for k in order if E[k]>-600):.2f} A")
