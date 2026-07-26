"""Isolate whether the GB blowups come from _replace_customgb_with_native.
Score the SAME structures (as coordinates, 50-step minimized) under 3 GB paths:
  (a) OBC2 native GBSAOBCForce  (the swap, current fast tier)
  (b) OBC2 interpreted CustomGBForce (no swap)
  (c) GBn2 CustomGBForce (original live-source default)
Report GB solvation + total for each. Change nothing (scratch only)."""
import os, sys
import numpy as np
import openmm
from openmm import app, unit
import amber_hamiltonian as amb
import representations as reps
import protein_geometry as geo
import vqe as real_vqe
print("interp:", sys.executable, flush=True); assert "WindowsApps" in sys.executable

_orig=app.ForceField
def _obc(*a): return _orig(*tuple("implicit/obc2.xml" if x=="implicit/gbn2.xml" else x for x in a))

def build(model):
    rep=reps.TorsionStateRepresentation(10,4)   # rep irrelevant for coord scoring
    if model=="gbn2":
        H=amb.AmberHamiltonian("GYDPETGTWG",rep)          # live default
        return H
    amb.app.ForceField=_obc
    try: H=amb.AmberHamiltonian("GYDPETGTWG",rep)          # obc2 interpreted
    finally: amb.app.ForceField=_orig
    if model=="obc2_native":                               # + native swap
        for i,f in enumerate(H.system.getForces()):
            if isinstance(f,openmm.CustomGBForce): cgb,gi=f,i;break
        nb=openmm.GBSAOBCForce();nb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
        nb.setSolventDielectric(78.5);nb.setSoluteDielectric(1.0);nb.setSurfaceAreaEnergy(2.25936)
        for p in range(cgb.getNumParticles()):
            c,or_,sr=cgb.getParticleParameters(p);nb.addParticle(c,or_+0.009,sr/or_)
        nb.setForceGroup(5);H.system.removeForce(gi);H.system.addForce(nb)
        H.context=openmm.Context(H.system,openmm.VerletIntegrator(0.001*unit.picoseconds),H.platform,H.platform_properties)
    return H

Ha=build("obc2_native"); Hb=build("obc2_interp"); Hc=build("gbn2")
rep4=reps.TorsionStateRepresentation(10,4); rep8=reps.TorsionStateRepresentation(10,8)
_,nc,nphi,npsi=geo.native_coords_from_pdb(os.path.join(os.path.dirname(amb.__file__),"pdbs","1UAO.pdb"))
nat_ca=np.asarray(nc["CA"]); rmsd=lambda cd: geo.ca_rmsd(cd["CA"],nat_ca)

# capture a negative-blowup structure from a quick VQE on path (a)
print("capturing a blowup structure from a short lightning VQE on path (a)...", flush=True)
resa=real_vqe.run_global_cvar_vqe(Ha, layers=2, alpha=0.15, shots=512, maxiter=50, restarts=2, seed=0)
blow_bits=resa["best_seen_bitstring"]
print(f"  best_seen under (a): {resa['best_seen_energy']:.1f} kcal  bits={blow_bits}", flush=True)

# build the structure set (as coords)
structs=[("native", nc, None, None)]
structs.append(("pred_4state(4.96A)", rep4.build_coords("10101000000000111010"), None, None))
structs.append(("near-native_8st(2.45A)", rep8.build_coords("010001011100000000111111101100"), None, None))
structs.append(("BLOWUP(a)", rep4.build_coords(blow_bits), None, None))
rng=np.random.default_rng(1)
for i in range(12):
    structs.append((f"rand4_{i}", rep4.build_coords(rep4.random_bitstring(rng)), None, None))

def score(H, coords, phi=None, psi=None):
    e,comp,_=H._evaluate(H._heavy_positions(coords), want_components=True)
    return comp["solvation"], e

print(f"\n{'structure':>22} {'RMSD':>6} | {'(a)solv':>10}{'(b)solv':>10}{'(c)solv':>10} | "
      f"{'(a)tot':>10}{'(b)tot':>10}{'(c)tot':>10}")
for name, cd, ph, ps in structs:
    sa,ta=score(Ha,cd); sb,tb=score(Hb,cd); sc,tc=score(Hc,cd)
    r = rmsd(cd) if name!="native" else 0.0
    print(f"{name:>22} {r:6.2f} | {sa:10.1f}{sb:10.1f}{sc:10.1f} | {ta:10.1f}{tb:10.1f}{tc:10.1f}", flush=True)
