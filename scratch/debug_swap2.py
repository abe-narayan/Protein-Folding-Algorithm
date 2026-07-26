"""Full component breakdown of the blowup structure (bits 10110000010000101010)
under the 3 GB paths, to confirm the divergence is NOT the GB solvation term."""
import os, sys, numpy as np, openmm
from openmm import app, unit
import amber_hamiltonian as amb
import representations as reps
print("interp:", sys.executable, flush=True); assert "WindowsApps" in sys.executable
_orig=app.ForceField
def _obc(*a): return _orig(*tuple("implicit/obc2.xml" if x=="implicit/gbn2.xml" else x for x in a))
def build(model):
    rep=reps.TorsionStateRepresentation(10,4)
    if model=="gbn2": return amb.AmberHamiltonian("GYDPETGTWG",rep)
    amb.app.ForceField=_obc
    try: H=amb.AmberHamiltonian("GYDPETGTWG",rep)
    finally: amb.app.ForceField=_orig
    if model=="obc2_native":
        for i,f in enumerate(H.system.getForces()):
            if isinstance(f,openmm.CustomGBForce): cgb,gi=f,i;break
        nb=openmm.GBSAOBCForce();nb.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
        nb.setSolventDielectric(78.5);nb.setSoluteDielectric(1.0);nb.setSurfaceAreaEnergy(2.25936)
        for p in range(cgb.getNumParticles()):
            c,or_,sr=cgb.getParticleParameters(p);nb.addParticle(c,or_+0.009,sr/or_)
        nb.setForceGroup(5);H.system.removeForce(gi);H.system.addForce(nb)
        H.context=openmm.Context(H.system,openmm.VerletIntegrator(0.001*unit.picoseconds),H.platform,H.platform_properties)
    return H
rep4=reps.TorsionStateRepresentation(10,4)
cd=rep4.build_coords("10110000010000101010")
TERMS=("bond","angle","torsion","nonbonded","solvation","total")
print(f"\nblowup structure components:  {'':>10}"+"".join(f"{t:>12}" for t in TERMS))
for name in ("obc2_native","obc2_interp","gbn2"):
    H=build(name)
    e,comp,_=H._evaluate(H._heavy_positions(cd), want_components=True); comp=dict(comp); comp["total"]=e
    print(f"  {name:>18}"+"".join(f"{comp[t]:12.1f}" for t in TERMS), flush=True)
