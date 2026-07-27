"""Build an :class:`amber_hamiltonian.AmberHamiltonian` on the OBC2 implicit
solvent using OpenMM's native :class:`GBSAOBCForce` instead of the ff14SB default
(``implicit/gbn2.xml`` -> ``CustomGBForce``).

The native GBSAOBCForce is ~6-7x faster than the CustomGBForce path and is what the
STATES_8 chignolin seed-0 evaluation set was generated with. This is the single
source of truth for that construction (previously copy-pasted into several
``scratch/`` scripts).

The construction is deliberately verbatim so its energies stay byte-identical to the
persisted seed-0 data:

1. Temporarily swap ``implicit/gbn2.xml`` -> ``implicit/obc2.xml`` while the
   AmberHamiltonian builds its ForceField (monkeypatch, restored in ``finally``).
2. Replace the resulting ``CustomGBForce`` with a ``GBSAOBCForce`` carrying the same
   per-particle charge/radius/scale (with the standard +0.009 nm radius offset and
   scale = scaledRadius / offsetRadius), force group 5, NoCutoff.
3. Rebuild the OpenMM ``Context`` so the swapped force is live.
"""
import openmm
from openmm import app, unit

import amber_hamiltonian as amb

__all__ = ["build_obc2_native"]


def build_obc2_native(sequence: str, representation) -> "amb.AmberHamiltonian":
    """AmberHamiltonian for ``sequence`` with the native GBSAOBCForce (OBC2)."""
    orig_forcefield = app.ForceField

    def obc2_forcefield(*args):
        return orig_forcefield(*tuple(
            "implicit/obc2.xml" if x == "implicit/gbn2.xml" else x for x in args))

    amb.app.ForceField = obc2_forcefield
    try:
        H = amb.AmberHamiltonian(sequence, representation)
    finally:
        amb.app.ForceField = orig_forcefield

    custom_gb, gb_index = None, None
    for i, force in enumerate(H.system.getForces()):
        if isinstance(force, openmm.CustomGBForce):
            custom_gb, gb_index = force, i
            break

    native = openmm.GBSAOBCForce()
    native.setNonbondedMethod(openmm.GBSAOBCForce.NoCutoff)
    native.setSolventDielectric(78.5)
    native.setSoluteDielectric(1.0)
    native.setSurfaceAreaEnergy(2.25936)
    for p in range(custom_gb.getNumParticles()):
        charge, offset_radius, scaled_radius = custom_gb.getParticleParameters(p)
        native.addParticle(charge, offset_radius + 0.009,
                           scaled_radius / offset_radius)
    native.setForceGroup(5)
    H.system.removeForce(gb_index)
    H.system.addForce(native)

    H.context = openmm.Context(
        H.system, openmm.VerletIntegrator(0.001 * unit.picoseconds),
        H.platform, H.platform_properties)
    return H
