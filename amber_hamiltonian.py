"""Amber ff14SB + GBn2 folding Hamiltonian.

Drop-in replacement for `hamiltonian.FoldingHamiltonian`. Same interface:

    .energy(bitstring)       .energy_from_coords(coords, phi, psi)
    .components(bitstring)   .n_qubits  .n_bits  .rep  .sequence  .weights
    .reset_counters()        .cache_size()       .n_energy_evaluations

WHY THIS EXISTS
---------------
The 7-term hand-weighted energy in energy_terms.py ranks native chignolin at
+5.51 while returning structures at -2.17: the native is scored WORSE than
the prediction, so the search cannot help. VQE, simulated annealing and random
search all land within 0.25 A of each other, which says the search is solved
and the energy is wrong. This module replaces the energy outright with a real
molecular-mechanics force field. Nothing else in the pipeline changes.

E(x) = ff14SB bonded + nonbonded + GBn2 implicit solvation, in kcal/mol,
evaluated after a capped, backbone-restrained minimization of the structure
that `rep` decodes from x.

LEAKAGE GUARANTEE (unchanged from FoldingHamiltonian): the constructor takes
a sequence, a representation and numerical settings. No PDB path, no native
coordinates, no code path by which a native structure can influence E(x).
`energy_from_coords` is an EVALUATION path used to score the native for the
energy-gap metric, exactly as before.

--------------------------------------------------------------------------
HYDROGEN DETERMINISM -- the central design problem
--------------------------------------------------------------------------
`Modeller.addHydrogens` literally places hydrogens at random positions:

    delta = Vec3(random.random(), random.random(), random.random())*nanometer

and then relaxes them. Day 1 measured 48.9 kcal/mol of spread across four
identical runs on trpzip, with hydrogen positions differing by up to 1.75 A.
`random.seed(0)` reduces this but does not remove it, because addHydrogens'
own relaxation runs on the DEFAULT platform, i.e. multithreaded CPU, whose
force summation order is not reproducible.

`test_energy_is_finite_and_deterministic` requires |e1 - e2| < 1e-12, so
hydrogens must be a pure function of the heavy atoms.

CHOSEN STRATEGY: frozen local frames -- the second of the two options, i.e.
minimize hydrogens ONCE and thereafter transform them rigidly with their
parent heavy atoms.

  * Topology, System and Context are built once per sequence in __init__.
    Hydrogens are added exactly once, there.
  * For every hydrogen h with heavy parent P, a rigid orthonormal frame
    (P; A, B) is chosen from heavy atoms whose geometry relative to P does
    NOT depend on any torsion the bitstring can change. h's displacement is
    stored as three fixed coefficients in that frame.
  * Per evaluation: rebuild heavy atoms, rebuild each frame from the new
    heavy positions, replay the stored coefficients, setPositions, minimize,
    read the energy. No hydrogen is ever re-placed by Modeller.

WHY FRAMES RATHER THAN GEOMETRIC PLACEMENT. Geometric placement would mean
hand-coding H templates for every parent environment -- sp3 CH/CH2/CH3, sp2
aromatic CH, amide NH, NH3+ and OH rotors -- and those hand-coded internal
coordinates would not be ff14SB's own equilibrium geometry, so every
structure would start with a spurious hydrogen strain that the capped
minimization has to spend its 50 steps removing. The frame approach reuses
the force field's own relaxed hydrogen geometry. It is also EXACT rather than
approximate here, because sidechains.py pins every chi angle: with chi fixed,
each hydrogen's parent frame is genuinely rigid, so a rigid transform
reproduces the reference geometry to machine precision.

The frames are rigid by case:
  * amide N of residue i>0: frame (N_i; CA_i, C_{i-1}). The peptide unit is
    planar and omega is fixed trans, so H is rigid in it. Using an in-residue
    third atom instead would make H depend on phi, which is WRONG.
  * every other parent: frame atoms are taken from the SAME residue, never
    the backbone O or OXT (O rotates with psi about the CA-C axis, so it is
    not rigid relative to N). Everything else inside a residue -- N, CA, C,
    CB and the whole fixed-chi sidechain -- is mutually rigid.

REFERENCE HYDROGEN GEOMETRY is itself made reproducible across processes by
seeding `random` immediately before addHydrogens AND forcing addHydrogens'
internal relaxation onto the Reference platform, which is single-threaded and
bit-deterministic. Measured: three builds bit-identical (max diff 0.0). With
the default platform instead, builds differ by 0.0025 A, which the capped
minimization amplifies into several kcal/mol.

--------------------------------------------------------------------------
PLATFORM DETERMINISM
--------------------------------------------------------------------------
'CPU' is requested explicitly, as required, to avoid a slow CUDA/OpenCL
fallback. It is configured with Threads=1 and DeterministicForces=true.
This is NOT optional: with default threading, repeated single-point energies
on IDENTICAL positions drift by 4e-5 kcal/mol, and 50 L-BFGS steps amplify
that to 0.13 kcal/mol -- eleven orders of magnitude above the 1e-12 the
determinism test demands. At 138-218 atoms one thread is also no slower.

--------------------------------------------------------------------------
CAPPED MINIMIZATION
--------------------------------------------------------------------------
Ideal-geometry structures with fixed chi rotamers start with severe clashes
(~1e8 kcal/mol for the all-helix chignolin), so a raw single-point energy is
meaningless. 50 L-BFGS steps are run with harmonic position restraints on
every backbone N, CA and C, so the bitstring -> structure mapping survives:
measured CA-RMSD between pre- and post-minimization coordinates, WITHOUT
superposition, is at most 0.21 A over both benchmark peptides.

25 steps was measured and REJECTED: it leaves the all-helix chignolin at
+59.1 kcal/mol, i.e. the clashes are not yet relieved, and max CA-RMSD rises
to 0.50 A. 50 is the value used.

The restraint is a CustomExternalForce whose strength is a global parameter.
It is set to `restraint_k` for the minimization and then to ZERO before the
energy is read, so the reported number is pure ff14SB + GBn2 with no
restraint contamination.
"""
import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import openmm
from openmm import app, unit

import protein_geometry as geo
import sidechains as sc


__all__ = ["AmberHamiltonian", "KCAL_PER_KJ", "AMBER_TERMS"]


KJ_PER_KCAL = 4.184
KCAL_PER_KJ = 1.0 / KJ_PER_KCAL

# force-group -> component name. LJ and Coulomb both live in NonbondedForce
# and are not separable without re-deriving the reciprocal sum, so they are
# reported together as "nonbonded" rather than split on a guess.
AMBER_TERMS = ("bond", "angle", "torsion", "nonbonded", "solvation")
_GROUP_OF = {"HarmonicBondForce": 1, "HarmonicAngleForce": 2,
             "PeriodicTorsionForce": 3, "CMAPTorsionForce": 3,
             "NonbondedForce": 4}
_TERM_GROUP = {"bond": 1, "angle": 2, "torsion": 3,
               "nonbonded": 4, "solvation": 5}
_RESTRAINT_GROUP = 6

RESTRAINED_BACKBONE = ("N", "CA", "C")
# never usable as a frame reference: O rotates with psi, OXT with it.
_NON_FRAME_ATOMS = ("O", "OXT")


class AmberHamiltonian:
    """ff14SB + GBn2 energy over a discrete conformational representation.

    Parameters
    ----------
    sequence, representation
        As for FoldingHamiltonian.
    weights
        Accepted for interface compatibility and echoed on `.weights`, which
        defaults to 1.0 for every Amber term. Amber terms are NOT free
        parameters -- reweighting them stops the result being ff14SB -- so
        this is here so that `main.py`'s breakdown printer keeps working, not
        as a tuning knob.
    restraint_k
        Backbone position restraint, kcal/mol/A^2, in E = 1/2 k r^2.
    minimization_steps
        L-BFGS iteration cap. 50; 25 is not enough (see module docstring).
    platform_name
        'CPU' explicitly, per the design constraint.
    reference_states
        Representation state indices for the conformation used to build the
        topology and calibrate hydrogens. Defaults to all-extended (state 1),
        which is the least clashed conformation available and therefore the
        cleanest one in which to relax reference hydrogens. The choice does
        not affect E(x): only the frame-local hydrogen coefficients are kept.
    """

    def __init__(self, sequence: str, representation,
                 weights: Optional[Dict[str, float]] = None,
                 restraint_k: float = 100.0,
                 minimization_steps: int = 50,
                 minimization_tolerance: float = 2.0,
                 cache_limit: int = 500_000,
                 platform_name: str = "CPU",
                 reference_states: Optional[Sequence[int]] = None):
        self.sequence = sequence.strip().upper()
        self.rep = representation
        if len(self.sequence) != self.rep.n_residues:
            raise ValueError(
                f"sequence length {len(self.sequence)} != representation "
                f"n_residues {self.rep.n_residues}")
        if getattr(self.rep, "is_lattice", False):
            raise ValueError(
                "AmberHamiltonian requires a full-backbone representation; "
                "the tetrahedral lattice provides CA only, which cannot be "
                "given an ff14SB topology. Use TorsionStateRepresentation.")

        self.restraint_k = float(restraint_k)
        self.minimization_steps = int(minimization_steps)
        self.minimization_tolerance = float(minimization_tolerance)

        self._cache: Dict[str, float] = {}
        self._cache_limit = int(cache_limit)
        self.n_energy_evaluations = 0   # cache MISSES only == real work
        self.t_build = 0.0              # heavy atoms + hydrogen replay
        self.t_minimize = 0.0           # restrained L-BFGS
        self.t_energy = 0.0             # setParameter + getState readout

        t0 = time.time()
        self._build_topology()
        self._build_system(platform_name)
        self._calibrate_hydrogens(reference_states)
        self.setup_time = time.time() - t0

        self.weights = {t: 1.0 for t in AMBER_TERMS}
        if weights:
            self.weights.update(weights)

    # -- properties ---------------------------------------------------------
    @property
    def n_qubits(self) -> int:
        return self.rep.n_qubits

    @property
    def n_bits(self) -> int:
        return self.rep.n_bits

    # ======================================================================
    # one-time construction
    # ======================================================================
    def _reference_backbone(self) -> Dict[str, np.ndarray]:
        """All-extended ideal backbone. Used only to seed the topology."""
        n = len(self.sequence)
        phi = np.full(n, math.radians(-120.0))
        psi = np.full(n, math.radians(130.0))
        phi[0] = geo.DEFAULT_PHI
        psi[n - 1] = geo.DEFAULT_PSI
        return geo.build_backbone(phi, psi)

    def _build_topology(self) -> None:
        """Topology + hydrogens, built ONCE. This is the only addHydrogens call."""
        seq = self.sequence
        ref = sc.build_full_structure(seq, self._reference_backbone())

        top = app.Topology()
        chain = top.addChain("A")
        self._heavy_names: List[Tuple[int, str]] = []
        positions: List[np.ndarray] = []
        prev_C = None
        for i, aa in enumerate(seq):
            rn = geo.ONE_TO_THREE[aa]
            res = top.addResidue(rn, chain)
            objs = {}
            for nm, xyz in ref["residues"][i].items():
                objs[nm] = top.addAtom(
                    nm, app.element.Element.getBySymbol(nm[0]), res)
                self._heavy_names.append((i, nm))
                positions.append(np.asarray(xyz, dtype=float) * 0.1)  # A -> nm
            for a_, b_ in sc.residue_bonds(rn):
                top.addBond(objs[a_], objs[b_])
            if "OXT" in objs:
                top.addBond(objs["C"], objs["OXT"])
            if prev_C is not None:
                top.addBond(prev_C, objs["N"])
            prev_C = objs["C"]

        self.forcefield = app.ForceField("amber14/protein.ff14SB.xml",
                                         "implicit/gbn2.xml")
        # Reference platform: single-threaded and bit-deterministic. Combined
        # with the seed this makes addHydrogens reproducible across processes.
        self._reference_platform = openmm.Platform.getPlatformByName("Reference")
        random.seed(0)
        modeller = app.Modeller(top, np.array(positions) * unit.nanometer)
        modeller.addHydrogens(self.forcefield,
                              platform=self._reference_platform)
        self.topology = modeller.topology
        self.n_atoms = self.topology.getNumAtoms()
        self._modeller_positions = np.array(
            modeller.positions.value_in_unit(unit.nanometer), dtype=float)

        self._index_topology()

    def _index_topology(self) -> None:
        """Name/index maps, heavy-atom adjacency, and hydrogen parentage."""
        self._elem = {a.index: (a.element.symbol if a.element else "X")
                      for a in self.topology.atoms()}
        self._resof = {a.index: a.residue.index for a in self.topology.atoms()}
        self._nameof = {a.index: a.name for a in self.topology.atoms()}

        self._heavy_index: Dict[Tuple[int, str], int] = {}
        self._hydrogens: List[int] = []
        for a in self.topology.atoms():
            if self._elem[a.index] == "H":
                self._hydrogens.append(a.index)
            else:
                self._heavy_index[(a.residue.index, a.name)] = a.index

        # position in the flat heavy array -> OpenMM atom index
        self._heavy_order = [self._heavy_index[k] for k in self._heavy_names]
        self._restraint_idx = [self._heavy_index[(i, nm)]
                               for i in range(len(self.sequence))
                               for nm in RESTRAINED_BACKBONE]

        adj: Dict[int, List[int]] = {}
        parent: Dict[int, int] = {}
        for bond in self.topology.bonds():
            i, j = bond[0].index, bond[1].index
            ei, ej = self._elem[i], self._elem[j]
            if ei == "H" and ej != "H":
                parent[i] = j
            elif ej == "H" and ei != "H":
                parent[j] = i
            elif ei != "H" and ej != "H":
                adj.setdefault(i, []).append(j)
                adj.setdefault(j, []).append(i)
        for k in adj:
            adj[k].sort()          # index order == deterministic frame choice
        self._adj = adj
        self._hparent = parent
        missing = [h for h in self._hydrogens if h not in parent]
        if missing:
            raise RuntimeError(
                f"{len(missing)} hydrogens have no heavy parent in the "
                "topology; cannot build deterministic frames")

    def _frame_atoms(self, p: int) -> Tuple[int, int]:
        """Two heavy atoms rigid with respect to `p`. See module docstring."""
        ri = self._resof[p]
        if self._nameof[p] == "N" and ri > 0:
            # peptide plane: H is rigid against (N_i, CA_i, C_{i-1}), and
            # NOT against any in-residue triple, because that depends on phi.
            return self._heavy_index[(ri, "CA")], self._heavy_index[(ri - 1, "C")]
        same = [q for q in self._adj.get(p, [])
                if self._resof[q] == ri and self._nameof[q] not in _NON_FRAME_ATOMS]
        if len(same) >= 2:
            return same[0], same[1]
        if len(same) == 1:
            a = same[0]
            second = [q for q in self._adj.get(a, [])
                      if q != p and self._resof[q] == ri
                      and self._nameof[q] not in _NON_FRAME_ATOMS]
            if second:
                return a, second[0]
        raise RuntimeError(
            f"no rigid frame for {self._nameof[p]} in residue {ri}")

    @staticmethod
    def _frame(p: np.ndarray, a: np.ndarray, b: np.ndarray):
        """Right-handed orthonormal frame at p: e1 towards a, e2 in the pab plane."""
        e1 = a - p
        e1 = e1 / np.linalg.norm(e1)
        v = b - p
        e2 = v - np.dot(v, e1) * e1
        e2 = e2 / np.linalg.norm(e2)
        return e1, e2, np.cross(e1, e2)

    def _build_system(self, platform_name: str) -> None:
        self.system = self.forcefield.createSystem(
            self.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,          # H bonds must be free to minimize
            rigidWater=False,
            removeCMMotion=False,
            implicitSolventKappa=0.0 / unit.nanometer)   # zero salt
        for f in self.system.getForces():
            f.setForceGroup(_GROUP_OF.get(f.__class__.__name__, 5))

        # E = 1/2 k r^2, k switched by a global parameter so the reported
        # energy can be read with the restraint contributing exactly zero.
        rest = openmm.CustomExternalForce(
            "0.5*k_rest*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        rest.addGlobalParameter("k_rest", 0.0)
        for nm in ("x0", "y0", "z0"):
            rest.addPerParticleParameter(nm)
        for idx in self._restraint_idx:
            rest.addParticle(idx, [0.0, 0.0, 0.0])
        rest.setForceGroup(_RESTRAINT_GROUP)
        self._rest_force = rest
        self.system.addForce(rest)
        # kcal/mol/A^2 -> kJ/mol/nm^2
        self._k_rest_internal = self.restraint_k * KJ_PER_KCAL * 100.0

        self.platform = openmm.Platform.getPlatformByName(platform_name)
        names = set(self.platform.getPropertyNames())
        props = {}
        if "Threads" in names:
            props["Threads"] = "1"
        if "DeterministicForces" in names:
            props["DeterministicForces"] = "true"
        self.platform_properties = props
        self.context = openmm.Context(
            self.system,
            openmm.VerletIntegrator(0.001 * unit.picoseconds),
            self.platform, props)

    def _calibrate_hydrogens(self, reference_states) -> None:
        """Relax hydrogens once against frozen heavy atoms, then freeze frames."""
        n = len(self.sequence)
        states = [1] * n if reference_states is None else list(reference_states)
        bits = self.rep.bitstring_from_states(states)
        heavy = self._heavy_positions(self.rep.build_coords(bits))

        pos = np.array(self._modeller_positions, dtype=float)
        for k, idx in enumerate(self._heavy_order):
            pos[idx] = heavy[k]

        # A scratch System with every heavy mass set to zero: OpenMM's
        # minimizer holds zero-mass particles fixed, so only hydrogens move.
        scratch = self.forcefield.createSystem(
            self.topology, nonbondedMethod=app.NoCutoff, constraints=None,
            rigidWater=False, removeCMMotion=False,
            implicitSolventKappa=0.0 / unit.nanometer)
        for i in range(self.n_atoms):
            if self._elem[i] != "H":
                scratch.setParticleMass(i, 0.0)
        ctx = openmm.Context(scratch, openmm.VerletIntegrator(0.001),
                             self.platform, self.platform_properties)
        ctx.setPositions(pos * unit.nanometer)
        openmm.LocalEnergyMinimizer.minimize(ctx, 1e-4, 2000)
        relaxed = np.array(
            ctx.getState(getPositions=True).getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer), dtype=float)
        del ctx

        # (hydrogen, parent, frame a, frame b, coefficients in that frame)
        self._h_local: List[Tuple[int, int, int, int, np.ndarray]] = []
        for h in self._hydrogens:
            p = self._hparent[h]
            a, b = self._frame_atoms(p)
            e1, e2, e3 = self._frame(relaxed[p], relaxed[a], relaxed[b])
            d = relaxed[h] - relaxed[p]
            self._h_local.append(
                (h, p, a, b,
                 np.array([np.dot(d, e1), np.dot(d, e2), np.dot(d, e3)])))

    # ======================================================================
    # per-evaluation geometry
    # ======================================================================
    def _heavy_positions(self, backbone: Dict[str, np.ndarray]) -> np.ndarray:
        """Backbone dict -> (n_heavy, 3) nanometre array in topology order."""
        full = sc.build_full_structure(self.sequence, backbone)
        out = np.empty((len(self._heavy_names), 3), dtype=float)
        for k, (i, nm) in enumerate(self._heavy_names):
            out[k] = full["residues"][i][nm]
        return out * 0.1

    def _assemble(self, heavy_nm: np.ndarray) -> np.ndarray:
        """Heavy atoms + frame-replayed hydrogens -> full (n_atoms, 3) in nm."""
        pos = np.empty((self.n_atoms, 3), dtype=float)
        for k, idx in enumerate(self._heavy_order):
            pos[idx] = heavy_nm[k]
        for h, p, a, b, c in self._h_local:
            e1, e2, e3 = self._frame(pos[p], pos[a], pos[b])
            pos[h] = pos[p] + c[0] * e1 + c[1] * e2 + c[2] * e3
        return pos

    def _evaluate(self, heavy_nm: np.ndarray, want_components: bool = False,
                  want_positions: bool = False):
        """The single energy path. Everything else routes through here."""
        t0 = time.time()
        pos = self._assemble(heavy_nm)
        self.t_build += time.time() - t0

        ctx = self.context
        t0 = time.time()
        ctx.setPositions(pos * unit.nanometer)
        for k, idx in enumerate(self._restraint_idx):
            self._rest_force.setParticleParameters(k, idx, pos[idx].tolist())
        self._rest_force.updateParametersInContext(ctx)
        ctx.setParameter("k_rest", self._k_rest_internal)
        openmm.LocalEnergyMinimizer.minimize(
            ctx, self.minimization_tolerance, self.minimization_steps)
        self.t_minimize += time.time() - t0

        t0 = time.time()
        ctx.setParameter("k_rest", 0.0)     # restraint contributes exactly 0
        state = ctx.getState(getEnergy=True, getPositions=want_positions)
        energy = state.getPotentialEnergy().value_in_unit(
            unit.kilocalorie_per_mole)
        comp = None
        if want_components:
            comp = {}
            for term in AMBER_TERMS:
                s = ctx.getState(getEnergy=True, groups={_TERM_GROUP[term]})
                comp[term] = s.getPotentialEnergy().value_in_unit(
                    unit.kilocalorie_per_mole)
        out_pos = None
        if want_positions:
            out_pos = np.array(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=float)
        self.t_energy += time.time() - t0
        return energy, comp, out_pos

    # ======================================================================
    # public interface
    # ======================================================================
    def energy(self, bitstring: str) -> float:
        """E(x) in kcal/mol. Cached; hits do not count as energy evaluations."""
        if len(bitstring) != self.rep.n_bits:
            raise ValueError(
                f"bitstring length {len(bitstring)} != n_bits {self.rep.n_bits}")
        hit = self._cache.get(bitstring)
        if hit is not None:
            return hit
        heavy = self._heavy_positions(self.rep.build_coords(bitstring))
        e, _, _ = self._evaluate(heavy)
        if len(self._cache) < self._cache_limit:
            self._cache[bitstring] = e
        self.n_energy_evaluations += 1
        return e

    def components(self, bitstring: str) -> Dict[str, float]:
        """Per-term breakdown for one bitstring, kcal/mol. Not cached.

        The five Amber terms sum to "total" exactly; there is no residual.
        """
        if len(bitstring) != self.rep.n_bits:
            raise ValueError(
                f"bitstring length {len(bitstring)} != n_bits {self.rep.n_bits}")
        heavy = self._heavy_positions(self.rep.build_coords(bitstring))
        e, comp, _ = self._evaluate(heavy, want_components=True)
        comp["total"] = e
        return comp

    def energy_from_coords(self, coords: Dict[str, np.ndarray],
                           phi=None, psi=None) -> float:
        """Energy of an arbitrary structure under the SAME Hamiltonian.

        Used to score the native for the energy-gap metric. EVALUATION path,
        not an optimization path. `coords` is a backbone dict with N, CA, C,
        CB and O, as produced by geo.native_coords_from_pdb; sidechains are
        rebuilt from it and the same capped minimization is applied, so the
        native gets exactly the treatment a predicted structure gets.
        `phi`/`psi` are accepted for interface compatibility and unused: the
        force field reads coordinates, not torsions.
        """
        heavy = self._heavy_positions(coords)
        e, _, _ = self._evaluate(heavy)
        return e

    def minimized_ca(self, bitstring: str) -> Tuple[np.ndarray, np.ndarray]:
        """(pre, post) CA coordinates in ANGSTROMS around the minimization.

        Diagnostic: lets validation check that the backbone restraints hold,
        i.e. that the bitstring -> structure mapping survives. Returned in the
        same lab frame, deliberately NOT superposed, so the RMSD between them
        is the strict measure of how far minimization moved the backbone.
        """
        heavy = self._heavy_positions(self.rep.build_coords(bitstring))
        pre = np.array([heavy[k] for k, (_, nm) in enumerate(self._heavy_names)
                        if nm == "CA"]) * 10.0
        _, _, pos = self._evaluate(heavy, want_positions=True)
        post = np.array([pos[self._heavy_index[(i, "CA")]]
                         for i in range(len(self.sequence))]) * 10.0
        return pre, post

    def reset_counters(self) -> None:
        self.n_energy_evaluations = 0
        self.t_build = self.t_minimize = self.t_energy = 0.0

    def cache_size(self) -> int:
        return len(self._cache)

    def describe(self) -> Dict[str, object]:
        return {
            "forcefield": "amber14/protein.ff14SB.xml + implicit/gbn2.xml",
            "units": "kcal/mol",
            "n_atoms": self.n_atoms,
            "n_hydrogens": len(self._hydrogens),
            "platform": self.platform.getName(),
            "platform_properties": dict(self.platform_properties),
            "restraint_k_kcal_per_mol_A2": self.restraint_k,
            "minimization_steps": self.minimization_steps,
            "hydrogen_strategy": "frozen local frames (relaxed once)",
            "setup_time_s": self.setup_time,
        }