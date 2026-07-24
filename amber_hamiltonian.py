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

AMBER_TERMS = ("bond", "angle", "torsion", "nonbonded", "solvation")
_GROUP_OF = {"HarmonicBondForce": 1, "HarmonicAngleForce": 2,
             "PeriodicTorsionForce": 3, "CMAPTorsionForce": 3,
             "NonbondedForce": 4}
_TERM_GROUP = {"bond": 1, "angle": 2, "torsion": 3,
               "nonbonded": 4, "solvation": 5}
_RESTRAINT_GROUP = 6

RESTRAINED_BACKBONE = ("N", "CA", "C")
_NON_FRAME_ATOMS = ("O", "OXT")


class AmberHamiltonian:


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
        self.n_energy_evaluations = 0   
        self.t_build = 0.0            
        self.t_minimize = 0.0         
        self.t_energy = 0.0            

        t0 = time.time()
        self._build_topology()
        self._build_system(platform_name)
        self._calibrate_hydrogens(reference_states)
        self.setup_time = time.time() - t0

        self.weights = {t: 1.0 for t in AMBER_TERMS}
        if weights:
            self.weights.update(weights)

    @property
    def n_qubits(self) -> int:
        return self.rep.n_qubits

    @property
    def n_bits(self) -> int:
        return self.rep.n_bits

    def _reference_backbone(self) -> Dict[str, np.ndarray]:
        n = len(self.sequence)
        phi = np.full(n, math.radians(-120.0))
        psi = np.full(n, math.radians(130.0))
        phi[0] = geo.DEFAULT_PHI
        psi[n - 1] = geo.DEFAULT_PSI
        return geo.build_backbone(phi, psi)

    def _build_topology(self) -> None:
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
            adj[k].sort()         
        self._adj = adj
        self._hparent = parent
        missing = [h for h in self._hydrogens if h not in parent]
        if missing:
            raise RuntimeError(
                f"{len(missing)} hydrogens have no heavy parent in the "
                "topology; cannot build deterministic frames")

    def _frame_atoms(self, p: int) -> Tuple[int, int]:
        ri = self._resof[p]
        if self._nameof[p] == "N" and ri > 0:
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
            constraints=None,      
            rigidWater=False,
            removeCMMotion=False,
            implicitSolventKappa=0.0 / unit.nanometer) 
        for f in self.system.getForces():
            f.setForceGroup(_GROUP_OF.get(f.__class__.__name__, 5))


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
        n = len(self.sequence)
        states = [1] * n if reference_states is None else list(reference_states)
        bits = self.rep.bitstring_from_states(states)
        heavy = self._heavy_positions(self.rep.build_coords(bits))

        pos = np.array(self._modeller_positions, dtype=float)
        for k, idx in enumerate(self._heavy_order):
            pos[idx] = heavy[k]

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

        self._h_local: List[Tuple[int, int, int, int, np.ndarray]] = []
        for h in self._hydrogens:
            p = self._hparent[h]
            a, b = self._frame_atoms(p)
            e1, e2, e3 = self._frame(relaxed[p], relaxed[a], relaxed[b])
            d = relaxed[h] - relaxed[p]
            self._h_local.append(
                (h, p, a, b,
                 np.array([np.dot(d, e1), np.dot(d, e2), np.dot(d, e3)])))

    def _heavy_positions(self, backbone: Dict[str, np.ndarray]) -> np.ndarray:
        full = sc.build_full_structure(self.sequence, backbone)
        out = np.empty((len(self._heavy_names), 3), dtype=float)
        for k, (i, nm) in enumerate(self._heavy_names):
            out[k] = full["residues"][i][nm]
        return out * 0.1

    def _assemble(self, heavy_nm: np.ndarray) -> np.ndarray:
        pos = np.empty((self.n_atoms, 3), dtype=float)
        for k, idx in enumerate(self._heavy_order):
            pos[idx] = heavy_nm[k]
        for h, p, a, b, c in self._h_local:
            e1, e2, e3 = self._frame(pos[p], pos[a], pos[b])
            pos[h] = pos[p] + c[0] * e1 + c[1] * e2 + c[2] * e3
        return pos

    def _evaluate(self, heavy_nm: np.ndarray, want_components: bool = False,
                  want_positions: bool = False):
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
        ctx.setParameter("k_rest", 0.0)    
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


    def energy(self, bitstring: str) -> float:
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

        if len(bitstring) != self.rep.n_bits:
            raise ValueError(
                f"bitstring length {len(bitstring)} != n_bits {self.rep.n_bits}")
        heavy = self._heavy_positions(self.rep.build_coords(bitstring))
        e, comp, _ = self._evaluate(heavy, want_components=True)
        comp["total"] = e
        return comp

    def energy_from_coords(self, coords: Dict[str, np.ndarray],
                           phi=None, psi=None) -> float:

        heavy = self._heavy_positions(coords)
        e, _, _ = self._evaluate(heavy)
        return e

    def minimized_ca(self, bitstring: str) -> Tuple[np.ndarray, np.ndarray]:

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