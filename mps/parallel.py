"""Process-parallel orchestration for the CVaR-VQE driver (DIAGNOSIS.md REC 1/2/4).

These recommendations are *result-identical*: they change only which process computes an
energy and when, never the energy itself. Every energy is a pure, deterministic function
of its bitstring (ff14SB + native GBSAOBCForce, CPU, Threads=1, DeterministicForces), so
whether it is served from a shared cache, recomputed in a worker, or computed serially,
the value is byte-identical. Final selection depends only on those (unchanged) energies
and on RNG streams seeded from ``seed``/restart-seed, which are independent of process
layout. See DIAGNOSIS.md §5, §7.

Contents:
  * ``pin_threads``            — REC 4: pin BLAS/OMP to 1 thread (auto-run at import).
  * ``OBC2Builder``            — picklable factory so workers rebuild their own OBC2
                                 Hamiltonian (an ``openmm.Context`` cannot be pickled).
  * ``MinimizationPool``       — REC 2: persistent pool of worker Hamiltonians over the
                                 unique-structures loop.
  * ``run_parallel_restarts``  — REC 1: the 3 COBYLA restarts as independent processes
                                 with a shared-memory energy cache.
"""
import os


def pin_threads() -> None:
    """REC 4 — pin numeric-library thread pools to 1 (free enabler for REC 1/2).

    Set BEFORE numpy/OpenBLAS initialise their pools so the value takes effect. On
    Windows ``spawn`` a child inherits the parent environment at creation, so setting
    these in the parent before the pool is created pins the workers too; each worker
    initializer also calls this defensively. Pinning avoids BLAS oversubscription during
    the numpy ``assemble`` step when many minimizations run at once (DIAGNOSIS.md REC 4:
    lifted 8-way OpenMM scaling 2.44x -> 2.95x). It changes no computed value — numpy
    results are identical regardless of thread count, and OpenMM already runs Threads=1.
    """
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


# Pin at import so any process that imports this module (parent or worker) is pinned as
# early as possible. Pure os.environ writes — importing this module pulls in no numpy.
pin_threads()


class OBC2Builder:
    """Picklable factory that rebuilds the exact OBC2-native Hamiltonian in a worker.

    Carries everything the representation depends on — everything else in the construction
    is deterministic and verbatim (``amber_obc2.build_obc2_native``), so a worker's
    Hamiltonian is byte-identical to the parent's for energy purposes.

    ``sequence`` and ``legacy_library`` are load-bearing and must match the parent. The
    representation selects per-residue torsion libraries from the sequence
    (``representations.residue_classes``), so a worker that rebuilt without them would
    decode the same bitstring to different (phi, psi) values and therefore return a
    *different energy for the same structure* — silently, with no error, breaking exactly
    the byte-identical invariant this class exists to preserve.
    """

    def __init__(self, sequence: str, n_states: int,
                 legacy_library: bool = False):
        self.sequence = sequence
        self.n_states = int(n_states)
        self.legacy_library = bool(legacy_library)

    def __call__(self):
        # Imports deferred to call time (worker side); repo modules resolve via the
        # sys.path the spawn machinery restores from the parent.
        import representations as reps
        from amber_obc2 import build_obc2_native
        rep = reps.TorsionStateRepresentation(
            len(self.sequence), n_states=self.n_states, sequence=self.sequence,
            legacy_library=self.legacy_library)
        return build_obc2_native(self.sequence, rep)


class AmberBuilder:
    """Picklable factory for the current ff14SB + GBn2 Hamiltonian.

    This is the non-pinned counterpart of :class:`OBC2Builder` and is used only by new
    AMBER studies.  Each worker owns one OpenMM Context, so independent structures in a
    VQE/CEM batch can be minimized concurrently without sharing a non-thread-safe Context.
    """

    def __init__(self, sequence: str, n_states: int = 4,
                 restraint_k: float = 100.0, minimization_steps: int = 50,
                 minimization_tolerance: float = 2.0,
                 collapse_floor: float = -600.0):
        self.sequence = sequence.strip().upper()
        self.n_states = int(n_states)
        self.restraint_k = float(restraint_k)
        self.minimization_steps = int(minimization_steps)
        self.minimization_tolerance = float(minimization_tolerance)
        self.collapse_floor = float(collapse_floor)

    def __call__(self):
        import representations as reps
        import amber_hamiltonian as amber
        rep = reps.TorsionStateRepresentation(
            len(self.sequence), n_states=self.n_states, sequence=self.sequence)
        return amber.AmberHamiltonian(
            self.sequence, rep, restraint_k=self.restraint_k,
            minimization_steps=self.minimization_steps,
            minimization_tolerance=self.minimization_tolerance,
            collapse_floor=self.collapse_floor, eval_budget=None)


import sys
import multiprocessing as mp


def _ensure_repo_on_path() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _chunks(lst, k):
    """Round-robin ``lst`` into ``k`` balanced chunks (matches prof3 prototype)."""
    return [lst[i::k] for i in range(k)]


# --------------------------------------------------------------------------- REC 2
# Persistent pool of worker Hamiltonians over the unique-structures loop. Each worker
# owns its own OBC2 Hamiltonian (Threads=1) and computes energies for a chunk of
# bitstrings; results are byte-identical to the serial path (pure function of bitstring),
# verified with 0 mismatches against the golden CSVs.

_POOL_H = None


def _pool_init(builder):
    pin_threads()
    _ensure_repo_on_path()
    global _POOL_H
    _POOL_H = builder()


def _pool_work(bslist):
    global _POOL_H
    return [(bs, _POOL_H.energy(bs)) for bs in bslist]


class MinimizationPool:
    """REC 2 — a persistent ``spawn`` process pool that minimizes independent structures
    in parallel. ``compute(bitstrings)`` returns ``{bitstring: energy}``."""

    def __init__(self, builder, n_workers: int):
        self.n_workers = max(1, int(n_workers))
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(processes=self.n_workers,
                              initializer=_pool_init, initargs=(builder,))

    def compute(self, bitstrings):
        bslist = list(bitstrings)
        if not bslist:
            return {}
        k = min(self.n_workers, len(bslist))
        parts = self._pool.map(_pool_work, _chunks(bslist, k))
        out = {}
        for part in parts:
            out.update(dict(part))
        return out

    def close(self):
        try:
            self._pool.close()
            self._pool.join()
        except Exception:
            pass


class BudgetedEnergyBatch:
    """Connect a worker minimization pool to a parent budget/cache.

    ``compute`` may return only a prefix when the parent budget is nearly exhausted.
    Search drivers then fall back to ``parent.energy`` for the first missing value, which
    raises the normal :class:`budget.BudgetExhausted`.  This preserves the hard allowance
    while avoiding an all-or-nothing batch that would leave budget unused.
    """

    def __init__(self, parent_hamiltonian, builder, n_workers: int):
        self.parent = parent_hamiltonian
        self.pool = MinimizationPool(builder, n_workers)
        self.runtime = 0.0

    def compute(self, bitstrings):
        import time
        out, misses = {}, []
        for bs in bitstrings:
            hit = self.parent._cache.get(bs)
            if hit is None:
                if bs not in misses:
                    misses.append(bs)
            else:
                out[bs] = hit

        remaining = self.parent.budget_remaining
        allowed = len(misses) if remaining == float("inf") else min(
            len(misses), max(0, int(remaining)))
        todo = misses[:allowed]
        for _ in todo:
            self.parent._charge()
        if todo:
            t0 = time.time()
            computed = self.pool.compute(todo)
            self.runtime += time.time() - t0
            self.parent._cache.update(computed)
            out.update(computed)
        return out

    def close(self):
        self.pool.close()


# --------------------------------------------------------------------------- REC 1
# The 3 COBYLA restarts as independent processes. Each restart rebuilds its own
# Hamiltonian, runs the unchanged ``_run_single`` COBYLA trajectory (its result depends
# only on its restart seed + unchanged energies), and shares a read-mostly energy cache
# with the others so cross-restart reuse is preserved. Result is identical with or
# without sharing — selection depends only on the (unchanged) energies. (DIAGNOSIS §5.)


def _restart_worker(args):
    (builder, sampler, n_qubits, layers, alpha, shots, maxiter, rseed,
     optimizer, init_scale, verbose, shared_cache, minim_workers) = args
    pin_threads()
    _ensure_repo_on_path()
    from mps.driver import _run_single
    from vqe import BestSeenTracker

    H = builder()
    pool = (MinimizationPool(builder, minim_workers)
            if minim_workers and minim_workers > 1 else None)
    tracker = BestSeenTracker()
    n_min = [0]

    def energy_batch(bslist):
        # Local H cache (fast hits) -> shared cache (cross-restart reuse) -> compute.
        out, misses = {}, []
        for bs in bslist:
            v = H._cache.get(bs)
            if v is None and shared_cache is not None:
                v = shared_cache.get(bs)
                if v is not None:
                    H._cache[bs] = v
            if v is None:
                misses.append(bs)
            else:
                out[bs] = v
        if misses:
            if pool is not None:
                computed = pool.compute(misses)          # REC 2 parallel minimizations
            else:
                computed = {bs: H.energy(bs) for bs in misses}   # serial (REC 1 alone)
            n_min[0] += len(misses)
            for bs, e in computed.items():
                out[bs] = e
                if bs not in H._cache:
                    H._cache[bs] = e
                if shared_cache is not None:
                    shared_cache[bs] = e                 # populate shared cache
        return out

    run = _run_single(H, None, sampler, n_qubits, layers, alpha, shots, maxiter,
                      rseed, optimizer, tracker, init_scale, verbose,
                      energy_batch=energy_batch)
    if pool is not None:
        pool.close()
    run["final_params"] = np.asarray(run["final_params"], dtype=float).tolist()
    return {"run": run, "best_seen_bs": tracker.best_bitstring,
            "best_seen_e": tracker.best_energy, "n_minimizations": n_min[0]}


def _restart_worker_proc(idx, args, result_q):
    try:
        result_q.put((idx, _restart_worker(args), None))
    except Exception:
        import traceback
        result_q.put((idx, None, traceback.format_exc()))


def run_parallel_restarts(builder, sampler, restart_seeds, n_qubits, layers, alpha,
                          shots, maxiter, optimizer, init_scale, verbose,
                          minim_workers: int = 0, use_shared_cache: bool = True):
    """Run each restart in its own process. Returns (results_in_restart_order,
    shared_cache_snapshot). ``results[i]`` = {run, best_seen_bs, best_seen_e,
    n_minimizations}."""
    ctx = mp.get_context("spawn")
    mgr = ctx.Manager() if use_shared_cache else None
    shared_cache = mgr.dict() if mgr is not None else None
    result_q = ctx.Queue()

    procs = []
    for idx, rseed in enumerate(restart_seeds):
        args = (builder, sampler, n_qubits, layers, alpha, shots, maxiter,
                int(rseed), optimizer, init_scale, verbose, shared_cache, minim_workers)
        p = ctx.Process(target=_restart_worker_proc, args=(idx, args, result_q),
                        daemon=False)
        p.start()
        procs.append(p)

    results = [None] * len(restart_seeds)
    errors = []
    for _ in range(len(restart_seeds)):
        idx, res, err = result_q.get()
        if err is not None:
            errors.append((idx, err))
        else:
            results[idx] = res
    for p in procs:
        p.join()
    if errors:
        raise RuntimeError("parallel restart worker(s) failed:\n"
                           + "\n---\n".join(f"[restart {i}]\n{e}" for i, e in errors))

    cache_snapshot = dict(shared_cache) if shared_cache is not None else {}
    if mgr is not None:
        mgr.shutdown()
    return results, cache_snapshot


import numpy as np  # noqa: E402  (kept after the heavy defs; used by _restart_worker)
