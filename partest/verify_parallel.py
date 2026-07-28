"""Gate for turning on the REC 1/2/4 parallel paths (optimizations.md O2).

`mps/parallel.py` claims its process-parallel restarts and minimization pool are
result-identical, and `run_8state_seed0.py` keeps them behind `PFA_RESTART_PROCS` /
`PFA_MINIM_WORKERS`, both defaulting to serial. The claim is only worth acting on if it
is checked *on the machine that will run the experiment*, because the thing that could
break it -- OpenMM platform selection, thread pinning, worker Hamiltonian
reconstruction -- is environment-dependent.

This runs the three existing checks in order of increasing scope and reports one verdict:

  1. golden_check     -- AmberHamiltonian.energy still returns the persisted seed-0
                         values exactly (max abs error 0.0). Pins the energy function.
  2. rec2_pool_check  -- the same energies routed through MinimizationPool workers.
                         Pins REC 2.
  3. equiv_check      -- serial vs parallel driver on a small real OBC2 system, comparing
                         selection, history and distribution stats bit-for-bit. Pins REC 1.

Requires `openmm` and `quimb`. Exit code 0 means it is safe to enable.

    python partest/verify_parallel.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

CHECKS = [
    ("golden_check.py", "energy function is byte-identical to persisted seed-0 data"),
    ("rec2_pool_check.py", "REC 2 minimization pool is byte-identical"),
    ("equiv_check.py", "REC 1 parallel restarts select bit-for-bit identically"),
]


def main() -> int:
    try:
        import openmm  # noqa: F401
        import quimb  # noqa: F401
    except ImportError as exc:
        print(f"  CANNOT VERIFY: {exc}")
        print("  openmm and quimb are required. Do NOT enable the parallel paths on an")
        print("  environment where these checks have not run -- the result-identity")
        print("  claim is untested there.")
        return 2

    results = []
    for script, what in CHECKS:
        print("=" * 72)
        print(f"  {script}  --  {what}")
        print("=" * 72, flush=True)
        rc = subprocess.call([sys.executable, os.path.join(HERE, script)], cwd=REPO)
        results.append((script, rc))
        print(f"  -> {'PASS' if rc == 0 else f'FAIL (exit {rc})'}\n", flush=True)

    print("=" * 72)
    ok = all(rc == 0 for _, rc in results)
    for script, rc in results:
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] {script}")
    if ok:
        print("\n  ALL PASS -- the parallel paths are byte-identical on this machine.")
        print("  Enable with:\n")
        print("      PFA_RESTART_PROCS=3 PFA_MINIM_WORKERS=8 \\")
        print("          python run_8state_seed0.py\n")
        print("  Measured speedups on the profiling captures in diag_raw/:")
        print("      REC 2, 8-way minimization pool  2.95x  (prof3)")
        print("      REC 1, 3 concurrent MPS procs   2.45x  (prof6)")
    else:
        print("\n  FAIL -- do NOT enable the parallel paths. A failure here means a")
        print("  number moved, which is exactly what these recommendations promise")
        print("  never happens.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
