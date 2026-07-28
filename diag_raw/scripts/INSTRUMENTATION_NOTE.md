These prof*.py scripts import the repo READ-ONLY via a scratchpad COPY on PYTHONPATH.
The ONLY code change made (to the COPY only, never the real repo) was adding 3 timing
counters inside AmberHamiltonian._evaluate to split the bundled t_minimize into
setPositions / restraint-update / minimize-only, via `self._t_* = getattr(...) + dt`.
This does not alter any energy (verified: 800+ golden energies matched with max abs
err 0.0). Golden CSVs referenced by absolute path in prior-session scratchpads:
  chignolin: ...\04a0f21c-...\scratchpad\seed0_full_eval_set.csv
  trpzip:    ...\84b6b99e-...\scratchpad\trpzip_seed0_full_eval_set.csv
