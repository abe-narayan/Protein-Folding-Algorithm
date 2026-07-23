STATUS
- 24/24 validation passing
- Ceiling bug fixed (was greedy projection, now annealed search)
- Chignolin ceiling 0.96 A, achieved 5.44 A -> 4.5 A gap, all energy-model
- Trpzip ceiling 1.72 A, achieved 8.33 A -> 6.6 A gap
- Energy gaps: -7.68 (1UAO), -6.78 (1LE0). Native scores WORSE than
  predictions on both. This is the whole problem.
- VQE, SA, random all within 0.25 A and 0.4 kcal/mol on chignolin ->
  search is solved, all three find the true minimum of a wrong Hamiltonian
- Corrected MJ gives W-W exactly 0.0 and Y-W -0.045 by construction
  (self-energy subtraction cancels like-with-like). Both benchmarks are
  held together by aromatic stacking the model cannot see.

DECISION: replace energy_terms.py with OpenMM Amber ff14SB + GBn2.
OpenMM 8.5.2 installed via pip, forcefields load OK.

PLAN (25h over 5 days)
D1 sidechains.py - fixed chi rotamers, full geometry for F Y W S T D E G P N K
D2 amber_hamiltonian.py - topology, backbone-restrained capped minimization,
   drop-in for FoldingHamiltonian. GO/NO-GO: native chignolin must score
   below all-helix. If not, debug, do not proceed.
D3 chignolin + trpzip, 3 seeds, 4 arms
D4 held-out: 1LE1, 2EVQ, 1J4M
D5 write-up

EXPECT 3-4 A, not 1 A. Fixed chi angles is a stated limitation
(variable chi would need 34 qubits = 17 GB, not simulable).