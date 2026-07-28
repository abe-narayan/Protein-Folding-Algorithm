"""Shared evaluation-budget accounting for every energy model.

The comparison in README.md is quantum-vs-classical *at equal cost*. The only
hardware-agnostic currency available here is the number of distinct structures whose
energy had to be computed: under the amber model that is one OpenMM
backbone-restrained minimization each, and it dominates wall-clock for every arm.
Cache hits are free -- re-visiting a structure costs nothing -- so only cache misses
are charged.

Every search method (CVaR-VQE, simulated annealing, random search) is handed the same
budget through the Hamiltonian it shares. Whichever finds the lowest energy inside that
budget wins, and no method can buy a better answer with a larger optimizer allowance.

This is also what makes the VQE optimizer allowance well-posed. Previously ``maxiter``
was chosen per-arm and set the VQE's cost independently of what the baselines received;
now ``maxiter`` is set deliberately high and the shared budget is the binding
constraint. If COBYLA spends its whole allowance constructing an initial simplex in
120 dimensions while annealing converges inside the same budget, that is a genuine
result about optimizer choice rather than an artifact of how the arms were configured.

Readout reservation
-------------------
A run must be able to report its answer even after the optimizer has spent everything.
``reserve(k)`` withholds ``k`` evaluations behind a soft limit for the duration of the
search; ``release()`` hands them back for the final read-out. Charges still stop at the
hard limit, so the reserve bounds the read-out rather than exempting it.
"""
from typing import Dict, Optional


__all__ = ["BudgetExhausted", "BudgetedEnergyModel"]


class BudgetExhausted(RuntimeError):
    """Raised by ``energy()`` when a model has spent its evaluation budget.

    Search drivers catch this, stop, and return the best structure found so far. It is
    a normal termination condition, not a failure.
    """

    def __init__(self, limit: int, spent: int):
        super().__init__(
            f"evaluation budget exhausted: {spent} unique energy evaluations "
            f"spent of {limit} allowed")
        self.limit = int(limit)
        self.spent = int(spent)


class BudgetedEnergyModel:
    """Cache + budget bookkeeping shared by ``FoldingHamiltonian`` and ``AmberHamiltonian``.

    Subclasses call ``_init_budget`` from ``__init__`` and ``_charge()`` immediately
    before performing an uncached energy computation.
    """

    def _init_budget(self, cache_limit: int,
                     eval_budget: Optional[int] = None) -> None:
        self._cache: Dict[str, float] = {}
        self._cache_limit = int(cache_limit)
        self.eval_budget = None if eval_budget is None else int(eval_budget)
        self._reserved = 0
        self.n_energy_evaluations = 0

    # -- accounting ---------------------------------------------------------
    def _charge(self) -> None:
        """Account for one uncached energy computation, or raise if none is left."""
        if self.eval_budget is not None:
            limit = self.eval_budget - self._reserved
            if self.n_energy_evaluations >= limit:
                raise BudgetExhausted(limit, self.n_energy_evaluations)
        self.n_energy_evaluations += 1

    @property
    def budget_remaining(self) -> float:
        if self.eval_budget is None:
            return float("inf")
        limit = self.eval_budget - self._reserved
        return float(max(0, limit - self.n_energy_evaluations))

    @property
    def budget_exhausted(self) -> bool:
        return self.budget_remaining <= 0

    # -- readout reservation ------------------------------------------------
    def reserve(self, k: int) -> None:
        """Withhold ``k`` evaluations from the search so a read-out is always possible."""
        if self.eval_budget is None:
            return
        self._reserved = max(0, min(int(k), self.eval_budget))

    def release(self) -> None:
        """Return reserved evaluations to the pool (call before the final read-out)."""
        self._reserved = 0

    def end_search(self) -> None:
        """Stop charging. Everything after this point -- energy breakdowns, native
        comparisons, metrics -- is post-hoc analysis rather than search, and must not
        be billed to the search budget or be able to fail once it is spent. The spent
        count is preserved so reporting stays accurate.
        """
        self.eval_budget = None
        self._reserved = 0

    # -- lifecycle ----------------------------------------------------------
    def reset_counters(self) -> None:
        self.n_energy_evaluations = 0
        self._reserved = 0

    def cache_size(self) -> int:
        return len(self._cache)

    def budget_state(self) -> Dict[str, object]:
        """Serializable budget summary, for CSV rows and audit output."""
        return {
            "eval_budget": self.eval_budget,
            "n_energy_evaluations": self.n_energy_evaluations,
            "budget_remaining": self.budget_remaining,
            "budget_exhausted": self.budget_exhausted,
            "cache_size": self.cache_size(),
        }
