"""Backed structural-validity report for a sequence against a reference PDB.

Run:  python validate_structure.py

Produces, for oxytocin (CYIQNCPLG vs 7OFG):
  * a null RMSD distribution and where the energy-selected ensemble sits in it,
  * the energy->RMSD Spearman correlation (predictive validity of the model),
  * a Mann-Whitney test of the experimentally-known disulfide's ensemble effect,
  * results/structure_validity_CYIQNCPLG.png and a stats CSV.

All numbers are measured, none tuned to the reference.
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

from analysis import (
    build_fold_table, variant_energy, empirical_pvalue, predictive_validity,
    ground_state_ensemble, disulfide_effect, CONTACT_D2,
)

SEQUENCE = "CYIQNCPLG"
PDB_ID = "7OFG"
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    table = build_fold_table(SEQUENCE, f"{PDB_ID}.pdb", chain_id="A")
    r = table["rmsd"]
    print(f"Sequence {SEQUENCE} vs {PDB_ID}  "
          f"({'exact' if table['exact'] else 'sampled'}, {table['n_folds']} folds)")
    print(f"Disulfide pairs inferred: {table['disulfide_pairs']}")
    print(f"RMSD null: best possible={r.min():.3f}  median(random)={np.median(r):.3f}")
    print()

    # --- predictive validity of energy minimization (the core check) ---
    E_base = variant_energy(table, compactness_weight=0.5)
    rho, p = predictive_validity(E_base, r)
    gs = ground_state_ensemble(E_base, r)
    print("ENERGY MODEL (contacts + compactness):")
    print(f"  Spearman(energy, RMSD) = {rho:+.3f}  (p={p:.1e})   "
          f"[positive => lower energy predicts more-native; negative => anti-predictive]")
    print(f"  energy ground state: E={gs['min_energy']:.3f}, {gs['degeneracy']}-fold "
          f"degenerate, RMSD median={gs['rmsd_median']:.3f} "
          f"[{gs['rmsd_min']:.3f}-{gs['rmsd_max']:.3f}]")
    print(f"  ensemble empirical p vs null = "
          f"{empirical_pvalue(gs['rmsd_median'], r):.3f}")
    print()

    # --- disulfide as a hard topological constraint: ensemble effect ---
    eff = disulfide_effect(table)
    print("DISULFIDE (hard topological constraint, Cys-Cys within bond shell):")
    print(f"  constrained folds: {eff['n_constrained']} / {table['n_folds']}")
    print(f"  median RMSD  unconstrained={eff['median_unconstrained']:.3f}  "
          f"-> constrained={eff['median_constrained']:.3f}  "
          f"(best constrained={eff['best_constrained']:.3f})")
    print(f"  Mann-Whitney U (one-sided, constrained more native): "
          f"p={eff['p_value']:.2e}")
    print()

    # --- write machine-readable stats ---
    csv_path = os.path.join(RESULTS, f"structure_validity_{SEQUENCE}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_folds", table["n_folds"]])
        w.writerow(["rmsd_best_possible", f"{r.min():.4f}"])
        w.writerow(["rmsd_null_median", f"{np.median(r):.4f}"])
        w.writerow(["spearman_energy_rmsd", f"{rho:.4f}"])
        w.writerow(["spearman_p", f"{p:.2e}"])
        w.writerow(["gs_degeneracy", gs["degeneracy"]])
        w.writerow(["gs_rmsd_median", f"{gs['rmsd_median']:.4f}"])
        for k, v in eff.items():
            w.writerow([f"disulfide_{k}", v])
    print(f"Stats saved: {csv_path}")

    # --- figure: (1) energy-RMSD anti-correlation, (2) disulfide ensemble shift ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.scatter(r, E_base, s=6, alpha=0.15, color="#4C6EF5", edgecolors="none")
    ax1.set_xlabel("Cα RMSD to reference (normalized units)")
    ax1.set_ylabel("Model energy (relative MJ units)")
    ax1.set_title(f"Predictive validity: Spearman ρ = {rho:+.2f}\n"
                  f"(ρ<0 ⇒ energy minimization is anti-native)", fontsize=11)
    ax1.grid(alpha=0.2)

    sat = table["disulfide_satisfied"]
    bins = np.linspace(r.min(), r.max(), 40)
    ax2.hist(r[~sat], bins=bins, density=True, alpha=0.55, color="#ADB5BD",
             label=f"unconstrained (med {eff['median_unconstrained']:.3f})")
    ax2.hist(r[sat], bins=bins, density=True, alpha=0.65, color="#F03E3E",
             label=f"disulfide-constrained (med {eff['median_constrained']:.3f})")
    ax2.axvline(r.min(), color="black", ls="--", lw=1,
                label=f"best possible ({r.min():.3f})")
    ax2.set_xlabel("Cα RMSD to reference (normalized units)")
    ax2.set_ylabel("density")
    ax2.set_title(f"Disulfide constraint shifts the ensemble\n"
                  f"toward native (Mann–Whitney p={eff['p_value']:.1e})", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    fig.suptitle(f"Structural validity of the lattice model — {SEQUENCE} vs {PDB_ID}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig_path = os.path.join(RESULTS, f"structure_validity_{SEQUENCE}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {fig_path}")

    if os.environ.get("SHOW_PLOTS", "1") != "0":
        plt.show()


if __name__ == "__main__":
    main()
