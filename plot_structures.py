"""Plot predicted and native structures from files on disk.

This used to run its own VQE with a hardcoded ``layers=2, maxiter=150, restarts=1``
config that bypassed ``experiments.run_one``, ``default_vqe_config`` and the shared
evaluation budget -- a fourth, undocumented way to produce a prediction, and one whose
numbers were not comparable with anything in ``results/``. It also carried its own
hardcoded ``{sequence: pdb_id}`` map, so it only ever drew native comparisons for two
specific peptides.

It is now a plotter. Produce the prediction with::

    python main.py --predict --sequence <SEQ>

then::

    python plot_structures.py results/<SEQ>_prediction.pdb
"""
import os
import sys
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import protein_geometry as geo
import dataset as ds
import evaluation as ev


COLOR_MAP = {
    **{aa: "red" for aa in "AVLIMFWC"},      # hydrophobic
    **{aa: "blue" for aa in "STNQGYP"},      # polar
    **{aa: "green" for aa in "KRH"},         # basic
    **{aa: "orange" for aa in "DE"},         # acidic
}


def plot_chain(coords, sequence, title, path):
    coords = np.asarray(coords, dtype=float)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
            color="gray", linestyle="--", linewidth=2, zorder=1)

    for i, (x, y, z) in enumerate(coords):
        aa = sequence[i]
        ax.scatter(x, y, z, color=COLOR_MAP.get(aa, "black"), s=120,
                   zorder=2, edgecolors="k", linewidths=0.5)
        ax.text(x, y, z + 0.4, f"{i + 1}:{aa}", fontsize=9)

    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    ax.set_xlabel("X (A)")
    ax.set_ylabel("Y (A)")
    ax.set_zlabel("Z (A)")
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")
    return fig


def main(pred_pdb: str, out_dir: Optional[str] = None) -> int:
    if not os.path.exists(pred_pdb):
        print(f"no such file: {pred_pdb}")
        print(__doc__)
        return 1
    out_dir = out_dir or os.path.dirname(os.path.abspath(pred_pdb)) or "."
    os.makedirs(out_dir, exist_ok=True)

    seq, N, CA, C = geo.parse_pdb(pred_pdb)
    stem = os.path.splitext(os.path.basename(pred_pdb))[0]
    print(f"prediction : {pred_pdb}")
    print(f"sequence   : {seq}  (N = {len(seq)})")

    # Native, resolved by sequence match against the local PDB cache rather than from a
    # hardcoded table -- so this works for any target that has been downloaded.
    pdb_id = ds.resolve_pdb_for_sequence(seq)
    native_ca = None
    title = f"Prediction: '{seq}'"

    if pdb_id:
        ensemble = geo.native_ensemble_from_pdb(
            os.path.join(ds.PDB_DIR, f"{pdb_id}.pdb"))
        models = [np.asarray(c["CA"]) for _, c, _, _ in ensemble]
        native_ca = models[0]
        pred_ca = geo.kabsch_superpose(CA, native_ca)
        ens = geo.ca_rmsd_to_ensemble(CA, models)
        spread = geo.ensemble_spread(models)
        mask = ev.well_determined_mask(models)
        core = ev.core_ca_rmsd(CA, models[ens["best_model"]], mask)
        print(f"native     : {pdb_id}  ({ens['n_models']} deposited model(s))")
        print(f"  CA-RMSD to model 1        : {ens['model1']:.2f} A")
        print(f"  CA-RMSD, best model       : {ens['min']:.2f} A "
              f"(model {ens['best_model'] + 1})")
        print(f"  CA-RMSD, mean over models : {ens['mean']:.2f} A")
        print(f"  ensemble spread           : {spread:.2f} A  "
              f"<- experimental resolution floor")
        print(f"  core CA-RMSD ({int(mask.sum())}/{len(mask)} ordered) : {core:.2f} A")
        title = (f"Prediction for '{seq}'\n"
                 f"CA-RMSD to {pdb_id}: {ens['min']:.2f} A "
                 f"(best of {ens['n_models']} models; spread {spread:.2f} A)")
    else:
        pred_ca = CA
        print("native     : no cached PDB matches this sequence")

    plot_chain(pred_ca, seq, title, os.path.join(out_dir, f"{stem}_fold.png"))
    if native_ca is not None:
        plot_chain(native_ca, seq,
                   f"Experimental structure ({pdb_id}, model 1) for '{seq}'",
                   os.path.join(out_dir, f"{stem}_native_{pdb_id}.png"))

    if os.environ.get("SHOW_PLOTS", "1") != "0":
        plt.show()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1],
                          sys.argv[2] if len(sys.argv) > 2 else None))