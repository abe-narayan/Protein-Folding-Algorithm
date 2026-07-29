"""Native plot + RMSD for an existing prediction, with the PDB id given explicitly.

    python native_rmsd.py results/SWTWENGKWTWK_prediction.pdb 1LE0

Does not re-run any search -- it only reads the prediction already on disk.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import protein_geometry as geo
import dataset as ds
import evaluation as ev

pred_pdb, pdb_id = sys.argv[1], sys.argv[2].upper()
seq, _, CA, _ = geo.parse_pdb(pred_pdb)

path = ds.download_pdb(pdb_id)
if path is None:
    raise SystemExit(f"could not fetch {pdb_id}")
ensemble = geo.native_ensemble_from_pdb(path)
nseq = ensemble[0][0]
models = [np.asarray(c["CA"]) for _, c, _, _ in ensemble]

print(f"prediction : {seq}  (N={len(seq)})")
print(f"native     : {nseq}  (N={len(nseq)}, {len(models)} model(s))")
if len(nseq) != len(seq):
    raise SystemExit("length mismatch -- CA-RMSD is undefined between different lengths")
if nseq != seq:
    print("  !! sequences differ -- this is a CROSS-SEQUENCE RMSD, label it as such")

ens = geo.ca_rmsd_to_ensemble(CA, models)
best = models[ens["best_model"]]
pred = geo.kabsch_superpose(CA, best)
spread = geo.ensemble_spread(models)
mask = ev.well_determined_mask(models)
core = ev.core_ca_rmsd(CA, best, mask)

print(f"  CA-RMSD, model 1        : {ens['model1']:.2f} A")
print(f"  CA-RMSD, best model     : {ens['min']:.2f} A (model {ens['best_model'] + 1})")
print(f"  CA-RMSD, mean           : {ens['mean']:.2f} A")
print(f"  ensemble spread         : {spread:.2f} A  <- experimental floor")
print(f"  core CA-RMSD ({int(mask.sum())}/{len(mask)})   : {core:.2f} A")

stem = os.path.splitext(pred_pdb)[0]
for tag, title, series in (
    (f"native_{pdb_id}", f"Experimental {pdb_id} (model {ens['best_model'] + 1})",
     [("native", best, "black", "-")]),
    (f"overlay_{pdb_id}", f"{seq}  CA-RMSD {ens['min']:.2f} A "
                          f"(spread {spread:.2f} A)",
     [("native " + pdb_id, best, "black", "-"), ("prediction", pred, "crimson", "--")]),
):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label, xyz, color, ls in series:
        ax.plot(*xyz.T, color=color, lw=2.5, ls=ls, label=label)
        ax.scatter(*xyz.T, color=color, s=40)
    for i, (x, y, z) in enumerate(series[-1][1]):
        ax.text(x, y, z + 0.4, f"{i + 1}:{seq[i]}", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("X (A)"); ax.set_ylabel("Y (A)"); ax.set_zlabel("Z (A)")
    ax.view_init(elev=20, azim=45); ax.legend()
    out = f"{stem}_{tag}.png"
    pts = np.vstack([xyz for _, xyz, _, _ in series])
    c = pts.mean(axis=0)
    r = float(np.abs(pts - c).max()) * 1.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)

if os.environ.get("SHOW_PLOTS", "1") != "0":
    plt.show()
