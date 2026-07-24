
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import protein_geometry as geo
import representations as reps
import hamiltonian as ham
import vqe as vqe_mod
import evaluation as ev


SEQ = (sys.argv[1] if len(sys.argv) > 1 else "GYDPETGTWG").strip().upper()
KNOWN = {"GYDPETGTWG": "1UAO", "SWTWEGNKWTWK": "1LE0"}
PDB_ID = KNOWN.get(SEQ)

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


rep = reps.TorsionStateRepresentation(len(SEQ), n_states=4)
H = ham.FoldingHamiltonian(SEQ, rep)

print(f"sequence : {SEQ}  (N = {len(SEQ)}, {rep.n_qubits} qubits)")
geo.reset_pdb_log()
res = vqe_mod.run_global_cvar_vqe(H, layers=2, alpha=0.15, shots=1024,
                                  maxiter=150, restarts=1, seed=0,
                                  final_shots=4096, verbose=True)
assert len(geo.get_pdb_log()) == 0, "LEAKAGE: PDB read during optimization"

pred_ca = rep.build_coords(res["vqe_bitstring"])["CA"]
print(f"\nVQE energy : {res['vqe_energy']:.4f}")
print(f"bitstring  : {res['vqe_bitstring']}")

os.makedirs("results", exist_ok=True)

native_ca = None
if PDB_ID:
    pdb_path = os.path.join("pdbs", f"{PDB_ID}.pdb")
    if os.path.exists(pdb_path):
        nseq, ncoords, nphi, npsi = geo.native_coords_from_pdb(pdb_path)
        native_ca = np.asarray(ncoords["CA"])[:len(SEQ)]
        pred_ca = geo.kabsch_superpose(pred_ca, native_ca)
        r = geo.rmsd(pred_ca, native_ca)
        ceil_ = ev.representation_ceiling(rep, native_ca, nphi, npsi, seed=0)
        e_nat = H.energy_from_coords(ncoords, nphi, npsi)
        print(f"CA-RMSD    : {r:.2f} A   (ceiling {ceil_['ceiling_ca_rmsd']:.2f} A)")
        print(f"native E   : {e_nat:.4f}   gap {res['vqe_energy'] - e_nat:+.4f}")
        pred_title = (f"VQE prediction for '{SEQ}'\n"
                      f"CA-RMSD to {PDB_ID}: {r:.2f} A  |  "
                      f"E = {res['vqe_energy']:.2f}")
    else:
        print(f"  [{PDB_ID}.pdb not found in pdbs/ -- prediction only]")
        pred_title = f"VQE prediction for '{SEQ}'  (E = {res['vqe_energy']:.2f})"
else:
    pred_title = f"VQE prediction for '{SEQ}'  (E = {res['vqe_energy']:.2f})"

print()
plot_chain(pred_ca, SEQ, pred_title,
           os.path.join("results", f"{SEQ}_vqe_fold.png"))

if native_ca is not None:
    plot_chain(native_ca, SEQ,
               f"Experimental structure ({PDB_ID}) for '{SEQ}'",
               os.path.join("results", f"{SEQ}_native_{PDB_ID}.png"))

geo.write_pdb(os.path.join("results", f"{SEQ}_prediction.pdb"),
              SEQ, rep.build_coords(res["vqe_bitstring"]),
              remark=f"global CVaR-VQE prediction for {SEQ}")
print(f"  saved results/{SEQ}_prediction.pdb")

if os.environ.get("SHOW_PLOTS", "1") != "0":
    plt.show()