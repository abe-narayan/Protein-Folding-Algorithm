"""Peptide dataset: PDB download/cache, identity clustering, train/test splits.

2026-07-28 changes:

* `_identity` is a real global alignment identity. It was a longest-common-*subsequence*
  ratio, which allows unlimited gaps and so over-merged clusters. That matters now:
  identity clustering is the *only* guard against fitting energy weights to the answer
  (`energy_quality.holdout_report` depends on it), so it has to be load-bearing.
* NMR ensembles are loaded as ensembles -- see `load_ensemble`.
* `resolve_pdb_for_sequence` replaces the hardcoded sequence -> PDB dictionaries that
  were scattered through `main.py` and `plot_structures.py`. Those made two entry points
  silently special-case one particular peptide.
"""
import os
import socket
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

import protein_geometry as geo


BASE = os.path.dirname(os.path.abspath(__file__))
PDB_DIR = os.path.join(BASE, "pdbs")


CANDIDATE_PDB_IDS = [
    "1UAO",   # chignolin, 10, beta-hairpin
    "5AWL",   # CLN025, 10, beta-hairpin
    "1LE0",   # trpzip2, 12, beta-hairpin
    "1LE1",   # trpzip3, 12, beta-hairpin
    "1LE3",   # trpzip4, 16, beta-hairpin
    "2EVQ",   # 12, beta-hairpin
    "1J4M",   # 14, beta-hairpin
    "1E0Q",   # ubiquitin N-term hairpin, 17
    "1V4Z",   # 17
    "1B03",   # V3 loop peptide, 18
    "1DU1",   # charged helix, 20
    "1L2Y",   # trp-cage, 20, mixed
    "2JOF",   # trp-cage variant, 20
]

MIN_LEN = 10
MAX_LEN = 20
IDENTITY_THRESHOLD = 0.6


@dataclass
class PeptideEntry:
    pdb_id: str
    sequence: str
    pdb_path: str
    chain_id: Optional[str] = None
    cluster: int = -1

    @property
    def length(self) -> int:
        return len(self.sequence)

    def __repr__(self) -> str:
        return (f"PeptideEntry({self.pdb_id}, len={self.length}, "
                f"cluster={self.cluster})")


def download_pdb(pdb_id: str, cache_dir: str = PDB_DIR) -> Optional[str]:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        socket.setdefaulttimeout(30)
        urllib.request.urlretrieve(url, path)
        return path
    except Exception as exc:
        print(f"  [dataset] download failed for {pdb_id}: {exc}")
        if os.path.exists(path):
            os.remove(path)
        return None


def _identity(a: str, b: str, gap: float = -1.0) -> float:
    """Global (Needleman-Wunsch) alignment identity, normalised by the LONGER sequence.
    The previous implementation was an LCS-length ratio, which charges nothing for gaps
    and so scores unrelated sequences as similar -- it would happily cluster a 10-mer
    with a 20-mer that merely contains six of its residues in order. Since clustering is
    what keeps the weight calibration in `energy_quality` honest, a permissive identity
    measure silently leaks train sequences into the held-out set.
    
    Normalising by the *longer* sequence is what makes gaps cost something. Aligning a
    10-mer into a 19-mer forces nine gaps regardless of the gap penalty, and the
    max-match path also minimises them, so it wins at any penalty -- against `min(n, m)`
    a short sequence fully contained in a long one scores 1.00 no matter how the
    alignment is parameterised.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    # Score matrix with traceback-free identity counting: each cell carries
    # (best score, matches along that best path).
    prev = [(gap * j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(gap * i, 0)]
        ai = a[i - 1]
        for j in range(1, m + 1):
            match = 1.0 if ai == b[j - 1] else 0.0
            diag = (prev[j - 1][0] + match, prev[j - 1][1] + int(match > 0))
            up = (prev[j][0] + gap, prev[j][1])
            left = (cur[j - 1][0] + gap, cur[j - 1][1])
            best = diag
            if up[0] > best[0]:
                best = up
            if left[0] > best[0]:
                best = left
            cur.append(best)
        prev = cur
    return prev[m][1] / max(n, m)

def cluster_by_identity(entries: List[PeptideEntry],
                        threshold: float = IDENTITY_THRESHOLD):
    reps: List[str] = []
    for e in entries:
        assigned = -1
        for ci, rep_seq in enumerate(reps):
            if _identity(e.sequence, rep_seq) >= threshold:
                assigned = ci
                break
        if assigned < 0:
            assigned = len(reps)
            reps.append(e.sequence)
        e.cluster = assigned
    return entries


def build_dataset(pdb_ids: Optional[List[str]] = None,
                  cache_dir: str = PDB_DIR,
                  min_len: int = MIN_LEN, max_len: int = MAX_LEN,
                  verbose: bool = True) -> List[PeptideEntry]:
    ids = list(pdb_ids) if pdb_ids else list(CANDIDATE_PDB_IDS)
    entries: List[PeptideEntry] = []
    for pdb_id in ids:
        path = download_pdb(pdb_id, cache_dir)
        if path is None:
            continue
        try:
            seq, _, _, _ = geo.parse_pdb(path)
        except Exception as exc:
            if verbose:
                print(f"  [dataset] parse failed {pdb_id}: {exc}")
            continue
        if not (min_len <= len(seq) <= max_len):
            if verbose:
                print(f"  [dataset] skipping {pdb_id}: length {len(seq)} "
                      f"outside [{min_len}, {max_len}]")
            continue
        entries.append(PeptideEntry(pdb_id, seq, os.path.abspath(path)))
    cluster_by_identity(entries)
    if verbose:
        print(f"  [dataset] {len(entries)} usable peptides, "
              f"{len(set(e.cluster for e in entries))} identity clusters")
    return entries


def split_by_cluster(entries: List[PeptideEntry], seed: int = 0,
                     n_test_clusters: int = 3, n_val_clusters: int = 1):
    clusters: Dict[int, List[PeptideEntry]] = {}
    for e in entries:
        clusters.setdefault(e.cluster, []).append(e)
    ids = sorted(clusters.keys(), key=lambda c: (-len(clusters[c]), c))
    order = list(ids)
    np.random.default_rng(seed).shuffle(order)
    test_c = set(order[:n_test_clusters])
    val_c = set(order[n_test_clusters:n_test_clusters + n_val_clusters])
    train, val, test = [], [], []
    for c in ids:
        bucket = test if c in test_c else (val if c in val_c else train)
        bucket.extend(clusters[c])
    return train, val, test


def check_no_cluster_leak(train, val, test) -> bool:
    tc = {e.cluster for e in train}
    vc = {e.cluster for e in val}
    sc = {e.cluster for e in test}
    return tc.isdisjoint(sc) and tc.isdisjoint(vc) and vc.isdisjoint(sc)


def load_native(entry: PeptideEntry, model_index: int = 0):
    """First (or `model_index`-th) deposited model."""
    return geo.native_coords_from_pdb(entry.pdb_path, chain_id=entry.chain_id,
                                      model_index=model_index)


def load_ensemble(entry: PeptideEntry):
    """Every deposited model, as a list of (sequence, coords, phi, psi).

    For the NMR targets in `CANDIDATE_PDB_IDS` this is what RMSD should be measured
    against; a single model is one arbitrary draw from the ensemble's own spread.
    """
    return geo.native_ensemble_from_pdb(entry.pdb_path, chain_id=entry.chain_id)


@lru_cache(maxsize=1)
def _sequence_index(cache_dir: str = PDB_DIR) -> Tuple[Tuple[str, str], ...]:
    """(sequence, pdb_id) for every parseable PDB in the local cache."""
    out = []
    if not os.path.isdir(cache_dir):
        return tuple()
    for name in sorted(os.listdir(cache_dir)):
        if not name.lower().endswith(".pdb"):
            continue
        path = os.path.join(cache_dir, name)
        try:
            seq, _, _, _ = geo.parse_pdb(path)
        except Exception:
            continue
        out.append((seq, os.path.splitext(name)[0].upper()))
    return tuple(out)


def resolve_pdb_for_sequence(sequence: str,
                             cache_dir: str = PDB_DIR) -> Optional[str]:
    """PDB id whose deposited sequence matches `sequence` exactly, or None.

    Replaces the hardcoded ``{"GYDPETGTWG": "1UAO"}`` lookups that had accumulated in
    `main.py` and `plot_structures.py`. Those made a specific peptide the only one for
    which those code paths produced native comparisons, which is exactly the kind of
    single-target coupling that makes an accuracy claim untrustworthy.

    Note this *reads PDB files*, so it is only ever safe to call outside a search --
    `protein_geometry` records the access and the leakage assertions will fire otherwise.
    """
    seq = sequence.strip().upper()
    for cached_seq, pdb_id in _sequence_index(cache_dir):
        if cached_seq == seq:
            return pdb_id
    return None
