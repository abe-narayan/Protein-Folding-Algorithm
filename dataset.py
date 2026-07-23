"""Short-peptide benchmark set from the RCSB PDB.

All structures are experimentally determined. Splitting is sequence-identity
clustered so that if any learned component is ever added, no near-homologue
appears on both sides. The DEFAULT pipeline has no learned parameters, so
the split exists for methodological hygiene and future use, not because it is
currently load-bearing.
"""
import os
import socket
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

import protein_geometry as geo


BASE = os.path.dirname(os.path.abspath(__file__))
PDB_DIR = os.path.join(BASE, "pdbs")

# Curated 10-20mers with well-resolved experimental structures, spanning
# beta-hairpin, alpha-helix, and mixed topologies.
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


def _identity(a: str, b: str) -> float:
    """LCS-based identity proxy, normalized by the shorter sequence."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m] / min(n, m)


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
    clusters = {}
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


def load_native(entry: PeptideEntry):
    """Native sequence, coordinates, and torsions. EVALUATION USE ONLY."""
    return geo.native_coords_from_pdb(entry.pdb_path, chain_id=entry.chain_id)