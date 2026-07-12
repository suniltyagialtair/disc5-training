# disc5_eval_val_persource.py
# Post-hoc val evaluation of the ep21 ArcFace embeddings, split by source.
# Loads the saved val SEGMENT embeddings + their row index, pools segments to
# passages (vessel_id, session_id) per D44, and reports same/diff cosine, GAP,
# EER, rank-1 and diff<0.5 for the BLEND, IARA-only, and ShipsEar-only.
# Pure numpy on saved arrays -- no model, no GPU, no training compute.
#
# Passage key = (vessel_id, session_id). session_id is recovered by joining each
# segment's parent_clip -> recording_id -> manifest session_id on (source, recording_id),
# because recording_id is NOT globally unique across sources.
#
# Usage:
#   python disc5_eval_val_persource.py
#   (edit the three paths below if files are not in the current directory)

import csv
import sys
import numpy as np

EMB_PATH      = r"G:\My Drive\DISC5_Checkpoints\disc5_arcface_8k_best_val_seg_emb.npy"
INDEX_PATH    = r"G:\My Drive\DISC5_Checkpoints\disc5_arcface_8k_val_tensors_used.csv"
MANIFEST_PATH = r"G:\My Drive\DISC5\disc5_recording_manifest.csv"


def parse_recording_id(source, parent_clip):
    """IARA 'A-0025' -> 25 ; ShipsEar '10__10_07_13_marDeOnza_Sale' -> 10."""
    s = source.upper()
    if s == "IARA":
        tok = parent_clip.split("-")[-1]
    elif s == "SHIPSEAR":
        tok = parent_clip.split("__")[0]
    else:
        raise ValueError(f"unknown source {source!r} (parent_clip {parent_clip!r})")
    return str(int(tok))  # strip leading zeros, normalise to manifest's int-as-str


def load_session_map(manifest_path):
    """(source, recording_id_str) -> session_id, from the manifest."""
    m = {}
    with open(manifest_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[(r["source"].upper(), str(int(r["recording_id"])))] = r["session_id"]
    return m


def pool_passages(emb, rows, session_map):
    """Group segment rows into passages; return passage embeddings + labels.
    Returns: P (n_pass, d) L2-normed, vessels (list), sources (list)."""
    groups = {}  # passage_key -> list of row indices
    for i, r in enumerate(rows):
        rid = parse_recording_id(r["source"], r["parent_clip"])
        key = (r["source"].upper(), rid)
        if key not in session_map:
            raise KeyError(f"no manifest session for {key} (row {i}, clip {r['parent_clip']})")
        sess = session_map[key]
        pkey = (r["vessel_id"], sess)
        groups.setdefault(pkey, []).append(i)

    P, vessels, sources = [], [], []
    for (vessel, _sess), idxs in groups.items():
        v = emb[idxs].mean(axis=0)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        P.append(v)
        vessels.append(vessel)
        sources.append(vessel.split(":")[0].upper())  # 'IARA' / 'SHIPSEAR'
    return np.asarray(P, dtype=np.float64), vessels, sources


def pair_stats(P, vessels):
    """Same/diff passage-cosine arrays over all unordered passage pairs."""
    n = len(P)
    C = P @ P.T
    same, diff = [], []
    for a in range(n):
        for b in range(a + 1, n):
            (same if vessels[a] == vessels[b] else diff).append(C[a, b])
    return np.asarray(same), np.asarray(diff)


def eer(same, diff):
    """Equal error rate by threshold sweep. FAR = diff>=t accepted; FRR = same<t rejected."""
    if len(same) == 0 or len(diff) == 0:
        return float("nan")
    ts = np.unique(np.concatenate([same, diff]))
    best_t, best_gap = None, 1e9
    for t in ts:
        far = np.mean(diff >= t)
        frr = np.mean(same < t)
        if abs(far - frr) < best_gap:
            best_gap, best_t = abs(far - frr), t
    far = np.mean(diff >= best_t)
    frr = np.mean(same < best_t)
    return (far + frr) / 2.0


def rank1(P, vessels):
    """For each passage query, nearest OTHER passage; correct if same vessel.
    Only queries whose vessel has >=2 passages are scorable."""
    n = len(P)
    C = P @ P.T
    np.fill_diagonal(C, -np.inf)
    from collections import Counter
    cnt = Counter(vessels)
    hits = total = 0
    for i in range(n):
        if cnt[vessels[i]] < 2:
            continue  # no correct answer exists in gallery
        j = int(np.argmax(C[i]))
        hits += (vessels[j] == vessels[i])
        total += 1
    return (hits / total) if total else float("nan"), total


def report(name, P, vessels):
    same, diff = pair_stats(P, vessels)
    if len(same) == 0:
        print(f"  {name:14s}  (no same-vessel pairs -- cannot score)")
        return
    gap = same.mean() - diff.mean()
    d05 = np.mean(diff < 0.5) if len(diff) else float("nan")
    e = eer(same, diff)
    r1, r1n = rank1(P, vessels)
    print(f"  {name:14s}  passages {len(P):3d}  same-pairs {len(same):4d}  diff-pairs {len(diff):5d}")
    print(f"  {'':14s}  same {same.mean():.4f}  diff {diff.mean():.4f}  GAP {gap:+.4f}"
          f"  EER {e:.4f}  rank-1 {r1:.4f} (n={r1n})  diff<0.5 {d05:.3f}")


def main():
    emb = np.load(EMB_PATH, allow_pickle=True)
    with open(INDEX_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(emb):
        sys.exit(f"row/emb mismatch: {len(rows)} rows vs {len(emb)} embeddings")
    session_map = load_session_map(MANIFEST_PATH)

    P, vessels, sources = pool_passages(emb, rows, session_map)
    sources = np.asarray(sources)

    print(f"loaded {len(emb)} segments -> {len(P)} passages "
          f"({(sources=='IARA').sum()} IARA, {(sources=='SHIPSEAR').sum()} ShipsEar)\n")

    print("BLEND (all val passages):")
    report("blend", P, vessels)
    print("\nIARA only (deployment-relevant -- merchant-like):")
    mi = sources == "IARA"
    report("IARA", P[mi], [v for v, m in zip(vessels, mi) if m])
    print("\nShipsEar only (short-tail / small-boat):")
    ms = sources == "SHIPSEAR"
    report("ShipsEar", P[ms], [v for v, m in zip(vessels, ms) if m])


if __name__ == "__main__":
    main()
