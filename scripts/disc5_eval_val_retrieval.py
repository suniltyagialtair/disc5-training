# disc5_eval_val_retrieval.py
# Per-passage retrieval listing for the ep21 ArcFace val embeddings (IARA + ShipsEar).
# For every val passage: its single nearest OTHER passage, whether that match is the
# same hull (HIT/MISS), and the cosine. Pools val SEGMENT embeddings to passages
# (vessel_id, session_id) per D44, exactly as disc5_eval_val_persource.py.
# Pure numpy on saved arrays -- no model, no GPU, no training compute.
#
# A query whose hull has only ONE passage in val has no possible correct answer in
# the gallery; it is listed but marked "(only passage of hull)" and excluded from
# the rank-1 tally, matching the eval convention.
#
# Usage:  python disc5_eval_val_retrieval.py

import csv
import sys
import numpy as np
from collections import Counter

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
    return str(int(tok))


def load_session_map(manifest_path):
    m = {}
    with open(manifest_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[(r["source"].upper(), str(int(r["recording_id"])))] = r["session_id"]
    return m


def short_label(vessel, session):
    """Compact, readable passage label."""
    sess = session.split("#", 1)[1] if "#" in session else session
    if vessel.startswith("SHIPSEAR:"):
        v = "SE:" + vessel.split(":")[-1]
    else:
        v = vessel  # 'IARA:455'
    return f"{v}/{sess}"


def pool_passages(emb, rows, session_map):
    """Group segments into passages; return P (L2-normed), vessels, sessions, sources."""
    groups = {}
    for i, r in enumerate(rows):
        rid = parse_recording_id(r["source"], r["parent_clip"])
        key = (r["source"].upper(), rid)
        if key not in session_map:
            raise KeyError(f"no manifest session for {key} (row {i}, clip {r['parent_clip']})")
        pkey = (r["vessel_id"], session_map[key])
        groups.setdefault(pkey, []).append(i)

    P, vessels, sessions, sources = [], [], [], []
    for (vessel, sess), idxs in groups.items():
        v = emb[idxs].mean(axis=0)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        P.append(v)
        vessels.append(vessel)
        sessions.append(sess)
        sources.append(vessel.split(":")[0].upper())
    return np.asarray(P, dtype=np.float64), vessels, sessions, sources


def main():
    emb = np.load(EMB_PATH, allow_pickle=True)
    with open(INDEX_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(emb):
        sys.exit(f"row/emb mismatch: {len(rows)} rows vs {len(emb)} embeddings")
    session_map = load_session_map(MANIFEST_PATH)

    P, vessels, sessions, sources = pool_passages(emb, rows, session_map)
    n = len(P)
    cnt = Counter(vessels)

    C = P @ P.T
    np.fill_diagonal(C, -np.inf)

    # stable display order: source, then vessel, then session
    order = sorted(range(n), key=lambda i: (sources[i] != "IARA", vessels[i], sessions[i]))

    hdr = f"{'QUERY PASSAGE':30s}  {'NEAREST MATCH':30s}  {'COS':>7s}  RESULT"
    cur_src = None
    hit = tot = 0
    src_hit = src_tot = 0

    def flush_src(s):
        if s is not None and src_tot:
            print(f"  -> {s} rank-1: {src_hit}/{src_tot} = {src_hit/src_tot:.4f}\n")

    for i in order:
        s = "IARA" if sources[i] == "IARA" else "ShipsEar"
        if s != cur_src:
            flush_src(cur_src)
            cur_src, src_hit, src_tot = s, 0, 0
            print(f"=== {s} ===")
            print(hdr)
        j = int(np.argmax(C[i]))
        cos = C[i, j]
        q = short_label(vessels[i], sessions[i])
        m = short_label(vessels[j], sessions[j])
        if cnt[vessels[i]] < 2:
            res = "-- (only passage of hull)"
        else:
            ok = vessels[j] == vessels[i]
            res = "HIT " if ok else "MISS"
            hit += ok; tot += 1
            src_hit += ok; src_tot += 1
        print(f"{q:30s}  {m:30s}  {cos:7.4f}  {res}")
    flush_src(cur_src)

    print(f"OVERALL rank-1 (scorable queries): {hit}/{tot} = {hit/tot:.4f}")


if __name__ == "__main__":
    main()
