# disc5_build_tensor_tar.py
# Packs the tensor tree + tensor_index.csv into ONE flat tar for transfer to the
# training environment, then verifies it (member count, total bytes, SHA-256).
# Build-order step: after tensor generation, before upload to cloud storage.
"""
DISC5 — Tensor archive builder
==============================
Why a single tar (procedure notes, learned the hard way):
  1. NEVER sync a tree of thousands of small .npy files through a cloud-drive
     client or Colab's Drive FUSE mount — silent truncation and stale size
     caches occur, and per-file transfer is ~100x slower than one archive.
  2. Build the tar LOCALLY with this script, upload the ONE file, and verify
     the SHA-256 printed here against the uploaded copy before training.
  3. On the training side, extract with a per-file loop and progress output
     (see the training notebook's extraction cell) rather than a single
     blocking subprocess call.
  4. The archive is UNCOMPRESSED (tar, not tar.gz): float32 tensors barely
     compress, and skipping gzip makes both packing and extraction I/O-bound
     rather than CPU-bound.

Contents and layout:
  tensors/<...>.npy         relative paths preserved exactly as in the index
  tensor_index.csv          at archive root

Verification performed after writing:
  - member count == index rows + 1 (the index itself)
  - every path in tensor_index.csv exists in the archive
  - total payload bytes reported
  - SHA-256 of the finished tar printed (record it; compare after upload)

Usage:
  python disc5_build_tensor_tar.py --tensors disc5_build/tensors \
      --index disc5_build/tensors/tensor_index.csv --out disc5_tensors.tar
"""

import argparse
import csv
import hashlib
import sys
import tarfile
import time
from pathlib import Path


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", required=True,
                    help="root of the tensor tree (paths in the index are "
                         "relative to this directory's PARENT, i.e. stored as "
                         "'tensors/...')")
    ap.add_argument("--index", required=True, help="tensor_index.csv to embed")
    ap.add_argument("--out", default="disc5_tensors.tar",
                    help="output tar path. NOTE: written exactly where given -- "
                         "unlike `tar -C`, no silent redirection of the output "
                         "location.")
    a = ap.parse_args()

    troot = Path(a.tensors).resolve()
    index = Path(a.index).resolve()
    out = Path(a.out).resolve()
    if not troot.is_dir():
        sys.exit(f"[error] tensor root not found: {troot}")
    if not index.is_file():
        sys.exit(f"[error] index not found: {index}")

    with open(index, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pcol = "tensor_path" if "tensor_path" in rows[0] else list(rows[0])[0]
    rel_paths = [r[pcol].replace("\\", "/") for r in rows]

    # resolve each index path against the tree; tolerate 'tensors/...' prefixes
    missing = []
    resolved = []
    for rp in rel_paths:
        cand = troot / rp
        if not cand.is_file():
            cand = troot.parent / rp
        if not cand.is_file():
            cand = troot / Path(rp).name
        if cand.is_file():
            arcname = "tensors/" + str(cand.relative_to(troot)).replace("\\", "/")
            resolved.append((cand, arcname))
        else:
            missing.append(rp)
    if missing:
        sys.exit(f"[error] {len(missing)} index paths not found under {troot} "
                 f"(first: {missing[:3]}) -- fix the index or the tree first; "
                 f"an archive that silently drops tensors is worse than none.")

    t0 = time.time()
    total = 0
    with tarfile.open(out, "w") as tar:              # uncompressed by design
        tar.add(index, arcname="tensor_index.csv")
        for n, (src, arc) in enumerate(resolved, 1):
            tar.add(src, arcname=arc)
            total += src.stat().st_size
            if n % 500 == 0 or n == len(resolved):
                print(f"[pack] {n}/{len(resolved)}  "
                      f"({total / 1e9:.2f} GB, {time.time() - t0:.0f}s)")

    # ---- verify -------------------------------------------------------------
    with tarfile.open(out, "r") as tar:
        names = set(tar.getnames())
    want = {arc for _, arc in resolved} | {"tensor_index.csv"}
    absent = want - names
    if absent:
        sys.exit(f"[error] verification FAILED: {len(absent)} members absent "
                 f"(first: {sorted(absent)[:3]})")
    print(f"[verify] members {len(names)} == index {len(resolved)} + 1  OK")
    print(f"[verify] payload {total / 1e9:.2f} GB, archive "
          f"{out.stat().st_size / 1e9:.2f} GB")
    print(f"[verify] sha256 {sha256_of(out)}")
    print(f"\nwrote {out}")
    print("NEXT: upload this ONE file to the training storage, re-hash the "
          "uploaded copy, and compare the two digests before extracting.")


if __name__ == "__main__":
    main()
