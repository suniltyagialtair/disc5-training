# disc5_eval_train_retrieval.py
# Memorization/recall check on TRAINING hulls with the ep21 ArcFace model.
# Train hulls were SEEN in training, so this is NOT a re-ID generalization test --
# it measures how well the model recalls identities it was optimized on. The
# informative quantity is the CONTRAST with val (IARA val rank-1 was 0.435): if
# train recall is near-perfect while val is poor, that is the overfit signature
# the train-val spread (+0.445) predicts; if train ALSO fails, the model did not
# capture identity even on seen data.
#
# No saved train embeddings exist (Cell 8 dumps val only), so this does a forward
# pass through best.pth. To stay tractable on CPU it samples N_HULLS train hulls
# (>=2 sessions, originals only) and caps segments per passage. Model + embedding
# extraction are copied verbatim from DISC5_Training_100.ipynb (Cells 5 / 11) so
# embeddings reproduce ep21. Passage = (vessel_id, session_id) per D44.
#
# Usage:  python disc5_eval_train_retrieval.py

import csv
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict, Counter

# ----- paths (local) -----
BEST_PTH   = r"G:\My Drive\DISC5_Checkpoints\disc5_arcface_8k_best.pth"
TENSOR_DIR = r"G:\My Drive\DISC5\tensors"          # holds tensor_index.csv + the .npy tree
MANIFEST   = r"G:\My Drive\DISC5\disc5_recording_manifest.csv"

# ----- sampling knobs (raise if you have a GPU / time) -----
N_HULLS              = 40     # train hulls to sample (mirrors the 40 IARA val hulls)
SEGS_PER_PASSAGE_CAP = 16     # max segments pooled per passage (CPU tractability)
MIN_SESSIONS         = 2      # a hull needs >=2 sessions for a same-hull match to exist
BATCH                = 8
SEED                 = 1234

# ----- model: copied verbatim from notebook Cell 5 -----
SK_KERNELS, SK_CH, EMBED = (127, 511, 2047, 8191), 64, 512

class SKFilterbank(nn.Module):
    def __init__(self, in_ch=1, out_ch=64, kernels=(127, 511, 2047, 8191), d=4):
        super().__init__(); self.convs = nn.ModuleList(); self.norms = nn.ModuleList(); ng = min(16, out_ch)
        for k in kernels:
            self.convs.append(nn.Conv1d(in_ch, out_ch, k, padding=k // 2)); self.norms.append(nn.GroupNorm(ng, out_ch))
        self.squeeze = nn.Linear(out_ch, d); self.excites = nn.ModuleList([nn.Linear(d, out_ch) for _ in kernels])
    def forward(self, x):
        outs = [F.relu(n(c(x))) for c, n in zip(self.convs, self.norms)]
        L = min(o.shape[-1] for o in outs); outs = [o[..., :L] for o in outs]
        st = torch.stack(outs, 0); U = st.sum(0); z = F.relu(self.squeeze(U.mean(-1)))
        a = F.softmax(torch.stack([e(z) for e in self.excites], 0), 0).unsqueeze(-1)
        return (st * a).sum(0)

class DISC5Encoder(nn.Module):
    def __init__(self, kernels, sk_ch=64, embed_dim=512):
        super().__init__(); self.fb = SKFilterbank(1, sk_ch, tuple(kernels))
        self.l1 = nn.Sequential(nn.Conv2d(1, 64, 3, stride=(1, 1), padding=1), nn.GroupNorm(16, 64), nn.ReLU(True))
        self.l2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=(1, 4), padding=1), nn.GroupNorm(16, 128), nn.ReLU(True))
        self.l3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=(1, 4), padding=1), nn.GroupNorm(16, 256), nn.ReLU(True))
        self.l4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=(2, 2), padding=1), nn.GroupNorm(16, 512), nn.ReLU(True))
        self.l5 = nn.Sequential(nn.Conv2d(512, embed_dim, 3, stride=(2, 2), padding=1), nn.GroupNorm(16, embed_dim), nn.ReLU(True))
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        h = self.fb(x).unsqueeze(1)
        for l in (self.l1, self.l2, self.l3, self.l4, self.l5):
            h = l(h)
        return F.normalize(self.pool(h).flatten(1), dim=1)


def stem_of(r):
    if r["source"] == "IARA":
        return f"{r['collection']}-{int(r['recording_id']):04d}"
    return Path(str(r["collection"])).stem


def short_label(vessel, session):
    sess = session.split("#", 1)[1] if "#" in session else session
    v = "SE:" + vessel.split(":")[-1] if vessel.startswith("SHIPSEAR:") else vessel
    return f"{v}/{sess}"


def main():
    rng = np.random.default_rng(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tdir = Path(TENSOR_DIR)
    with open(tdir / "tensor_index.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(MANIFEST, encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    stem2session = {stem_of(r): str(r["session_id"]) for r in man}
    session_for = lambda pc: stem2session.get(pc, pc)

    # train originals only
    tr = [r for r in rows if r["split"] == "train" and r["is_augmented"] == "0"]
    # group hull -> session -> rows
    by_hull = defaultdict(lambda: defaultdict(list))
    for r in tr:
        by_hull[r["vessel_id"]][session_for(r["parent_clip"])].append(r)
    eligible = [h for h, sess in by_hull.items() if len(sess) >= MIN_SESSIONS]
    print(f"train hulls with >={MIN_SESSIONS} sessions: {len(eligible)} (sampling {min(N_HULLS, len(eligible))})")
    sample = sorted(rng.choice(eligible, size=min(N_HULLS, len(eligible)), replace=False).tolist())

    # build the segment work list (cap per passage)
    work = []  # (vessel, session, path)
    for h in sample:
        for sess, rws in by_hull[h].items():
            pick = rws if len(rws) <= SEGS_PER_PASSAGE_CAP else \
                [rws[i] for i in rng.choice(len(rws), SEGS_PER_PASSAGE_CAP, replace=False)]
            for r in pick:
                work.append((h, sess, tdir / r["path"]))
    print(f"segments to embed: {len(work)}")

    # model
    model = DISC5Encoder(SK_KERNELS, SK_CH, EMBED).to(device).float()
    ck = torch.load(BEST_PTH, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded best.pth epoch {ck.get('epoch','?')} best_gap {ck.get('best_gap', float('nan')):.4f}")

    # forward pass in batches
    embs = []
    with torch.no_grad():
        for s in range(0, len(work), BATCH):
            batch = work[s:s + BATCH]
            x = torch.from_numpy(np.stack([np.load(p).astype("float32").reshape(1, -1) for _, _, p in batch]))
            embs.append(model(x.to(device)).cpu().numpy())
            if (s // BATCH) % 20 == 0:
                print(f"  {min(s + BATCH, len(work))}/{len(work)}", end="\r")
    seg = np.concatenate(embs, 0)
    print(f"\nembedded {len(seg)} segments")

    # pool to passages
    groups = defaultdict(list)
    for i, (h, sess, _) in enumerate(work):
        groups[(h, sess)].append(i)
    P, vessels, sessions = [], [], []
    for (h, sess), idxs in groups.items():
        v = seg[idxs].mean(0); n = np.linalg.norm(v)
        P.append(v / n if n > 0 else v); vessels.append(h); sessions.append(sess)
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    print(f"passages: {n} across {len(set(vessels))} hulls\n")

    # same/diff GAP (for contrast with val)
    C = P @ P.T
    iu = np.triu_indices(n, 1)
    same_mask = np.array([vessels[a] == vessels[b] for a, b in zip(*iu)])
    cos = C[iu]
    sm, dm = cos[same_mask], cos[~same_mask]
    print(f"TRAIN-sample GAP {sm.mean()-dm.mean():+.4f}  (same {sm.mean():.4f} / diff {dm.mean():.4f})  "
          f"same-pairs {len(sm)} diff-pairs {len(dm)}\n")

    # per-passage retrieval
    Cd = C.copy(); np.fill_diagonal(Cd, -np.inf)
    cnt = Counter(vessels)
    order = sorted(range(n), key=lambda i: (vessels[i], sessions[i]))
    print(f"{'QUERY PASSAGE':30s}  {'NEAREST MATCH':30s}  {'COS':>7s}  RESULT")
    hull = defaultdict(lambda: [0, 0]); hit = tot = 0
    for i in order:
        j = int(np.argmax(Cd[i]))
        q, m = short_label(vessels[i], sessions[i]), short_label(vessels[j], sessions[j])
        if cnt[vessels[i]] < 2:
            res = "-- (only passage of hull)"
        else:
            ok = vessels[j] == vessels[i]
            res = "HIT " if ok else "MISS"
            hull[vessels[i]][0] += ok; hull[vessels[i]][1] += 1; hit += ok; tot += 1
        print(f"{q:30s}  {m:30s}  {Cd[i, j]:7.4f}  {res}")

    # per-hull rollup
    clean = [h for h, (a, b) in hull.items() if a == b]
    failed = [h for h, (a, b) in hull.items() if a == 0]
    partial = [h for h, (a, b) in hull.items() if 0 < a < b]
    print(f"\nper-hull: {len(hull)} scorable hulls -- clean {len(clean)}, partial {len(partial)}, failed {len(failed)}")
    if partial:
        print("  partial: " + ", ".join(f"{h}({hull[h][0]}/{hull[h][1]})" for h in sorted(partial)))
    if failed:
        print("  failed:  " + ", ".join(sorted(failed)))
    print(f"\nTRAIN-sample rank-1 (scorable): {hit}/{tot} = {hit/tot:.4f}" if tot else "no scorable queries")


if __name__ == "__main__":
    main()
