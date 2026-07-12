# disc5_score_onc_eval.py
# Scores the frozen ep21 SKANN checkpoint on the ONC SCVIP cross-passage re-ID benchmark.
# Consumes onc_eval_tensor_manifest.csv (.npy tensors, project convention, from
# disc5_build_onc_eval_manifest.py --cut),
# embeds every 5-s tensor with the frozen encoder (head discarded), pools each passage
# by mean+renorm (D-series: no linear averaging without renormalisation), enrolls
# role=gallery passages, scores role=query passages against the full gallery by cosine.
# Reports rank-1, EER, AUC and genuine/impostor stats -- overall, per cohort (core/annex),
# and macro-averaged per vessel so deep-passage ferries cannot dominate the headline.
# Output: onc_eval_results.json (navy-eval-style) + per-trial CSV for error analysis.
# CPU is fine (~1.5k segments); pass --device cuda if available.

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== model: verbatim from disc5_skann_navy_embed_score.ipynb Cell 3 ====================
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
    def __init__(self, kernels=(127, 511, 2047, 8191), sk_ch=64, embed_dim=512):
        super().__init__(); self.fb = SKFilterbank(1, sk_ch, tuple(kernels))
        self.l1 = nn.Sequential(nn.Conv2d(1, 64, 3, stride=(1, 1), padding=1), nn.GroupNorm(16, 64), nn.ReLU(True))
        self.l2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=(1, 4), padding=1), nn.GroupNorm(16, 128), nn.ReLU(True))
        self.l3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=(1, 4), padding=1), nn.GroupNorm(16, 256), nn.ReLU(True))
        self.l4 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=(2, 2), padding=1), nn.GroupNorm(16, 512), nn.ReLU(True))
        self.l5 = nn.Sequential(nn.Conv2d(512, embed_dim, 3, stride=(2, 2), padding=1), nn.GroupNorm(16, embed_dim), nn.ReLU(True))
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        h = self.fb(x).unsqueeze(1)
        for l in (self.l1, self.l2, self.l3, self.l4, self.l5): h = l(h)
        return F.normalize(self.pool(h).flatten(1), dim=1)
# ===== end verbatim ======================================================================

def load_encoder(ckpt_path, device):
    enc = DISC5Encoder().to(device).eval()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck.get('model', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
    enc_state = {k.replace('encoder.', '', 1): v for k, v in state.items() if k.startswith('encoder.')}
    if not enc_state:  # checkpoint may already be encoder-only
        enc_state = {k: v for k, v in state.items() if not k.startswith(('head.', 'arcface', 'fc'))}
    missing, unexpected = enc.load_state_dict(enc_state, strict=False)
    print(f'checkpoint: {ckpt_path}  (missing {len(missing)}, unexpected {len(unexpected)})')
    assert not missing, f'missing encoder weights: {missing[:5]}'
    return enc

def eer_from_scores(gen, imp):
    """EER via threshold sweep over pooled scores."""
    scores = np.concatenate([gen, imp]); labels = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    order = np.argsort(-scores); labels = labels[order]
    fnr = 1 - np.cumsum(labels) / max(labels.sum(), 1)
    fpr = np.cumsum(1 - labels) / max((1 - labels).sum(), 1)
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)

def auc_rank(gen, imp):
    """AUC by rank statistic (Mann-Whitney)."""
    allv = np.concatenate([gen, imp])
    r = pd.Series(allv).rank().values
    return float((r[:len(gen)].sum() - len(gen) * (len(gen) + 1) / 2) / (len(gen) * len(imp)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'G:\My Drive\SKANN_SSL\onc_data')
    ap.add_argument('--bench', default=None)
    ap.add_argument('--ckpt', default=r'G:\My Drive\DISC5\disc5_arcface_8k_best.pth')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    bench = Path(args.bench) if args.bench else Path(args.root) / 'benchmark'
    man = pd.read_csv(bench / 'onc_eval_tensor_manifest.csv')
    print(f'tensors: {len(man)}  passages: {man.passage_uid.nunique()}')
    device = torch.device(args.device)
    enc = load_encoder(args.ckpt, device)

    # ---- embed all segments, pool per passage (mean + renorm) ---------------------------
    embs = {}
    with torch.no_grad():
        for uid, g in man.groupby('passage_uid'):
            segs = []
            for chunk in range(0, len(g), args.batch):
                batch = torch.cat([torch.from_numpy(np.load(p)).view(1, 1, 40000)
                                   for p in g.tensor_path.iloc[chunk:chunk + args.batch]]).to(device)
                segs.append(enc(batch).cpu())
            e = torch.cat(segs).mean(0)
            embs[uid] = F.normalize(e, dim=0).numpy()
    print(f'embedded {len(embs)} passages')

    info = man.groupby('passage_uid').agg(mmsi=('vessel_mmsi', 'first'), role=('role', 'first'),
                                          cohort=('cohort', 'first')).reset_index()
    gal = info[info.role == 'gallery'].reset_index(drop=True)
    qry = info[info.role == 'query'].reset_index(drop=True)
    G = np.stack([embs[u] for u in gal.passage_uid])
    Q = np.stack([embs[u] for u in qry.passage_uid])
    S = Q @ G.T   # cosine (rows: queries, cols: gallery)

    # ---- trials --------------------------------------------------------------------------
    trials = []
    for qi, qrow in qry.iterrows():
        genuine_cols = np.where(gal.mmsi.values == qrow.mmsi)[0]
        rank_order = np.argsort(-S[qi])
        top_mmsi = gal.mmsi.values[rank_order[0]]
        # rank of the genuine gallery entry
        rank_of_gen = int(np.where(np.isin(rank_order, genuine_cols))[0][0]) + 1
        trials.append({'query_uid': qrow.passage_uid, 'mmsi': qrow.mmsi, 'cohort': qrow.cohort,
                       'gen_score': float(S[qi, genuine_cols].max()),
                       'best_imp_score': float(np.delete(S[qi], genuine_cols).max()),
                       'rank1_hit': bool(top_mmsi == qrow.mmsi), 'rank_of_genuine': rank_of_gen,
                       'top_match_mmsi': int(top_mmsi)})
    T = pd.DataFrame(trials)
    T.to_csv(bench / 'onc_eval_trials.csv', index=False)

    gen = T.gen_score.values
    imp = np.concatenate([np.delete(S[qi], np.where(gal.mmsi.values == qrow.mmsi)[0])
                          for qi, qrow in qry.iterrows()])

    def block(t, g_scores, i_scores):
        macro = t.groupby('mmsi').rank1_hit.mean().mean()  # per-vessel, then across vessels
        return {'n_queries': int(len(t)), 'rank1': round(float(t.rank1_hit.mean()), 4),
                'rank1_macro_vessel': round(float(macro), 4),
                'median_rank_of_genuine': float(t.rank_of_genuine.median()),
                'eer': round(eer_from_scores(g_scores, i_scores), 4),
                'auc': round(auc_rank(g_scores, i_scores), 4),
                'genuine_mean': round(float(g_scores.mean()), 4),
                'impostor_mean': round(float(i_scores.mean()), 4)}

    out = {'overall': block(T, gen, imp), 'gallery_size': int(len(gal)),
           'by_cohort': {}}
    for c in ('core', 'annex'):
        tc = T[T.cohort == c]
        if len(tc):
            qidx = qry[qry.cohort == c].index
            ic = np.concatenate([np.delete(S[qi], np.where(gal.mmsi.values == qry.loc[qi].mmsi)[0]) for qi in qidx])
            out['by_cohort'][c] = block(tc, tc.gen_score.values, ic)
    with open(bench / 'onc_eval_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f'\noutputs: {bench / "onc_eval_results.json"}, {bench / "onc_eval_trials.csv"}')

if __name__ == '__main__':
    main()
