# disc5_score_onc_holdout.py
# The pre-declared before/after exam for the ONC domain fine-tune probe. Scores TWO
# checkpoints (before = ep21 best.pth, after = ftONC ep006) on the HOLDOUT-7 vessels'
# cross-passage trials: queries restricted to role=query passages of holdout-side
# vessels (never trained), gallery = the full frozen 97-vessel gallery. The 8 trained
# vessels remain in the gallery as legitimate hard distractors but every trial records
# whether the top hit was a TRAINED vessel (attraction check: did fine-tuning create
# sinks?). Embedding, pooling (mean+renorm), and metrics are IMPORTED from
# disc5_score_onc_eval.py so both sides are computed by the identical code path.
# Run from C:\DISC5. CPU works (~2x 1.3k segment embeds, expect 15-40 min); pass
# --device cuda on a GPU machine. Score ONCE per the pre-declaration: no epoch shopping.
# Output: onc_holdout_results.json + onc_holdout_trials.csv + printed table.

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from disc5_score_onc_eval import load_encoder, eer_from_scores, auc_rank  # same dir

def embed_passages(enc, man, device, batch):
    embs = {}
    with torch.no_grad():
        for uid, g in man.groupby('passage_uid'):
            segs = []
            for c in range(0, len(g), batch):
                xb = torch.cat([torch.from_numpy(np.load(p)).view(1, 1, 40000)
                                for p in g.tensor_path.iloc[c:c + batch]]).to(device)
                segs.append(enc(xb).cpu())
            embs[uid] = F.normalize(torch.cat(segs).mean(0), dim=0).numpy()
    return embs

def score_side(tag, embs, gal, qry, trained_mmsi):
    G = np.stack([embs[u] for u in gal.passage_uid])
    Q = np.stack([embs[u] for u in qry.passage_uid])
    S = Q @ G.T
    trials, imp_all = [], []
    for qi, qrow in qry.reset_index(drop=True).iterrows():
        gen_cols = np.where(gal.mmsi.values == qrow.mmsi)[0]
        order = np.argsort(-S[qi])
        rank_of_gen = int(np.where(np.isin(order, gen_cols))[0][0]) + 1
        top_mmsi = int(gal.mmsi.values[order[0]])
        trials.append(dict(model=tag, query_uid=qrow.passage_uid, mmsi=int(qrow.mmsi),
                           gen_score=float(S[qi, gen_cols].max()),
                           best_imp_score=float(np.delete(S[qi], gen_cols).max()),
                           rank_of_genuine=rank_of_gen,
                           hit1=rank_of_gen == 1, hit5=rank_of_gen <= 5,
                           hit10=rank_of_gen <= 10, hit20=rank_of_gen <= 20,
                           top_match_mmsi=top_mmsi,
                           top_is_trained=bool(top_mmsi in trained_mmsi)))
        imp_all.append(np.delete(S[qi], gen_cols))
    T = pd.DataFrame(trials)
    gen = T.gen_score.values; imp = np.concatenate(imp_all)
    summ = dict(model=tag, n_trials=len(T),
                rank1=float(T.hit1.mean()), hit5=float(T.hit5.mean()),
                hit10=float(T.hit10.mean()), hit20=float(T.hit20.mean()),
                median_rank=float(T.rank_of_genuine.median()),
                auc=auc_rank(gen, imp), eer=eer_from_scores(gen, imp),
                gen_mean=float(gen.mean()), imp_mean=float(imp.mean()),
                top_is_trained_frac=float(T.top_is_trained.mean()))
    return T, summ

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'G:\My Drive\SKANN_SSL\onc_data')
    ap.add_argument('--bench', default=None)
    ap.add_argument('--split', default=r'C:\DISC5\onc_finetune_split.csv')
    ap.add_argument('--ckpt-before', default=r'G:\My Drive\DISC5\disc5_arcface_8k_best.pth')
    ap.add_argument('--ckpt-after',  default=r'G:\My Drive\DISC5_Checkpoints\disc5_arcface_8k_ftONC_ep006.pth')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch', type=int, default=32)
    a = ap.parse_args()

    bench = Path(a.bench) if a.bench else Path(a.root) / 'benchmark'
    man = pd.read_csv(bench / 'onc_eval_tensor_manifest.csv')
    split = pd.read_csv(a.split)
    holdout = set(split[split.side == 'holdout'].vessel_mmsi.astype(int))
    trained = set(split[split.side == 'train'].vessel_mmsi.astype(int))
    print(f'holdout vessels: {sorted(holdout)}')

    info = man.groupby('passage_uid').agg(mmsi=('vessel_mmsi', 'first'),
                                          role=('role', 'first')).reset_index()
    gal = info[info.role == 'gallery'].reset_index(drop=True)
    qry = info[(info.role == 'query') & (info.mmsi.astype(int).isin(holdout))].reset_index(drop=True)
    need = set(gal.passage_uid) | set(qry.passage_uid)
    man = man[man.passage_uid.isin(need)]
    print(f'gallery passages: {len(gal)} | holdout query passages: {len(qry)} | segments to embed: {len(man)}')

    device = torch.device(a.device)
    out_T, out_S = [], []
    for tag, ckpt in (('before_ep21', a.ckpt_before), ('after_ftONC_ep6', a.ckpt_after)):
        enc = load_encoder(ckpt, device)
        embs = embed_passages(enc, man, device, a.batch)
        T, summ = score_side(tag, embs, gal, qry, trained)
        out_T.append(T); out_S.append(summ)
        del enc, embs

    pd.concat(out_T).to_csv('onc_holdout_trials.csv', index=False)
    with open('onc_holdout_results.json', 'w') as f:
        json.dump(dict(holdout_mmsi=sorted(holdout), trained_mmsi=sorted(trained),
                       gallery_n=len(gal), results=out_S), f, indent=2)

    print(f'\nHOLDOUT-7 cross-passage re-ID  (N={out_S[0]["n_trials"]} genuine trials -> '
          f'coarse; read DIRECTION, not exact rates)')
    cols = ('rank1', 'hit5', 'hit10', 'hit20', 'median_rank', 'auc', 'eer',
            'gen_mean', 'imp_mean', 'top_is_trained_frac')
    hdr = 'model            ' + ''.join(f'{c:>14s}' for c in cols)
    print(hdr); print('-' * len(hdr))
    for s in out_S:
        print(f'{s["model"]:17s}' + ''.join(f'{s[c]:14.3f}' for c in cols))
    print('\nwrote onc_holdout_results.json + onc_holdout_trials.csv')

if __name__ == '__main__':
    main()
