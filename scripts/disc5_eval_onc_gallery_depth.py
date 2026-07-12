# disc5_eval_onc_gallery_depth.py
# Tests the gallery-grows hypothesis on the ONC benchmark: does enrolling MORE passages of
# a vessel rescue cross-passage re-ID even though single-passage enrollment is at chance?
# Leave-one-out over the 15 multi-passage vessels using the passage embeddings already saved
# by the scoring notebook (embedding_store/disc5_arcface_8k_onc_passage_emb.npz) -- no audio,
# no GPU, runs in seconds.
# Protocol per (vessel, held-out query passage, depth k): enroll the k earliest remaining
# passages of that vessel (gallery accumulates over time), pool by mean+renorm (store
# convention), rank the query against [pooled genuine entry + every other vessel's standard
# single gallery passage]. Reports rank-1 / median rank / genuine-impostor cosine vs k.
# Output: console table + onc_gallery_depth.json in the benchmark folder.

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=r'G:\My Drive\DISC5_Checkpoints\embedding_store\disc5_arcface_8k_onc_passage_emb.npz')
    ap.add_argument('--bench', default=r'G:\My Drive\SKANN_SSL\onc_data\benchmark')
    ap.add_argument('--max-depth', type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    df = pd.DataFrame({'uid': z['uids'], 'mmsi': z['mmsi'].astype(np.int64),
                       'role': z['role'], 'cohort': z['cohort']})
    E = z['emb'].astype(np.float64)                       # (135, 512), unit-norm
    df['pid'] = df.uid.str.rsplit('_p', n=1).str[-1].astype(int)   # passage number ~ time order
    df = df.sort_values(['mmsi', 'pid']).reset_index(drop=True)
    E = E[[int(i) for i in df.index]] if False else z['emb'][np.argsort(
        pd.DataFrame({'uid': z['uids']}).uid.map(dict(zip(df.uid, df.index)))).values]
    # simpler, exact: rebuild E in df order
    order = {u: i for i, u in enumerate(z['uids'])}
    E = np.stack([z['emb'][order[u]] for u in df.uid]).astype(np.float64)

    counts = df.groupby('mmsi').size()
    multi = counts[counts >= 2].index
    base_gal = df[df.role == 'gallery'].reset_index()     # one standard passage per vessel

    def pool(idx):
        e = E[idx].mean(0)
        return e / (np.linalg.norm(e) + 1e-12)

    rows = []
    for m in multi:
        v = df[df.mmsi == m]
        for _, q in v.iterrows():
            qi = df.index.get_loc(q.name) if hasattr(df.index, 'get_loc') else q.name
            qe = E[q.name]
            rest = v.drop(q.name).sort_values('pid')      # enrollment pool, time order
            # impostor gallery: every OTHER vessel's standard gallery passage
            imp = base_gal[base_gal.mmsi != m]
            imp_E = np.stack([E[i] for i in imp['index']])
            imp_scores = imp_E @ qe
            for k in range(1, min(len(rest), args.max_depth) + 1):
                g = pool(list(rest.index[:k]))
                gen = float(g @ qe)
                rank = 1 + int((imp_scores > gen).sum())
                rows.append({'mmsi': int(m), 'query_uid': q.uid, 'cohort': q.cohort, 'k': k,
                             'gen_score': gen, 'best_imp': float(imp_scores.max()),
                             'rank': rank, 'rank1': rank == 1,
                             'n_gallery': len(imp) + 1})
    R = pd.DataFrame(rows)
    R.to_csv(Path(args.bench) / 'onc_gallery_depth_trials.csv', index=False)

    print(f'{"k":>2s} {"trials":>7s} {"rank1":>7s} {"med_rank":>9s} {"gen_mean":>9s} {"best_imp":>9s}')
    summary = {}
    for k, g in R.groupby('k'):
        summary[int(k)] = {'n_trials': int(len(g)), 'rank1': round(float(g.rank1.mean()), 4),
                           'median_rank': float(g['rank'].median()),
                           'gen_mean': round(float(g.gen_score.mean()), 4),
                           'best_imp_mean': round(float(g.best_imp.mean()), 4)}
        s = summary[int(k)]
        print(f'{k:>2d} {s["n_trials"]:>7d} {s["rank1"]:>7.3f} {s["median_rank"]:>9.1f} '
              f'{s["gen_mean"]:>9.4f} {s["best_imp_mean"]:>9.4f}')
    # fixed-trial-set comparison: only queries whose vessel supports max depth, so the
    # depth curve is not confounded by the trial mix changing with k
    deep = R.groupby('query_uid').k.max()
    fixed = R[R.query_uid.isin(deep[deep >= args.max_depth].index)]
    if len(fixed):
        print(f'\nfixed trial set (vessels supporting k={args.max_depth}; '
              f'{fixed.query_uid.nunique()} queries):')
        for k, g in fixed.groupby('k'):
            print(f'  k={k}: rank1 {g.rank1.mean():.3f}  med_rank {g["rank"].median():.1f}  '
                  f'gen {g.gen_score.mean():.4f}')
        summary['fixed_set'] = {int(k): {'rank1': round(float(g.rank1.mean()), 4),
                                         'median_rank': float(g['rank'].median()),
                                         'gen_mean': round(float(g.gen_score.mean()), 4)}
                                for k, g in fixed.groupby('k')}
    with open(Path(args.bench) / 'onc_gallery_depth.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\noutputs: {Path(args.bench) / 'onc_gallery_depth.json'}, onc_gallery_depth_trials.csv")

if __name__ == '__main__':
    main()
