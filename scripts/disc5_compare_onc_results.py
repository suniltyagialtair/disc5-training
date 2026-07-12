# disc5_compare_onc_results.py
# Side-by-side comparison of SKANN (onc_eval_results.json/_trials.csv) vs LOFAR tonal
# (onc_tonal_results.json/_trials.csv) on the ONC cross-passage benchmark -- the same
# comparison shape used for the NODPAC 21-clip eval. Prints a metric table (overall +
# core/annex), per-trial agreement (which queries each method ranks correctly), and the
# genuine-score correlation between methods (complementarity signal: near-orthogonal
# failure modes on PS12 were corr ~ -0.19, enabling the z-score fusion).

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'G:\My Drive\SKANN_SSL\onc_data')
    ap.add_argument('--bench', default=None)
    args = ap.parse_args()
    bench = Path(args.bench) if args.bench else Path(args.root) / 'benchmark'

    skann = json.load(open(bench / 'onc_eval_results.json'))
    tonal = json.load(open(bench / 'onc_tonal_results.json'))
    ts = pd.read_csv(bench / 'onc_eval_trials.csv')
    tt = pd.read_csv(bench / 'onc_tonal_trials.csv')

    # ---- metric table --------------------------------------------------------------------
    metrics = ['rank1', 'rank1_macro_vessel', 'median_rank_of_genuine', 'eer', 'auc',
               'genuine_mean', 'impostor_mean']
    print(f'{"":28s}{"SKANN":>10s}{"TONAL":>10s}')
    for scope, key in [('overall', None), ('core', 'core'), ('annex', 'annex')]:
        a = skann['overall'] if key is None else skann['by_cohort'].get(key, {})
        b = tonal['overall'] if key is None else tonal['by_cohort'].get(key, {})
        print(f'--- {scope} (n={a.get("n_queries","?")}) ---')
        for m in metrics:
            print(f'  {m:26s}{a.get(m, float("nan")):>10.4f}{b.get(m, float("nan")):>10.4f}')

    # ---- per-trial agreement ---------------------------------------------------------------
    m = ts.merge(tt, on='query_uid', suffixes=('_sk', '_to'))
    both = (m.rank1_hit_sk & m.rank1_hit_to).sum()
    only_sk = (m.rank1_hit_sk & ~m.rank1_hit_to).sum()
    only_to = (~m.rank1_hit_sk & m.rank1_hit_to).sum()
    neither = (~m.rank1_hit_sk & ~m.rank1_hit_to).sum()
    print(f'\nrank-1 agreement over {len(m)} trials: both {both} | SKANN-only {only_sk} | '
          f'tonal-only {only_to} | neither {neither}')

    # genuine-score correlation (complementarity)
    corr = float(np.corrcoef(m.gen_score_sk, m.gen_score_to)[0, 1])
    print(f'genuine-score correlation (SKANN vs tonal): {corr:+.3f}')
    print('(PS12 reference: -0.19, near-orthogonal failure modes -> fusion gain)')

    # rank improvement per trial
    m['rank_sk'] = m.rank_of_genuine_sk
    m['rank_to'] = m.rank_of_genuine_to
    print(f'median rank of genuine: SKANN {m.rank_sk.median():.0f} | tonal {m.rank_to.median():.0f} '
          f'(gallery {skann["gallery_size"]})')
    w = (m.rank_to < m.rank_sk).sum()
    print(f'trials where tonal ranks the genuine higher than SKANN: {w}/{len(m)}')

    m[['query_uid', 'mmsi_sk', 'cohort_sk', 'rank_sk', 'rank_to',
       'gen_score_sk', 'gen_score_to', 'rank1_hit_sk', 'rank1_hit_to']].to_csv(
        bench / 'onc_compare_trials.csv', index=False)
    print(f'\nper-trial table: {bench / "onc_compare_trials.csv"}')

if __name__ == '__main__':
    main()
