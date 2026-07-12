# disc5_report_navy.py
# Combined NODPAC self-match report: reads disc5_arcface_8k_navy_skann.json and
# disc5_arcface_8k_navy_tonal.json from the embedding store and prints ONE condition x method
# comparison -- rows = clean / +noise / +speed / +speed+noise, columns = SKANN vs TONAL.
#
# Reports NUMBERS ONLY -- rank-1, open-set AUC, and false-alarm -- per method per condition, in two
# framings side by side (raw never overwritten):
#   * raw        -- every clip is a distinct identity (the scoring scripts' default).
#   * collapsed  -- clips asserted same-source by EXTERNAL provenance (CLUSTERS below, NOT SKANN
#                   cosine -- that would be circular) merged to one identity; a query retrieving any
#                   cluster member counts as a hit, within-cluster pairs move impostor->genuine.
# Also tallies, per condition, how many clips produced ZERO tonal lines (no signature) and tonal
# rank-1 restricted to clips that did form one. This is a diagnostic, not an argument.
#
# This script states the comparison and draws NO conclusions: the tonal side was just rebuilt with a
# corrected (TPSW) normaliser and the head-to-head must be read fresh from these numbers.
# Reads disc5_arcface_8k_navy_skann.json (UNCHANGED) + the new disc5_arcface_8k_navy_tonal.json.
# Writes disc5_navy_comparison.md. Local, stdlib + numpy.
# N=21 genuine + ~420 impostor per condition -> counts are coarse; read DIRECTION, not exact rates.

import json
from pathlib import Path
import numpy as np

STORE_DIR = Path(r'G:\My Drive\DISC5_Checkpoints\embedding_store')
TAG       = 'disc5_arcface_8k'
CONDS     = [('clean', 'clean'), ('noise', '+noise'), ('speed', '+speed'), ('speed_noise', '+speed+noise')]

# same-source clusters from EXTERNAL provenance (NOT from SKANN cosine). One list per identity.
CLUSTERS  = [['MV__MV_12', 'MV__MV_13', 'MV__MV_14']]

def load(name):
    p = STORE_DIR / f'{TAG}_navy_{name}.json'
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def sweep(gen, imp):
    taus = np.unique(np.concatenate([gen, imp])); P, N = len(gen), len(imp); best = None
    for t in taus:
        TP = int((gen >= t).sum()); FN = P - TP
        FP = int((imp >= t).sum()); TN = N - FP
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / P if P else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        row = dict(tau=float(t), TP=TP, FP=FP, FN=FN, TN=TN,
                   false_alarm=(FP / N if N else 0.0), f1=f1)
        if best is None or row['f1'] > best['f1']:
            best = row
    return best

def identity_map(clip_order, clusters):
    m = {c: c for c in clip_order}
    for grp in clusters:
        for c in grp:
            if c in m:
                m[c] = f'ID::{grp[0]}'
    return m

def score(S, clip_order, clusters):
    """rank-1, AUC, and F1-opt false-alarm under an identity map (clusters=[] -> raw)."""
    n = len(clip_order)
    ids = [identity_map(clip_order, clusters)[c] for c in clip_order]
    same = np.array([[ids[i] == ids[j] for j in range(n)] for i in range(n)])
    pred = S.argmax(1)
    rank1 = float(np.mean([same[i, pred[i]] for i in range(n)]))
    gen_q = np.array([S[i][same[i]].max() for i in range(n)])                 # best same-identity per query
    imp_q = np.array([S[i][~same[i]].max() if (~same[i]).any() else -np.inf for i in range(n)])
    g, m = gen_q[:, None], imp_q[None, :]
    auc = float((g > m).mean() + 0.5 * (g == m).mean())
    gen_pairs = S[same]            # within-identity (incl. diagonal)
    imp_pairs = S[~same]           # cross-identity
    f = sweep(gen_pairs, imp_pairs)
    return dict(rank1=rank1, auc=auc, fa=f['false_alarm'], n_imp=len(imp_pairs),
                hits=int(round(rank1 * n)), n=n)

def no_sig_rows(S):
    return [i for i in range(S.shape[0]) if np.allclose(S[i], 0.0)]

def restricted_rank1(S, clip_order, clusters):
    """rank-1 over only the clips whose query produced a signature (non-zero row)."""
    n = len(clip_order); keep = [i for i in range(n) if not np.allclose(S[i], 0.0)]
    if not keep:
        return None, 0
    ids = [identity_map(clip_order, clusters)[c] for c in clip_order]
    pred = S.argmax(1)
    hit = [1 for i in keep if ids[pred[i]] == ids[i]]
    return len(hit) / len(keep), len(keep)

def main():
    skann, tonal = load('skann'), load('tonal')
    if skann is None and tonal is None:
        print('no NODPAC result JSONs found in', STORE_DIR); return
    have = [(n, d) for n, d in [('SKANN', skann), ('TONAL', tonal)] if d is not None]

    # precompute raw + collapsed per method per condition
    R = {}
    for name, d in have:
        order = d['clip_order']
        R[name] = {}
        for key, _ in CONDS:
            S = np.array(d['self_match'][key]['S'])
            R[name][key] = dict(raw=score(S, order, []), col=score(S, order, CLUSTERS))
            # sanity: recomputed raw rank-1 must match the stored (4-dp rounded) value
            assert abs(R[name][key]['raw']['rank1'] - d['self_match'][key]['rank1']) < 1e-3

    cluster_lbl = ' + '.join('{' + ','.join(c.split('__')[-1] for c in g) + '}' for g in CLUSTERS)

    # ---- console ----
    print('\nNODPAC self-match -- condition x method  (constructed half-split ground truth)')
    print(f'collapse = same source -> one identity: {cluster_lbl}  (from provenance, not SKANN cosine)')
    print('N=21 genuine + ~420 impostor per condition -> counts coarse; read DIRECTION.\n')

    colw = 30
    def block(metric_label, get):
        head = f'{"condition":14s}' + ''.join(f'{n:>{colw}s}' for n, _ in have)
        print(metric_label); print(head); print('-' * len(head))
        for key, label in CONDS:
            line = f'{label:14s}'
            for name, _ in have:
                line += f'{get(name, key):>{colw}s}'
            print(line)
        print()

    block('rank-1  (raw -> collapsed):',
          lambda name, key: f"{R[name][key]['raw']['rank1']:.3f} -> {R[name][key]['col']['rank1']:.3f} "
                            f"({R[name][key]['col']['hits']}/21)")
    block('open-set AUC  (raw -> collapsed):',
          lambda name, key: f"{R[name][key]['raw']['auc']:.3f} -> {R[name][key]['col']['auc']:.3f}")
    block('false-alarm @ F1-opt  (raw -> collapsed):',
          lambda name, key: f"{R[name][key]['raw']['fa']:.3f} -> {R[name][key]['col']['fa']:.3f}")

    print('degradation (collapsed rank-1: clean -> +speed, the Doppler condition):')
    for name, _ in have:
        c = R[name]['clean']['col']['rank1']; s = R[name]['speed']['col']['rank1']
        print(f'  {name:6s} {c:.3f} -> {s:.3f}   (D{s - c:+.3f})')

    # ---- tonal fairness ----
    if tonal is not None:
        order = tonal['clip_order']
        print('\ntonal fairness  (SKANN has no no-signature clips):')
        print(f'{"condition":14s}{"no-signature":>14s}{"rank-1 all":>12s}{"rank-1 w/ signature":>22s}')
        for key, label in CONDS:
            S = np.array(tonal['self_match'][key]['S'])
            nsig = no_sig_rows(S); rr, kept = restricted_rank1(S, order, CLUSTERS)
            rr_s = f'{rr:.3f} ({kept} clips)' if rr is not None else 'n/a'
            print(f'{label:14s}{f"{len(nsig)}/21":>14s}{R["TONAL"][key]["col"]["rank1"]:>12.3f}{rr_s:>22s}')

    # ---- markdown ----
    md = ['# NODPAC 21-clip self-match: SKANN vs tonal', '',
          'Constructed half-split ground truth (gallery = clean first half; query = last half, degraded). '
          f'Collapsed columns merge provenance-asserted same-source clips to one identity: **{cluster_lbl}** '
          '(from external provenance, not from SKANN cosine). Raw is never overwritten. '
          'N=21 genuine + ~420 impostor per condition, so counts are coarse — read the **direction**.', '',
          '## rank-1 (raw → collapsed)', '',
          '| condition | ' + ' | '.join(n for n, _ in have) + ' |',
          '|' + '---|' * (len(have) + 1)]
    for key, label in CONDS:
        cells = [f"{R[n][key]['raw']['rank1']:.3f} → {R[n][key]['col']['rank1']:.3f} ({R[n][key]['col']['hits']}/21)"
                 for n, _ in have]
        md.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    md += ['', '## open-set AUC (raw → collapsed)', '',
           '| condition | ' + ' | '.join(n for n, _ in have) + ' |',
           '|' + '---|' * (len(have) + 1)]
    for key, label in CONDS:
        cells = [f"{R[n][key]['raw']['auc']:.3f} → {R[n][key]['col']['auc']:.3f}" for n, _ in have]
        md.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    md += ['', '## false-alarm @ F1-opt (raw → collapsed)', '',
           '| condition | ' + ' | '.join(n for n, _ in have) + ' |',
           '|' + '---|' * (len(have) + 1)]
    for key, label in CONDS:
        cells = [f"{R[n][key]['raw']['fa']:.3f} → {R[n][key]['col']['fa']:.3f}" for n, _ in have]
        md.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    if tonal is not None:
        order = tonal['clip_order']
        md += ['', '## tonal fairness (SKANN has no no-signature clips)', '',
               '| condition | no-signature | rank-1 (all) | rank-1 (clips with a signature) |',
               '|---|---|---|---|']
        for key, label in CONDS:
            S = np.array(tonal['self_match'][key]['S'])
            nsig = no_sig_rows(S); rr, kept = restricted_rank1(S, order, CLUSTERS)
            rr_s = f'{rr:.3f} ({kept} clips)' if rr is not None else 'n/a'
            md.append(f'| {label} | {len(nsig)}/21 | {R["TONAL"][key]["col"]["rank1"]:.3f} | {rr_s} |')
    md += ['', '_Numbers only. The tonal side was rebuilt with a TPSW (frequency-whitened) normaliser '
           'replacing the prior median-over-time background; SKANN is unchanged. Read the head-to-head '
           'fresh from these figures. N is small (21) — directional, not exact._']

    out = STORE_DIR / 'disc5_navy_comparison.md'
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')

    if tonal is None:
        print('\n(tonal JSON not found -- SKANN-only; run disc5_score_navy_tonal.py for the full table)')
    print(f'\nwrote {out}')

if __name__ == '__main__':
    main()
