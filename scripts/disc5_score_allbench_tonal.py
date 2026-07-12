# disc5_score_allbench_tonal.py
# TONAL side of the all-benchmark SKANN-vs-tonal comparison. Local CPU, no GPU -- the
# counterpart to DISC5_Allbench_SKANN.ipynb (Colab GPU), exactly the NODPAC-21 pattern
# (disc5_score_navy_tonal.py + disc5_skann_navy_embed_score.ipynb) extended to all four
# benchmarks. The two sides write line-for-line comparable JSONs anchored to the SAME
# passage orders; this script ALSO performs the comparison + z-fusion automatically when
# the SKANN JSON ({TAG}_allbench_skann__{MODEL}.json) is present in the embedding store.
#
# Benchmarks (tonal side):
#   [1] IARA val + [2] ShipsEar val : TPSW lines extracted HERE from the local build WAVs
#       (self-contained, like the navy tonal scorer), pooled clip -> passage
#       (passage = (vessel, session) per D44, via {TAG}_seg_meta.csv), scored
#       leave-one-passage-out against the full gallery.
#   [3] NODPAC-21 : INGESTED from the post-TPSW {TAG}_navy_tonal.json (S matrices per
#       condition); the {MV_12,13,14} identity collapse is applied at comparison time.
#   [4] ONC : INGESTED from onc_tonal_lines.json (post-TPSW rerun); the 38x97 S matrix is
#       rebuilt from the line sets for fusion. Full-97 and holdout-7 tables.
# TPSW discipline: hard-aborts if any ingested tonal artefact does not self-declare tpsw.
# Normaliser is the NAVY-scorer TPSW variant verbatim (win=8.0, guard=1.5, alpha=3.0).
#
# Canonical protocol (v2 patch): val benchmarks are scored SOURCE-PURE -- IARA queries
# see only the 114 other IARA passages, ShipsEar only its 13 -- closing the
# mixed-gallery artefact (ShipsEar AUC 0.849->0.559 was hardware discrimination).
# Printed comparison + compare JSON use the canonical metric triplet rank-1/AUC/
# median-rank, benchmarks ordered by SKANN rank-1 descending; ONC holdout-7 stays in
# the JSON (internal) but only full-97 is printed.
# Outputs (embedding store), ALL keyed by MODEL so ep21 / ftONC / ft2 never clobber:
#   {TAG}_allbench_tonal__{MODEL}.json   (tables + S matrices + orders)
#   {TAG}_allbench_compare__{MODEL}.json (printed side-by-side, when the SKANN JSON exists)
# It READS the SKANN side from {TAG}_allbench_skann__{MODEL}.json (matches the notebook's
# Cell 5 name) and asserts the file's self-declared model == MODEL, so a wrong/renamed file
# fails loudly instead of silently comparing the tonal lines against the wrong SKANN run.
# Set MODEL at the top to pick which SKANN run to score against. The tonal lines themselves
# are model-independent (identical every run); the suffix just keeps each run self-contained.

import csv, json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

# ---- paths (edit to your machine) -------------------------------------------------------
BUILD_ROOT = Path(r'C:\DISC5\disc5_build')
STORE_DIR  = Path(r'G:\My Drive\DISC5_Checkpoints\embedding_store')
ONC_BENCH  = Path(r'G:\My Drive\SKANN_SSL\onc_data\benchmark')
SPLIT_CSV  = Path(r'C:\DISC5\onc_finetune_split.csv')
TAG        = 'disc5_arcface_8k'
MODEL      = 'ft2'          # which SKANN run to compare against; suffixes ALL outputs.
                            # Must match the 'model' field inside the SKANN JSON (asserted).

# ---- LOFAR / tonal constants (verbatim from disc5_score_navy_tonal.py, post-TPSW) --------
SR = 8000
STFT_WINDOW_SEC, STFT_OVERLAP_FRAC, STFT_NFFT_MULT = 4.0, 0.75, 2
BACKGROUND_PERCENTILE, TONAL_THRESHOLD_DB, TONAL_MIN_PERSIST_SEC = 50, 8.0, 10
TPSW_WIN_HZ, TPSW_GUARD_HZ, TPSW_ALPHA = 8.0, 1.5, 3.0
TONAL_FREQ_MIN_HZ, TONAL_FREQ_MAX_HZ, TONAL_BAND_WIDTH_HZ = 3.5, 2000, 2.0
TOP_K, TOL_HZ = 20, 1.0
NORMALIZATION = f'tpsw_splitwindow_freq(win={TPSW_WIN_HZ}Hz,guard={TPSW_GUARD_HZ}Hz,alpha={TPSW_ALPHA})'
NODPAC_COLLAPSE = {'MV__MV_12', 'MV__MV_13', 'MV__MV_14'}
CONDS = ['clean', 'noise', 'speed', 'speed_noise']
FUSION_NOTE = 'per_query_impostor_z(mean of z_skann,z_tonal)'

# ===== LOFAR: verbatim from disc5_score_navy_tonal.py (post-TPSW) =========================
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import uniform_filter1d
import soundfile as sf

def compute_stft(audio, sr):
    nperseg = int(STFT_WINDOW_SEC * sr); noverlap = int(nperseg * STFT_OVERLAP_FRAC)
    nfft = nperseg * STFT_NFFT_MULT
    freqs, times, Sxx = scipy_spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap,
                                          nfft=nfft, mode='psd', scaling='density')
    return freqs, times, Sxx

def normalise_spectrogram(freqs, times, Sxx):
    # TPSW: two-pass split-window background over FREQUENCY (the standard LOFAR/DEMON whitener).
    fr = freqs[1] - freqs[0]
    M = max(1, int(round(TPSW_WIN_HZ / fr)))     # outer half-window (bins)
    G = max(1, int(round(TPSW_GUARD_HZ / fr)))   # centre guard half-width (excludes the line itself)

    def split_mean(X):                           # mean over [+-(G..M+G)] excluding centre +-G, per frame
        full = uniform_filter1d(X, size=2 * (M + G) + 1, axis=0, mode='reflect') * (2 * (M + G) + 1)
        cent = uniform_filter1d(X, size=2 * G + 1, axis=0, mode='reflect') * (2 * G + 1)
        return (full - cent) / max(2 * M, 1)

    bg1 = np.maximum(split_mean(Sxx), 1e-20)
    bg2 = np.maximum(split_mean(np.minimum(Sxx, TPSW_ALPHA * bg1)), 1e-20)  # 2nd pass, tonal-clipped
    spec_db = 10 * np.log10(np.maximum(Sxx / bg2, 1e-10))
    return spec_db, bg2

def detect_persistent_tonals(spec_db, freqs, times):
    freq_mask = (freqs >= TONAL_FREQ_MIN_HZ) & (freqs <= TONAL_FREQ_MAX_HZ)
    f_sub = freqs[freq_mask]; s_sub = spec_db[freq_mask, :]
    if len(times) < 2 or len(f_sub) < 2:
        return []
    dt = times[1] - times[0]; min_frames = max(1, int(TONAL_MIN_PERSIST_SEC / dt))
    freq_res = freqs[1] - freqs[0]; band_width_bins = max(1, int(TONAL_BAND_WIDTH_HZ / freq_res))
    n_bands = len(f_sub) // band_width_bins
    band_freqs = []; band_spec = np.zeros((n_bands, s_sub.shape[1]))
    for bi in range(n_bands):
        sb = bi * band_width_bins; eb = min(sb + band_width_bins, len(f_sub))
        band_spec[bi, :] = np.max(s_sub[sb:eb, :], axis=0)
        band_freqs.append(float(f_sub[sb + np.argmax(np.mean(s_sub[sb:eb, :], axis=1))]))
    above = band_spec > TONAL_THRESHOLD_DB; persistent = []
    for bi in range(n_bands):
        row = above[bi, :]; start_idx = None
        for ti in range(len(row)):
            if row[ti] and start_idx is None:
                start_idx = ti
            elif not row[ti] and start_idx is not None:
                if ti - start_idx >= min_frames:
                    persistent.append(dict(freq_hz=band_freqs[bi],
                                           mean_db=float(np.mean(band_spec[bi, start_idx:ti]))))
                start_idx = None
        if start_idx is not None and len(row) - start_idx >= min_frames:
            persistent.append(dict(freq_hz=band_freqs[bi],
                                   mean_db=float(np.mean(band_spec[bi, start_idx:]))))
    return persistent

def top_lines(persistent, k=TOP_K):
    best = {}
    for ln in persistent:
        f = ln['freq_hz']
        if f not in best or ln['mean_db'] > best[f]:
            best[f] = ln['mean_db']
    return sorted(best.items(), key=lambda x: -x[1])[:k]

def lofar_lines(y):
    if len(y) < int(STFT_WINDOW_SEC * SR):
        return []
    freqs, times, Sxx = compute_stft(y.astype('float64'), SR)
    spec_db, _ = normalise_spectrogram(freqs, times, Sxx)
    return top_lines(detect_persistent_tonals(spec_db, freqs, times))

def tonal_score(a, b, tol=TOL_HZ):
    if not a or not b:
        return 0.0
    def frac(q, g):
        gf = np.array([f for f, _ in g]); den = sum(w for _, w in q)
        if den <= 0:
            return 0.0
        num = sum(w for f, w in q if np.min(np.abs(gf - f)) <= tol)
        return num / den
    return 0.5 * (frac(a, b) + frac(b, a))

def merge_lines(line_lists, tol=TOL_HZ, k=TOP_K):
    alll = sorted((x for ll in line_lists for x in ll), key=lambda x: -x[1])
    kept = []
    for f, d in alll:
        if all(abs(f - kf) > tol for kf, _ in kept):
            kept.append((f, d))
        if len(kept) >= k:
            break
    return kept
# ==========================================================================================

# ===== metrics + protocol (shared with the SKANN notebook) ================================
def eer_from_scores(gen, imp):
    scores = np.concatenate([gen, imp]); labels = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    order = np.argsort(-scores); labels = labels[order]
    fnr = 1 - np.cumsum(labels) / max(labels.sum(), 1)
    fpr = np.cumsum(1 - labels) / max((1 - labels).sum(), 1)
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)

def auc_rank(gen, imp):
    allv = np.concatenate([gen, imp])
    vals, inv, cnts = np.unique(allv, return_inverse=True, return_counts=True)
    order = np.argsort(allv, kind='mergesort'); r = np.empty(len(allv)); r[order] = np.arange(1, len(allv) + 1)
    sums = np.zeros(len(vals)); np.add.at(sums, inv, r); r = (sums / cnts)[inv]
    return float((r[:len(gen)].sum() - len(gen) * (len(gen) + 1) / 2) / (len(gen) * len(imp)))

def zfuse_row(s1, s2):
    def z(v):
        sd = v.std()
        return (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
    return 0.5 * (z(s1) + z(s2))

def loo_table(SIM, vessels, qmask):
    """Leave-one-passage-out vs full gallery. Genuine = best same-vessel other passage."""
    P = len(vessels); vs = np.asarray(vessels); cnt = Counter(vs.tolist())
    gen_all, imp_all, ranks, hits = [], [], [], []
    for i in range(P):
        if not qmask[i] or cnt[vs[i]] < 2:
            continue
        row = SIM[i].copy(); row[i] = -np.inf
        same = (vs == vs[i]); same[i] = False
        order = np.argsort(-row)
        rank = int(np.where(vs[order] == vs[i])[0][0]) + 1
        gen_all.append(float(row[same].max()))
        imp_all.append(row[~same & (np.arange(P) != i)])
        ranks.append(rank); hits.append(rank == 1)
    if not ranks:
        return None
    gen_all = np.array(gen_all); imp_all = np.concatenate(imp_all); ranks = np.array(ranks)
    return dict(n_queries=len(ranks), rank1=round(float(np.mean(hits)), 4),
                hit5=round(float((ranks <= 5).mean()), 4), hit10=round(float((ranks <= 10).mean()), 4),
                median_rank_of_genuine=float(np.median(ranks)),
                auc=round(auc_rank(gen_all, imp_all), 4),
                eer=round(eer_from_scores(gen_all, imp_all), 4))

def gq_table(S, q_mmsi, g_mmsi, sub=None):
    """Gallery/query table (ONC style). S: [Q,G]; genuine cols = same mmsi."""
    gen_all, imp_all, ranks = [], [], []
    for qi in range(len(q_mmsi)):
        if sub is not None and q_mmsi[qi] not in sub:
            continue
        gcols = np.where(np.asarray(g_mmsi) == q_mmsi[qi])[0]
        order = np.argsort(-S[qi])
        ranks.append(int(np.where(np.isin(order, gcols))[0][0]) + 1)
        gen_all.append(S[qi, gcols].max()); imp_all.append(np.delete(S[qi], gcols))
    gen_all = np.array(gen_all); imp_all = np.concatenate(imp_all); ranks = np.array(ranks)
    return dict(n_queries=len(ranks), rank1=round(float((ranks == 1).mean()), 4),
                hit5=round(float((ranks <= 5).mean()), 4), hit10=round(float((ranks <= 10).mean()), 4),
                median_rank_of_genuine=float(np.median(ranks)),
                auc=round(auc_rank(gen_all, imp_all), 4),
                eer=round(eer_from_scores(gen_all, imp_all), 4))

def collapse_navy(S, order):
    """{MV_12,13,14} identity collapse; navy conventions (nearest-impostor AUC)."""
    ident = [('MVGROUP' if c in NODPAC_COLLAPSE else c) for c in order]
    uids = sorted(set(ident)); cols = {u: [j for j, c in enumerate(ident) if c == u] for u in uids}
    gen, nimp, hits = [], [], []
    for i in range(len(order)):
        iscore = {u: float(S[i, cols[u]].max()) for u in uids}
        gen.append(iscore[ident[i]])
        nimp.append(max(v for u, v in iscore.items() if u != ident[i]))
        hits.append(max(iscore, key=iscore.get) == ident[i])
    gen, nimp = np.array(gen), np.array(nimp)
    g, m = gen[:, None], nimp[None, :]
    return float(np.mean(hits)), float((g > m).mean() + 0.5 * (g == m).mean())

def nodpac_med_rank(S, order):
    """Median rank of the query's own identity among the 19 collapsed identities."""
    ident = [('MVGROUP' if c in NODPAC_COLLAPSE else c) for c in order]
    uids = sorted(set(ident)); cols = {u: [j for j, c in enumerate(ident) if c == u] for u in uids}
    ranks = []
    for i in range(len(order)):
        sc = {u: float(S[i, cols[u]].max()) for u in uids}
        ranks.append(sorted(sc, key=lambda u: -sc[u]).index(ident[i]) + 1)
    return float(np.median(ranks))

def require_tpsw(norm, what):
    if 'tpsw' not in str(norm).lower():
        raise SystemExit(f'ABORT: {what} is not post-TPSW (normalization={norm!r}).')
# ==========================================================================================

def main():
    out = dict(method='tonal', normalization=NORMALIZATION, fusion=FUSION_NOTE, benchmarks={})

    # ---- [1+2] IARA val + ShipsEar val: extract TPSW lines from local WAVs --------------
    csv.field_size_limit(10 ** 8)
    with open(STORE_DIR / f'{TAG}_seg_meta.csv', encoding='utf-8') as f:
        meta = [m for m in csv.DictReader(f) if m['split'] == 'val']
    pass_info, clip2pass = {}, {}
    for m in meta:
        pass_info[m['passage']] = (m['vessel_id'], m['source'].upper())
        clip2pass[m['parent_clip']] = m['passage']
    passages = sorted(pass_info)
    ves = np.array([pass_info[p][0] for p in passages])
    src = np.array([pass_info[p][1] for p in passages])
    P = len(passages)

    stem2path = {}
    for srcname, sub in (('IARA', 'IARA'), ('SHIPSEAR', 'ShipsEar')):
        root = BUILD_ROOT / sub
        for w in (root.rglob('*.wav') if root.exists() else []):
            stem2path[(srcname, w.stem)] = w

    print(f'[val] extracting TPSW lines for {len(clip2pass)} clips -> {P} passages')
    pass_clips, missing = defaultdict(list), []
    for n, (clip, pas) in enumerate(sorted(clip2pass.items()), 1):
        srcname = pass_info[pas][1]
        wav = stem2path.get((srcname, clip))
        if wav is None:
            missing.append(clip); continue
        y, fs = sf.read(str(wav))
        if y.ndim > 1:
            y = y.mean(axis=1)
        pass_clips[pas].append(lofar_lines(y - y.mean()))
        if n % 25 == 0 or n == len(clip2pass):
            print(f'  {n}/{len(clip2pass)} clips')
    if missing:
        print(f'  WARNING: {len(missing)} val clips without WAVs (first: {missing[:3]})')
    TL = [merge_lines(pass_clips.get(p, [])) for p in passages]

    S_val = np.zeros((P, P))
    for i in range(P):
        for j in range(i + 1, P):
            S_val[i, j] = S_val[j, i] = tonal_score(TL[i], TL[j])
    for bench, m in (('iara_val', src == 'IARA'), ('shipsear', src == 'SHIPSEAR')):
        idx = np.where(m)[0]                       # SOURCE-PURE gallery: same-source passages only
        Sp = S_val[np.ix_(idx, idx)]
        out['benchmarks'][bench] = dict(
            protocol='leave-one-passage-out, SOURCE-PURE gallery, passage=(vessel,session) per D44',
            n_candidates=int(len(idx) - 1),
            passage_order=passages, vessels=ves.tolist(), source_mask=m.tolist(),
            table=loo_table(Sp, ves[idx], np.ones(len(idx), bool)))
    out['benchmarks']['_val_S'] = np.round(S_val, 6).tolist()   # full matrix kept for fusion slicing

    # ---- [3] NODPAC-21: ingest the post-TPSW navy tonal JSON ----------------------------
    navy = json.load(open(STORE_DIR / f'{TAG}_navy_tonal.json'))
    require_tpsw(navy.get('normalization', ''), f'{TAG}_navy_tonal.json')
    nod = dict(protocol='half-split, 4 conditions, MV_12+13+14 collapse, navy conventions',
               clip_order=navy['clip_order'], conditions={})
    for cond in CONDS:
        S = np.array(navy['self_match'][cond]['S'])
        r1, auc = collapse_navy(S, navy['clip_order'])
        nod['conditions'][cond] = dict(rank1=round(r1, 4), auc=round(auc, 4),
                                       S=np.round(S, 6).tolist())
    out['benchmarks']['nodpac21'] = nod

    # ---- [4] ONC: rebuild 38x97 S from the post-TPSW line sets --------------------------
    import pandas as pd
    require_tpsw(json.load(open(ONC_BENCH / 'onc_tonal_results.json')).get('normalization', ''),
                 'onc_tonal_results.json')
    lines = {u: [tuple(x) for x in v]
             for u, v in json.load(open(ONC_BENCH / 'onc_tonal_lines.json')).items()}
    man = pd.read_csv(ONC_BENCH / 'onc_eval_tensor_manifest.csv')
    info = man.groupby('passage_uid').agg(mmsi=('vessel_mmsi', 'first'),
                                          role=('role', 'first')).reset_index()
    gal = info[info.role == 'gallery'].sort_values('passage_uid').reset_index(drop=True)
    qry = info[info.role == 'query'].sort_values('passage_uid').reset_index(drop=True)
    S_onc = np.zeros((len(qry), len(gal)))
    for qi, qu in enumerate(qry.passage_uid):
        for gi, gu in enumerate(gal.passage_uid):
            S_onc[qi, gi] = tonal_score(lines[qu], lines[gu])
    holdout = set()
    if SPLIT_CSV.exists():
        sp = pd.read_csv(SPLIT_CSV)
        holdout = set(sp[sp.side.str.lower().str.startswith('hold')].vessel_mmsi)
    onc = dict(protocol='cross-passage, 97-gallery; full + holdout-7 subset',
               query_uids=qry.passage_uid.tolist(), gallery_uids=gal.passage_uid.tolist(),
               query_mmsi=[int(x) for x in qry.mmsi], gallery_mmsi=[int(x) for x in gal.mmsi],
               S=np.round(S_onc, 6).tolist(),
               table_full97=gq_table(S_onc, qry.mmsi.values, gal.mmsi.values),
               table_holdout7=(gq_table(S_onc, qry.mmsi.values, gal.mmsi.values, holdout)
                               if holdout else None))
    out['benchmarks']['onc'] = onc

    tonal_path = STORE_DIR / f'{TAG}_allbench_tonal__{MODEL}.json'
    with open(tonal_path, 'w') as f:
        json.dump(out, f)
    print(f'\nwrote {tonal_path}')

    # ---- comparison + fusion (runs automatically when the SKANN JSON exists) ------------
    skann_path = STORE_DIR / f'{TAG}_allbench_skann__{MODEL}.json'
    if not skann_path.exists():
        print(f'NOTE: {skann_path.name} not found -- run DISC5_Allbench_SKANN_ftONC.ipynb in '
              f'Colab with MODEL={MODEL!r}, then re-run this script for the side-by-side + '
              f'fusion tables.')
        return
    sk = json.load(open(skann_path))
    model = sk.get('model', '?')
    assert model == MODEL, (f'SKANN JSON self-declares model={model!r} but MODEL={MODEL!r} -- '
                            f'wrong or mis-renamed file ({skann_path.name}).')
    cmp_out = dict(model=model, fusion=FUSION_NOTE,
                   protocol_notes='val source-pure; nodpac MV-collapsed navy conventions; '
                                  'onc_holdout7 internal-only (not printed)',
                   benchmarks={})

    def show(bench, ncand, rows):
        print(f'\n--- {bench} ({ncand} candidates) ---')
        print(f'{"method":14s} {"n":>4s} {"rank1":>7s} {"auc":>7s} {"med.rank":>9s}')
        for name, x in rows:
            print(f'{name:14s} {x["n_queries"]:4d} {x["rank1"]:7.3f} {x["auc"]:7.3f} '
                  f'{x["median_rank_of_genuine"]:9.1f}')

    ordered = []   # (bench, ncand, rows) in canonical order

    # nodpac: per condition, skann/tonal/fusion with med.rank, ordered by skann rank-1 desc
    tn, sn = out['benchmarks']['nodpac21'], sk['benchmarks']['nodpac21']
    assert sn['clip_order'] == tn['clip_order'], 'nodpac: clip order mismatch'
    corder = tn['clip_order']
    nod_rows = {}
    for cond in CONDS:
        S_sk = np.array(sn['conditions'][cond]['S']); S_to = np.array(tn['conditions'][cond]['S'])
        S_fu = np.vstack([zfuse_row(S_sk[i], S_to[i]) for i in range(len(S_sk))])
        row = {}
        for name, S in ((f'skann_{model}', S_sk), ('tonal', S_to), ('fusion', S_fu)):
            r1, auc = collapse_navy(S, corder)
            row[name] = dict(n_queries=len(corder), rank1=round(r1, 4), auc=round(auc, 4),
                             median_rank_of_genuine=nodpac_med_rank(S, corder))
        nod_rows[cond] = row
    for cond in sorted(CONDS, key=lambda c: -nod_rows[c][f'skann_{model}']['rank1']):
        ordered.append((f'nodpac_{cond}', 19, list(nod_rows[cond].items())))
    cmp_out['benchmarks']['nodpac21'] = nod_rows

    # val: source-pure galleries, fusion within the pure slice
    S_skv, S_tov = np.array(sk['benchmarks']['_val_S']), np.array(out['benchmarks']['_val_S'])
    assert sk['benchmarks']['iara_val']['passage_order'] == out['benchmarks']['iara_val']['passage_order']
    for bench in ('iara_val', 'shipsear'):
        m = np.array(out['benchmarks'][bench]['source_mask'])
        idx = np.where(m)[0]
        Ss, St = S_skv[np.ix_(idx, idx)], S_tov[np.ix_(idx, idx)]
        Sf = np.zeros_like(Ss)
        for i in range(len(Ss)):
            mk = np.arange(len(Ss)) != i
            Sf[i, mk] = zfuse_row(Ss[i, mk], St[i, mk]); Sf[i, i] = -np.inf
        v = np.array(out['benchmarks'][bench]['vessels'])[idx]
        qall = np.ones(len(idx), bool)
        rows = [(f'skann_{model}', loo_table(Ss, v, qall)), ('tonal', loo_table(St, v, qall)),
                ('fusion', loo_table(Sf, v, qall))]
        cmp_out['benchmarks'][bench] = {k: x for k, x in rows}
        ordered.append((bench, len(idx) - 1, rows))

    # onc: full-97 printed; holdout-7 computed into the JSON only
    to_, so_ = out['benchmarks']['onc'], sk['benchmarks']['onc']
    assert so_['query_uids'] == to_['query_uids'] and so_['gallery_uids'] == to_['gallery_uids']
    S_sk = np.array(so_['S']); S_to = np.array(to_['S'])
    S_fu = np.vstack([zfuse_row(S_sk[i], S_to[i]) for i in range(len(S_sk))])
    qm, gm = np.array(to_['query_mmsi']), np.array(to_['gallery_mmsi'])
    for bench, sub, printed in (('onc_full97', None, True), ('onc_holdout7', holdout or None, False)):
        if bench == 'onc_holdout7' and not holdout:
            continue
        rows = [(f'skann_{model}', gq_table(S_sk, qm, gm, sub)), ('tonal', gq_table(S_to, qm, gm, sub)),
                ('fusion', gq_table(S_fu, qm, gm, sub))]
        cmp_out['benchmarks'][bench] = {k: x for k, x in rows}
        if printed:
            ordered.append((bench, 97, rows))

    # print in canonical order: benchmarks sorted by skann rank-1 descending
    ordered.sort(key=lambda r: -dict(r[2])[f'skann_{model}']['rank1'])
    for bench, ncand, rows in ordered:
        show(bench, ncand, rows)

    cmp_path = STORE_DIR / f'{TAG}_allbench_compare__{MODEL}.json'
    with open(cmp_path, 'w') as f:
        json.dump(cmp_out, f, indent=2)
    print(f'\nwrote {cmp_path}')

if __name__ == '__main__':
    main()
