# disc5_score_onc_tonal.py
# Scores the LOFAR tonal method on the ONC SCVIP cross-passage benchmark, producing
# onc_tonal_results.json in the SAME format as the SKANN onc_eval_results.json so the two
# are directly comparable (the NODPAC-style SKANN-vs-tonal comparison).
# v2 (post-TPSW): normalise_spectrogram is now the TPSW split-window-over-FREQUENCY whitener,
# VERBATIM from the corrected disc5_score_navy_tonal.py (win=8.0 Hz, guard=1.5 Hz, alpha=3.0
# -- the NAVY-scorer variant, NOT the screening pipeline's win=10/gap=2/threshold-clip
# variant in disc5_build_onc_benchmark.py). The previous run of this script used the
# inherited percentile-over-TIME background, which suppresses steady tonals (a line present
# for the whole record equals its own temporal percentile -> ~0 dB, undetectable); the old
# onc_tonal_results.json is therefore a pre-TPSW artefact and is SUPERSEDED by this rerun.
# The output JSON self-declares its normaliser so the two runs can never be confused.
# All other machinery verbatim from disc5_score_navy_tonal.py: same LOFAR constants (4 s
# STFT, 75% overlap, 8 dB / 10 s persistence, 3.5-2000 Hz, 2 Hz bands), same top_lines
# (TOP_K=20 strongest distinct lines), same tonal_score symmetric weighted line-match at
# TOL_HZ=1.0.
# Per passage: the staged raw 5-s segment WAVs (benchmark/wav/<uid>/, the tensors' audit
# trail) are concatenated in time order and mean-subtracted, lines are extracted once, and
# queries are scored against the full 97-passage gallery. CPU, a few minutes.
# Outputs: onc_tonal_results.json + onc_tonal_trials.csv + onc_tonal_lines.json (per-passage
# line sets, for fusion later).

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import uniform_filter1d

# ===== LOFAR constants + functions: verbatim from disc5_score_navy_tonal.py (post-TPSW) ===
SR = 8000
STFT_WINDOW_SEC, STFT_OVERLAP_FRAC, STFT_NFFT_MULT = 4.0, 0.75, 2
BACKGROUND_PERCENTILE, TONAL_THRESHOLD_DB, TONAL_MIN_PERSIST_SEC = 50, 8.0, 10
# TPSW whitening (over FREQUENCY) replaces the inherited median-over-TIME background. The old
# normaliser divided each frequency by its own temporal median, which suppresses STEADY tonals
# (a line on the whole record == its own median -> ~0 dB, undetectable) -- the bug that hid MV_13's
# tonals entirely and dropped MV_3 self-match to 0.234. TPSW estimates a per-frame broadband floor
# from a split window over frequency (centre guard excluded), so steady narrowband lines stand
# proud. Verified on NODPAC-21: MV_3 self-match 0.234 -> 0.79, MV_13 0.000 -> 1.000.
TPSW_WIN_HZ, TPSW_GUARD_HZ, TPSW_ALPHA = 8.0, 1.5, 3.0
TONAL_FREQ_MIN_HZ, TONAL_FREQ_MAX_HZ, TONAL_BAND_WIDTH_HZ = 3.5, 2000, 2.0
TOP_K = 20
TOL_HZ = 1.0    # line-match tolerance (NODPAC frequency proximity), same as the val pass

def compute_stft(audio, sr):
    nperseg = int(STFT_WINDOW_SEC * sr)
    noverlap = int(nperseg * STFT_OVERLAP_FRAC)
    nfft = nperseg * STFT_NFFT_MULT
    freqs, times, Sxx = scipy_spectrogram(
        audio, fs=sr, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, mode='psd', scaling='density')
    return freqs, times, Sxx

def normalise_spectrogram(freqs, times, Sxx):
    # TPSW: two-pass split-window background over FREQUENCY (the standard LOFAR/DEMON whitener).
    # Returns dB above the local broadband floor, and that floor. Replaces the median-over-time
    # background, which suppressed steady tonals (see TPSW note above).
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
    f_sub = freqs[freq_mask]
    s_sub = spec_db[freq_mask, :]
    if len(times) < 2 or len(f_sub) < 2:
        return []
    dt = times[1] - times[0]
    min_frames = max(1, int(TONAL_MIN_PERSIST_SEC / dt))
    freq_res = freqs[1] - freqs[0]
    band_width_bins = max(1, int(TONAL_BAND_WIDTH_HZ / freq_res))
    n_bands = len(f_sub) // band_width_bins
    band_freqs = []
    band_spec = np.zeros((n_bands, s_sub.shape[1]))
    for bi in range(n_bands):
        sb = bi * band_width_bins
        eb = min(sb + band_width_bins, len(f_sub))
        band_spec[bi, :] = np.max(s_sub[sb:eb, :], axis=0)
        band_freqs.append(float(f_sub[sb + np.argmax(np.mean(s_sub[sb:eb, :], axis=1))]))
    above = band_spec > TONAL_THRESHOLD_DB
    persistent = []
    for bi in range(n_bands):
        row = above[bi, :]
        start_idx = None
        for ti in range(len(row)):
            if row[ti] and start_idx is None:
                start_idx = ti
            elif not row[ti] and start_idx is not None:
                if ti - start_idx >= min_frames:
                    persistent.append({'freq_hz': band_freqs[bi],
                                       'mean_db': float(np.mean(band_spec[bi, start_idx:ti]))})
                start_idx = None
        if start_idx is not None and len(row) - start_idx >= min_frames:
            persistent.append({'freq_hz': band_freqs[bi],
                               'mean_db': float(np.mean(band_spec[bi, start_idx:]))})
    return persistent

def top_lines(persistent, k=TOP_K):
    best = {}
    for ln in persistent:
        f = ln['freq_hz']
        if f not in best or ln['mean_db'] > best[f]:
            best[f] = ln['mean_db']
    return sorted(best.items(), key=lambda x: -x[1])[:k]   # [(freq, mean_db), ...]

def lofar_lines(y):
    if len(y) < int(STFT_WINDOW_SEC * SR):
        return []
    freqs, times, Sxx = compute_stft(y.astype('float64'), SR)
    spec_db, _ = normalise_spectrogram(freqs, times, Sxx)
    return top_lines(detect_persistent_tonals(spec_db, freqs, times))
# ===== end LOFAR verbatim ================================================================

# ===== tonal similarity: verbatim from disc5_eval_tonal_vs_embedding.py ==================
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
# ========================================================================================

def eer_from_scores(gen, imp):
    scores = np.concatenate([gen, imp]); labels = np.concatenate([np.ones_like(gen), np.zeros_like(imp)])
    order = np.argsort(-scores); labels = labels[order]
    fnr = 1 - np.cumsum(labels) / max(labels.sum(), 1)
    fpr = np.cumsum(1 - labels) / max((1 - labels).sum(), 1)
    i = int(np.argmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)

def auc_rank(gen, imp):
    allv = np.concatenate([gen, imp]); r = pd.Series(allv).rank().values
    return float((r[:len(gen)].sum() - len(gen) * (len(gen) + 1) / 2) / (len(gen) * len(imp)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'G:\My Drive\SKANN_SSL\onc_data')
    ap.add_argument('--bench', default=None)
    args = ap.parse_args()
    bench = Path(args.bench) if args.bench else Path(args.root) / 'benchmark'

    man = pd.read_csv(bench / 'onc_eval_tensor_manifest.csv')
    info = man.groupby('passage_uid').agg(mmsi=('vessel_mmsi', 'first'), role=('role', 'first'),
                                          cohort=('cohort', 'first')).reset_index()

    # ---- per-passage line extraction from the staged raw segment WAVs -------------------
    lines = {}
    for i, uid in enumerate(sorted(info.passage_uid), 1):
        pdir = bench / 'wav' / uid
        segs = sorted(p for p in pdir.glob('*.wav')
                      if '__cat' not in p.stem and '__listen' not in p.stem)
        y = np.concatenate([sf.read(str(s), dtype='float32')[0] for s in segs])
        y = y - y.mean()
        lines[uid] = lofar_lines(y)
        if i % 20 == 0 or i == len(info):
            print(f'  lines {i}/{len(info)}  ({uid}: {len(lines[uid])} lines)')
    with open(bench / 'onc_tonal_lines.json', 'w') as f:
        json.dump({u: lines[u] for u in lines}, f)

    # ---- score queries against the gallery (same trial structure as the SKANN scorer) ---
    gal = info[info.role == 'gallery'].reset_index(drop=True)
    qry = info[info.role == 'query'].reset_index(drop=True)
    S = np.zeros((len(qry), len(gal)))
    for qi, qrow in qry.iterrows():
        for gi, grow in gal.iterrows():
            S[qi, gi] = tonal_score(lines[qrow.passage_uid], lines[grow.passage_uid])

    trials = []
    for qi, qrow in qry.iterrows():
        gcols = np.where(gal.mmsi.values == qrow.mmsi)[0]
        order = np.argsort(-S[qi])
        trials.append({'query_uid': qrow.passage_uid, 'mmsi': qrow.mmsi, 'cohort': qrow.cohort,
                       'gen_score': float(S[qi, gcols].max()),
                       'best_imp_score': float(np.delete(S[qi], gcols).max()),
                       'rank1_hit': bool(gal.mmsi.values[order[0]] == qrow.mmsi),
                       'rank_of_genuine': int(np.where(np.isin(order, gcols))[0][0]) + 1,
                       'top_match_mmsi': int(gal.mmsi.values[order[0]])})
    T = pd.DataFrame(trials)
    T.to_csv(bench / 'onc_tonal_trials.csv', index=False)

    gen = T.gen_score.values
    imp = np.concatenate([np.delete(S[qi], np.where(gal.mmsi.values == qrow.mmsi)[0])
                          for qi, qrow in qry.iterrows()])

    def block(t, g_scores, i_scores):
        macro = t.groupby('mmsi').rank1_hit.mean().mean()
        return {'n_queries': int(len(t)), 'rank1': round(float(t.rank1_hit.mean()), 4),
                'rank1_macro_vessel': round(float(macro), 4),
                'median_rank_of_genuine': float(t.rank_of_genuine.median()),
                'eer': round(eer_from_scores(g_scores, i_scores), 4),
                'auc': round(auc_rank(g_scores, i_scores), 4),
                'genuine_mean': round(float(g_scores.mean()), 4),
                'impostor_mean': round(float(i_scores.mean()), 4)}

    out = {'normalization': f'tpsw_splitwindow_freq(win={TPSW_WIN_HZ}Hz,guard={TPSW_GUARD_HZ}Hz,alpha={TPSW_ALPHA})',
           'overall': block(T, gen, imp), 'gallery_size': int(len(gal)), 'by_cohort': {}}
    for c in ('core', 'annex'):
        tc = T[T.cohort == c]
        if len(tc):
            qidx = qry[qry.cohort == c].index
            ic = np.concatenate([np.delete(S[qi], np.where(gal.mmsi.values == qry.loc[qi].mmsi)[0]) for qi in qidx])
            out['by_cohort'][c] = block(tc, tc.gen_score.values, ic)
    with open(bench / 'onc_tonal_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f'\noutputs: {bench / "onc_tonal_results.json"}, {bench / "onc_tonal_trials.csv"}')

if __name__ == '__main__':
    main()
