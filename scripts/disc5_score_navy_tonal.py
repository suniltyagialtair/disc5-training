# disc5_score_navy_tonal.py
# Incumbent tonal-matching scoring of the NODPAC 21-clip out-of-domain test. Local CPU, no GPU.
# The tonal counterpart to disc5_score_navy_skann.py: SAME 21 clips, SAME constructed half-split
# ground truth, SAME four query conditions (clean | noise | speed | speed_noise), SAME scoring
# functions (score_condition / sweep copied verbatim) -> the two JSONs are line-for-line comparable.
#
# Reads the ORIGINAL WAVs (D:\Navy\...), not the encoder tensors. Gallery/query windows are read
#   EXACTLY from disc5_navy_manifest.csv (gallery_start_s/end_s, query_start_s/end_s) -- the same
#   boundaries SKANN's tensors were built from -- so both methods see identical audio portions:
#   gallery = first 5 min (clean reference); query = last 5 min, degraded (clips < 10 min: half/half).
# Perturbations applied at the HALF level (LOFAR needs the long waveform): speed = +-4% resample
# by the clip's manifest sign then read as 8 kHz; noise = real ambient bank @ 10 dB SNR. Same
# recipe as the tensors; per-segment ambient realizations are not bit-identical (the encoder eats
# z-normed 5 s segments, LOFAR eats the whole half) -- equivalent-degradation match, by design.
#
# Feature = NODPAC "most prominent tonals": top-20 persistent LOFAR lines, band capped 2 kHz.
# LOFAR functions are verbatim from disc5_extract_tonals.py / stage2_lofar.py EXCEPT the spectrogram
# normaliser: the inherited background was the median OVER TIME per frequency, which divides steady
# tonals down to ~0 dB and makes them undetectable (it silently turned MV_13's strong tonals into a
# "no signature" and dropped MV_3 self-match to 0.234). It is replaced here by TPSW -- a two-pass
# split-window background over FREQUENCY (the standard LOFAR/DEMON whitener) -- so steady narrowband
# lines are preserved. This is the ONLY change vs the prior tonal scorer; STFT, banding, persistence
# rule, 8 dB threshold, matching, and scoring functions are unchanged, so the JSON stays line-for-line
# comparable to the (untouched) SKANN JSON. Distance = symmetric strength-weighted +-1 Hz matched
# fraction (tonal_score verbatim from disc5_eval_tonal_vs_embedding.py). NO z-norm.
#
# Scores per condition: rank-1, open-set AUC = P(genuine sim > nearest-impostor sim), and a
# swept-threshold genuine-vs-impostor table -> TP/FP/FN/TN + F1-opt / EER points. Plus a 21x21
# gallery-vs-gallery DISCOVERY matrix (no ground truth). N=21 -> counts are coarse; read DIRECTION.
# Writes disc5_arcface_8k_navy_tonal.json to the embedding store.

import csv, json
from pathlib import Path
from math import gcd
from collections import defaultdict
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import uniform_filter1d

# ---- paths / params (edit to your machine; mirror disc5_prep_navy.py) --------------------
NAVY_FOLDERS = [Path(r'D:\Navy\Decommissioned'), Path(r'D:\Navy\MV')]
AMBIENT_DIRS = [Path(r'C:\DISC5\disc5_build\IARA\_ambient'),
                Path(r'C:\DISC5\disc5_build\ShipsEar\_ambient')]
CLIP_MANIFEST = Path(r'C:\DISC5\navy_tensors\disc5_navy_manifest.csv')   # sign + exact gallery/query windows
STORE_DIR     = Path(r'G:\My Drive\DISC5_Checkpoints\embedding_store')
TAG           = 'disc5_arcface_8k'

SR, MIN_TAIL = 8000, 8000
SNR_DB, SPEED_PCT, SEED, EPS = 10.0, 4.0, 1234, 1e-8
CONDS = ['clean', 'noise', 'speed', 'speed_noise']

# ---- LOFAR / tonal constants (verbatim from disc5_extract_tonals.py) ---------------------
TARGET_SR = 8000
STFT_WINDOW_SEC, STFT_OVERLAP_FRAC, STFT_NFFT_MULT = 4.0, 0.75, 2
BACKGROUND_PERCENTILE, TONAL_THRESHOLD_DB, TONAL_MIN_PERSIST_SEC = 50, 8.0, 10
# TPSW whitening (over FREQUENCY) replaces the inherited median-over-TIME background. The old
# normaliser divided each frequency by its own temporal median, which suppresses STEADY tonals
# (a line on the whole record == its own median -> ~0 dB, undetectable) -- the bug that hid MV_13's
# tonals entirely and dropped MV_3 self-match to 0.234. TPSW estimates a per-frame broadband floor
# from a split window over frequency (centre guard excluded), so steady narrowband lines stand
# proud. Verified: MV_3 self-match 0.234 -> 0.79, MV_13 0.000 -> 1.000.
TPSW_WIN_HZ, TPSW_GUARD_HZ, TPSW_ALPHA = 8.0, 1.5, 3.0
TONAL_FREQ_MIN_HZ, TONAL_FREQ_MAX_HZ, TONAL_BAND_WIDTH_HZ = 3.5, 2000, 2.0
TOP_K = 20
TOL_HZ = 1.0    # line-match tolerance (NODPAC frequency proximity), same as the val pass

# ===== LOFAR: verbatim from disc5_extract_tonals.py (= stage2_lofar.py),
# ===== EXCEPT normalise_spectrogram, which is corrected to TPSW (see TPSW note above) =====
def compute_stft(audio, sr):
    nperseg = int(STFT_WINDOW_SEC * sr); noverlap = int(nperseg * STFT_OVERLAP_FRAC)
    nfft = nperseg * STFT_NFFT_MULT
    freqs, times, Sxx = scipy_spectrogram(audio, fs=sr, nperseg=nperseg, noverlap=noverlap,
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
                                           mean_db=float(np.mean(band_spec[bi, start_idx:ti])),
                                           max_db=float(np.max(band_spec[bi, start_idx:ti]))))
                start_idx = None
        if start_idx is not None and len(row) - start_idx >= min_frames:
            persistent.append(dict(freq_hz=band_freqs[bi],
                                   mean_db=float(np.mean(band_spec[bi, start_idx:])),
                                   max_db=float(np.max(band_spec[bi, start_idx:]))))
    return sorted(persistent, key=lambda t: t['mean_db'], reverse=True)

def top_lines(persistent, k=TOP_K):
    best = {}
    for ln in persistent:
        f = ln['freq_hz']
        if f not in best or ln['mean_db'] > best[f]:
            best[f] = ln['mean_db']
    return sorted(best.items(), key=lambda x: -x[1])[:k]   # [(freq, mean_db), ...]
# ===== end LOFAR verbatim ================================================================

# ===== perturbations: verbatim from disc5_prep_navy.py ===================================
def load_audio_resample(path, sr_target):
    y, sr = sf.read(str(path), always_2d=True); y = y.mean(axis=1).astype('float32')
    if sr != sr_target:
        g = gcd(sr, sr_target); y = resample_poly(y, sr_target // g, sr // g).astype('float32')
    return y.astype('float32')

class AmbientBank:
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths if Path(p).exists()]; self._frames = {}
    def _f(self, p):
        if p not in self._frames:
            self._frames[p] = sf.info(str(p)).frames
        return self._frames[p]
    def chunk(self, n, rng):
        if not self.paths:
            return None
        p = self.paths[int(rng.integers(len(self.paths)))]; f = self._f(p)
        if f <= n:
            y, _ = sf.read(str(p), dtype='float32', always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            if len(y) == 0: return np.zeros(n, dtype='float32')
            y = np.tile(y, int(np.ceil(n / len(y))))[:n]
        else:
            start = int(rng.integers(0, f - n))
            y, _ = sf.read(str(p), start=start, frames=n, dtype='float32', always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
        return np.ascontiguousarray(y, dtype='float32')

def ambient_mix(y, bank, rng, snr_db=SNR_DB):
    amb = bank.chunk(len(y), rng) if bank is not None else None
    if amb is None:
        return y
    sp = np.mean(y * y) + EPS; target_np = sp / (10.0 ** (snr_db / 10.0))
    amb = amb * np.sqrt(target_np / (np.mean(amb * amb) + EPS))
    return (y + amb).astype('float32')

def speed_shift(y, sign):
    pct = SPEED_PCT / 100.0
    up, down = (1000, int(round(1000 * (1 + pct)))) if sign == 'approach' \
        else (1000, int(round(1000 * (1 - pct))))
    return resample_poly(y, up, down).astype('float32')
# ===== end perturbation verbatim ========================================================

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

# ===== self-match scoring: verbatim from disc5_score_navy_skann.py =======================
def sweep(gen, imp):
    taus = np.unique(np.concatenate([gen, imp])); P, N = len(gen), len(imp); rows = []
    for t in taus:
        TP = int((gen >= t).sum()); FN = P - TP
        FP = int((imp >= t).sum()); TN = N - FP
        prec = TP / (TP + FP) if TP + FP else 0.0
        rec = TP / P if P else 0.0
        fa = FP / N if N else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(tau=float(t), TP=TP, FP=FP, FN=FN, TN=TN,
                         precision=round(prec, 4), recall=round(rec, 4),
                         false_alarm=round(fa, 4), f1=round(f1, 4)))
    return rows

def score_condition(Sc):
    S = Sc; n = S.shape[0]
    rank1 = float((S.argmax(1) == np.arange(n)).mean())
    gen = np.diag(S).copy()
    off = S.copy(); np.fill_diagonal(off, -np.inf)
    nearest_imp = off.max(1)
    g, m = gen[:, None], nearest_imp[None, :]
    auc = float((g > m).mean() + 0.5 * (g == m).mean())
    imp_all = S[~np.eye(n, dtype=bool)]
    rows = sweep(gen, imp_all)
    f1_row = max(rows, key=lambda r: r['f1'])
    eer_row = min(rows, key=lambda r: abs((1 - r['recall']) - r['false_alarm']))
    return dict(rank1=round(rank1, 4), auc=round(auc, 4),
                genuine_mean=round(float(gen.mean()), 4), impostor_mean=round(float(imp_all.mean()), 4),
                f1_opt=f1_row, eer=eer_row, sweep=rows, S=np.round(S, 4).tolist())
# ========================================================================================

def lofar_lines(y):
    if len(y) < int(STFT_WINDOW_SEC * SR):
        return []
    freqs, times, Sxx = compute_stft(y.astype('float64'), SR)
    spec_db, _ = normalise_spectrogram(freqs, times, Sxx)
    return top_lines(detect_persistent_tonals(spec_db, freqs, times))

def main():
    # clip provenance (sign, split mode) keyed by clip_id, so conditions match the tensors
    with open(CLIP_MANIFEST, encoding='utf-8') as f:
        prov = {r['clip_id']: r for r in csv.DictReader(f)}

    bank = AmbientBank([p for d in AMBIENT_DIRS for p in (d.rglob('*.wav') if d.exists() else [])])
    print(f'ambient bank: {len(bank.paths)} clips')

    # gather clips in the same order disc5_prep_navy.py enumerated them (folder glob order)
    clips = []
    for folder in NAVY_FOLDERS:
        for w in sorted(folder.glob('*.wav')):
            cid = f"{folder.name}__{w.stem}".replace(' ', '_')
            clips.append((cid, w))
    assert len(clips) == 21, f'expected 21 clips, got {len(clips)}'
    order = [cid for cid, _ in clips]
    idx = {cid: i for i, cid in enumerate(order)}

    gal_lines = [None] * 21
    qry_lines = {cond: [None] * 21 for cond in CONDS}
    for ci, (cid, wav) in enumerate(clips):
        y = load_audio_resample(wav, SR); n = len(y); p = prov[cid]
        # EXACT same gallery/query windows SKANN's tensors were built from: read the boundaries
        # straight from disc5_navy_manifest.csv (not recomputed), so tonal and SKANN see the
        # identical audio portions per clip. (split_mode is implicit in those boundaries.)
        ga = max(0, int(round(float(p['gallery_start_s']) * SR))); gb = min(n, int(round(float(p['gallery_end_s']) * SR)))
        qa = max(0, int(round(float(p['query_start_s'])   * SR))); qb = min(n, int(round(float(p['query_end_s'])   * SR)))
        gal, qry = y[ga:gb], y[qa:qb]
        sign = p['doppler_sign']
        crng = np.random.default_rng(SEED + 1 + ci)

        gal_lines[idx[cid]] = lofar_lines(gal)                          # gallery: clean reference
        q_speed = speed_shift(qry, sign)
        qry_lines['clean'][idx[cid]]       = lofar_lines(qry)
        qry_lines['noise'][idx[cid]]       = lofar_lines(ambient_mix(qry, bank, crng))
        qry_lines['speed'][idx[cid]]       = lofar_lines(q_speed)
        qry_lines['speed_noise'][idx[cid]] = lofar_lines(ambient_mix(q_speed, bank, crng))
        print(f"  [{ci+1:2d}/21] {cid:24s} {sign:8s} gal_lines={len(gal_lines[idx[cid]])} "
              f"qry_clean={len(qry_lines['clean'][idx[cid]])}")

    # per-condition 21x21 tonal similarity (row=query clip, col=gallery clip)
    def sim_matrix(qlines):
        S = np.zeros((21, 21))
        for i in range(21):
            for j in range(21):
                S[i, j] = tonal_score(qlines[i], gal_lines[j])
        return S

    results = {cond: score_condition(sim_matrix(qry_lines[cond])) for cond in CONDS}

    # discovery: gallery vs gallery (clean refs), no ground truth
    SG = np.zeros((21, 21))
    for i in range(21):
        for j in range(21):
            SG[i, j] = tonal_score(gal_lines[i], gal_lines[j])
    off = SG.copy(); np.fill_diagonal(off, -np.inf)
    pairs = sorted(((float(off[i, j]), order[i], order[j])
                    for i in range(21) for j in range(i + 1, 21)), reverse=True)
    discovery = dict(clip_order=order, matrix=np.round(SG, 4).tolist(),
                     ranked_pairs=[dict(score=round(s, 4), a=a, b=b) for s, a, b in pairs])

    out = dict(tag=TAG, method='tonal', n_clips=21, clip_order=order,
               feature=f'top{TOP_K}_lofar_lines<= {TONAL_FREQ_MAX_HZ}Hz', distance='strength_weighted_pm1Hz',
               normalization=f'tpsw_splitwindow_freq(win={TPSW_WIN_HZ}Hz,guard={TPSW_GUARD_HZ}Hz,alpha={TPSW_ALPHA})',
               self_match=results, discovery=discovery)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STORE_DIR / f'{TAG}_navy_tonal.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print('\nTONAL NODPAC self-match (constructed half-split ground truth)')
    print('N=21 genuine + 420 impostor trials per condition -> counts are COARSE; read DIRECTION, not exact rates.\n')
    hdr = (f'{"condition":12s} {"rank-1":>11s} {"AUC":>6s} {"gen mu":>7s} {"imp mu":>7s} | '
           f'F1-opt @ tau  TP/FP/FN/TN')
    print(hdr); print('-' * len(hdr))
    for cond in CONDS:
        r = results[cond]; f = r['f1_opt']; hits = round(r['rank1'] * 21)
        print(f'{cond:12s} {r["rank1"]:.3f} ({hits:2d}/21) {r["auc"]:6.3f} '
              f'{r["genuine_mean"]:7.3f} {r["impostor_mean"]:7.3f} | '
              f'{f["tau"]:9.3f}  {f["TP"]:2d}/{f["FP"]:2d}/{f["FN"]:2d}/{f["TN"]:3d}')
    print('\ntop discovery pairs (gallery-vs-gallery, candidate same-vessel; NOT ground truth):')
    for d in discovery['ranked_pairs'][:8]:
        print(f'  {d["score"]:.3f}  {d["a"]}  <->  {d["b"]}')
    print(f'\nwrote {STORE_DIR / f"{TAG}_navy_tonal.json"}')
    print('compare against disc5_arcface_8k_navy_skann.json (same clips, same scoring).')

if __name__ == '__main__':
    main()
