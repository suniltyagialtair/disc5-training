# disc5_extract_tonals.py
# Phase-1 baseline feature pass for the tonal-vs-embedding comparison. SELF-CONTAINED:
# the LOFAR functions below are copied VERBATIM from the project's stage2_lofar.py
# (so the baseline is the real extractor), with DISC5 8 kHz constants -- EXCEPT
# normalise_spectrogram, which is corrected to TPSW (see note at the constants).
# Needs no config.py / stage2_lofar.py -- only numpy, scipy, soundfile. Pure CPU.
#
# v2 (post-TPSW): the original release of this script used the inherited
# percentile-over-TIME background, which suppresses steady tonals (a line present for the
# whole record equals its own temporal percentile -> ~0 dB, undetectable). Any
# {TAG}_tonal_lines.csv produced by the pre-TPSW version is SUPERSEDED; re-run this script.
# A sidecar {TAG}_tonal_lines_meta.json self-declares the normaliser so pre/post-TPSW
# line sets can never be confused.
#
# Settings (calibrated after a first run over-detected ~250x with the old 512 Hz config):
#   TONAL_FREQ_MAX_HZ = 2000  -- NODPAC report distinguishing tonals up to ~2 kHz.
#   TOP_K = 20                -- keep the 20 most prominent persistent lines per clip
#                                (dedup each band to its strongest first). This IS NODPAC's
#                                "most prominent tonals" procedure.
# Output: per-clip line set (<=TOP_K lines) for the comparison script to pool to passage.

import csv, argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import uniform_filter1d

# ---- paths (edit to your machine) -------------------------------------------------------
BUILD_ROOT = Path(r'C:\DISC5\disc5_build')
STORE_DIR  = Path(r'G:\My Drive\DISC5_Checkpoints\embedding_store')
TAG        = 'disc5_arcface_8k'
TARGET_SR  = 8000

# ---- DISC5 8 kHz constants --------------------------------------------------------------
STFT_WINDOW_SEC       = 4.0
STFT_OVERLAP_FRAC     = 0.75
STFT_NFFT_MULT        = 2
BACKGROUND_PERCENTILE = 50
TONAL_THRESHOLD_DB    = 8.0
TONAL_MIN_PERSIST_SEC = 10
TONAL_FREQ_MIN_HZ     = 3.5
TONAL_FREQ_MAX_HZ     = 2000     # NODPAC: distinguishing tonals observed up to ~2 kHz
TONAL_BAND_WIDTH_HZ   = 2.0
TOP_K                 = 20       # keep the 20 most prominent lines (NODPAC procedure)
# TPSW whitening (over FREQUENCY) replaces the inherited median-over-TIME background. The old
# normaliser divided each frequency by its own temporal median, which suppresses STEADY tonals
# (a line on the whole record == its own median -> ~0 dB, undetectable) -- the bug that hid MV_13's
# tonals entirely and dropped MV_3 self-match to 0.234. TPSW estimates a per-frame broadband floor
# from a split window over frequency (centre guard excluded), so steady narrowband lines stand
# proud. Verified on NODPAC-21: MV_3 self-match 0.234 -> 0.79, MV_13 0.000 -> 1.000.
TPSW_WIN_HZ, TPSW_GUARD_HZ, TPSW_ALPHA = 8.0, 1.5, 3.0
NORMALIZATION = f'tpsw_splitwindow_freq(win={TPSW_WIN_HZ}Hz,guard={TPSW_GUARD_HZ}Hz,alpha={TPSW_ALPHA})'

# ===== verbatim from stage2_lofar.py, EXCEPT normalise_spectrogram (corrected to TPSW) ====
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
        start_bin = bi * band_width_bins
        end_bin = min(start_bin + band_width_bins, len(f_sub))
        band_spec[bi, :] = np.max(s_sub[start_bin:end_bin, :], axis=0)
        peak_bin = start_bin + np.argmax(np.mean(s_sub[start_bin:end_bin, :], axis=1))
        band_freqs.append(float(f_sub[peak_bin]))
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
                    persistent.append({
                        'freq_hz': band_freqs[bi],
                        'mean_db': float(np.mean(band_spec[bi, start_idx:ti])),
                        'max_db': float(np.max(band_spec[bi, start_idx:ti])),
                    })
                start_idx = None
        if start_idx is not None and len(row) - start_idx >= min_frames:
            persistent.append({
                'freq_hz': band_freqs[bi],
                'mean_db': float(np.mean(band_spec[bi, start_idx:])),
                'max_db': float(np.max(band_spec[bi, start_idx:])),
            })
    return sorted(persistent, key=lambda t: t['mean_db'], reverse=True)
# ===== end verbatim ======================================================================

def top_lines(persistent, k=TOP_K):
    """Collapse each band-frequency to its strongest detection, then keep the K loudest."""
    best = {}
    for ln in persistent:
        f = ln['freq_hz']
        if f not in best or ln['mean_db'] > best[f]:
            best[f] = ln['mean_db']
    return sorted(best.items(), key=lambda x: -x[1])[:k]   # [(freq, mean_db), ...]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['val'])
    args = ap.parse_args()
    print(f'persist_sec={TONAL_MIN_PERSIST_SEC}  band={TONAL_FREQ_MIN_HZ}-{TONAL_FREQ_MAX_HZ} Hz  '
          f'threshold_db={TONAL_THRESHOLD_DB}  top_k={TOP_K}')
    print(f'normalization={NORMALIZATION}')

    with open(STORE_DIR / f'{TAG}_seg_meta.csv', encoding='utf-8') as f:
        meta = list(csv.DictReader(f))
    want = {}
    for m in meta:
        if m['split'] in args.splits:
            want.setdefault(m['parent_clip'], (m['vessel_id'], m['source'], m['split']))
    print(f'{len(want)} unique clips to extract across splits {args.splits}')

    stem2path = {'IARA': {}, 'SHIPSEAR': {}}
    for src, sub in (('IARA', 'IARA'), ('SHIPSEAR', 'ShipsEar')):
        root = BUILD_ROOT / sub
        if not root.exists():
            print(f'WARNING: {root} does not exist'); continue
        for w in root.rglob('*.wav'):
            stem2path[src][w.stem] = w

    out_rows, missing, zero = [], [], []
    per_src_n = defaultdict(list)
    kept_freqs = []
    for i, (clip, (vessel, source, split)) in enumerate(sorted(want.items())):
        wav = stem2path.get(source, {}).get(clip)
        if wav is None:
            missing.append(clip); continue
        try:
            audio, sr = sf.read(str(wav))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                print(f'  NOTE {clip}: sr={sr} (expected {TARGET_SR}); using as-is')
            freqs, times, Sxx = compute_stft(audio.astype('float64'), sr)
            spec_db, _ = normalise_spectrogram(freqs, times, Sxx)
            top = top_lines(detect_persistent_tonals(spec_db, freqs, times))
        except Exception as exc:
            print(f'  WARNING {clip}: {exc}'); zero.append(clip)
            out_rows.append([clip, vessel, source, split, 0, '']); continue
        if not top:
            zero.append(clip)
        per_src_n[source].append(len(top))
        kept_freqs += [f for f, _ in top]
        ser = ';'.join(f"{f:.3f}:{d:.2f}" for f, d in top)
        out_rows.append([clip, vessel, source, split, len(top), ser])
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(want)} clips')

    out_path = STORE_DIR / f'{TAG}_tonal_lines.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        wtr = csv.writer(f)
        wtr.writerow(['parent_clip', 'vessel_id', 'source', 'split', 'n_lines', 'lines'])
        wtr.writerows(out_rows)
    print(f'\nwrote {out_path}  ({len(out_rows)} clips)')

    # sidecar self-declaration: which normaliser produced this line CSV
    meta_path = STORE_DIR / f'{TAG}_tonal_lines_meta.json'
    with open(meta_path, 'w') as f:
        json.dump({'normalization': NORMALIZATION, 'splits': args.splits,
                   'threshold_db': TONAL_THRESHOLD_DB, 'persist_sec': TONAL_MIN_PERSIST_SEC,
                   'band_hz': [TONAL_FREQ_MIN_HZ, TONAL_FREQ_MAX_HZ], 'top_k': TOP_K}, f, indent=2)
    print(f'wrote {meta_path}')

    print('\n=== coverage ===')
    if missing:
        print(f'MISSING wavs: {len(missing)} (first few: {missing[:5]})')
    for src, counts in sorted(per_src_n.items()):
        counts = np.array(counts)
        print(f'  {src:9s} clips={len(counts):4d}  mean_lines={counts.mean():.1f}  '
              f'<{TOP_K} lines: {int((counts < TOP_K).sum())}  zero: {int((counts == 0).sum())}')
    if kept_freqs:
        kf = np.array(kept_freqs)
        print(f'  kept-line freq: p50 {np.median(kf):.0f} Hz  p90 {np.percentile(kf,90):.0f} Hz  max {kf.max():.0f} Hz')
    print(f'  zero-line clips: {len(zero)} (become tonal-method "no signature" cases)')

if __name__ == '__main__':
    main()
