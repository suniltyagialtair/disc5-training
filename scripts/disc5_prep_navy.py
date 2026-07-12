# disc5_prep_navy.py
# Local (CPU) prep of the 21 NODPAC clips for the tonal-vs-SKANN test. Writes ONE tensor
# per 5 s segment, shape [1,1,40000] float32 -- identical to disc5_segment_augment.py -- so
# the encoder consumes them exactly as it does training tensors. resample/downmix, znorm,
# windowing, and ambient-mix are lifted verbatim from the project scripts.
#
# Per clip:  gallery = first 5 min (clean)  ;  query = last 5 min, in 4 conditions:
#   clean | +noise (real ambient @ fixed SNR) | speed (+-4% resample) | speed+noise
# Clips < 10 min are split in half. Doppler sign assigned half-approach/half-recede across
# the 21, SEEDED, and written to disc5_navy_manifest.csv (perturbation provenance). The
# authoritative per-tensor manifest is built separately by disc5_navy_tensor_manifest.py.
#
# Layout:  OUT_DIR/<clip_id>/<clip_id>__<piece>__<cond>__segNN.npy

import csv
from pathlib import Path
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ---- paths / params (edit) --------------------------------------------------------------
NAVY_FOLDERS = [Path(r'D:\Navy\Decommissioned'), Path(r'D:\Navy\MV')]
AMBIENT_DIRS = [Path(r'C:\DISC5\disc5_build\IARA\_ambient'),
                Path(r'C:\DISC5\disc5_build\ShipsEar\_ambient')]
OUT_DIR      = Path(r'C:\DISC5\navy_tensors')

SR, SEG, MIN_TAIL = 8000, 40000, 8000
WIN_SEC, SHORT_SEC = 300, 600
SNR_DB, SPEED_PCT, SEED, EPS = 10.0, 4.0, 1234, 1e-8

# ===== verbatim from disc5_resample_arrange.py / disc5_segment_augment.py =================
def load_audio_resample(path, sr_target):
    y, sr = sf.read(str(path), always_2d=True)
    y = y.mean(axis=1).astype('float32')
    if sr != sr_target:
        g = gcd(sr, sr_target)
        y = resample_poly(y, sr_target // g, sr // g).astype('float32')
    return y.astype('float32'), sr

def znorm(y):
    m = float(y.mean()); s = float(y.std())
    return ((y - m) if s < EPS else (y - m) / s).astype('float32')

def windows(y, seg=SEG, min_tail=MIN_TAIL):
    n = len(y)
    if n < seg:
        return []
    out = [y[i * seg:(i + 1) * seg] for i in range(n // seg)]
    if n - (n // seg) * seg >= min_tail:
        out.append(y[n - seg:])
    return out

class AmbientBank:
    def __init__(self, paths):
        self.paths = [Path(p) for p in paths if Path(p).exists()]
        self._frames = {}
    def _f(self, p):
        if p not in self._frames:
            self._frames[p] = sf.info(str(p)).frames
        return self._frames[p]
    def chunk(self, n, rng):
        if not self.paths:
            return None
        p = self.paths[int(rng.integers(len(self.paths)))]
        f = self._f(p)
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
    sp = np.mean(y * y) + EPS
    target_np = sp / (10.0 ** (snr_db / 10.0))
    amb = amb * np.sqrt(target_np / (np.mean(amb * amb) + EPS))
    return (y + amb).astype('float32')
# ===== end verbatim ======================================================================

def speed_shift(y, sign):
    pct = SPEED_PCT / 100.0
    if sign == 'approach':                          # freqs * (1+pct)
        up, down = 1000, int(round(1000 * (1 + pct)))
    else:                                            # recede: freqs * (1-pct)
        up, down = 1000, int(round(1000 * (1 - pct)))
    return resample_poly(y, up, down).astype('float32')

def save_segments(segs, clipdir, clip_id, piece, cond):
    for i, s in enumerate(segs):
        t = znorm(s).reshape(1, 1, SEG).astype('float32')
        np.save(clipdir / f'{clip_id}__{piece}__{cond}__seg{i:02d}.npy', t)
    return len(segs)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bank = AmbientBank([p for d in AMBIENT_DIRS for p in (d.rglob('*.wav') if d.exists() else [])])
    print(f'ambient bank: {len(bank.paths)} clips')

    clips = []
    for folder in NAVY_FOLDERS:
        for w in sorted(folder.glob('*.wav')):
            info = sf.info(str(w))
            clips.append(dict(source=folder.name, file=w, name=w.stem,
                              sr=info.samplerate, ch=info.channels, dur=info.frames / info.samplerate))
    print(f'{len(clips)} clips found')

    rng = np.random.default_rng(SEED)
    order = list(range(len(clips))); rng.shuffle(order)
    half = len(clips) // 2
    sign_of = {idx: ('approach' if k < half else 'recede') for k, idx in enumerate(order)}

    man = []
    for ci, c in enumerate(clips):
        cid = f"{c['source']}__{c['name']}".replace(' ', '_')
        clipdir = OUT_DIR / cid; clipdir.mkdir(parents=True, exist_ok=True)
        y, _ = load_audio_resample(c['file'], SR)
        dur = len(y) / SR; win = WIN_SEC * SR
        if dur >= SHORT_SEC:
            gal, qry, mode = y[:win], y[-win:], 'first_last_5min'
            gs, ge, qs, qe = 0.0, float(WIN_SEC), dur - WIN_SEC, dur
        else:
            mid = len(y) // 2
            gal, qry, mode = y[:mid], y[mid:], 'half_split'
            gs, ge, qs, qe = 0.0, mid / SR, mid / SR, dur
        sign = sign_of[ci]
        crng = np.random.default_rng(SEED + 1 + ci)

        ng = save_segments(windows(gal), clipdir, cid, 'gallery', 'clean')
        q_clean = windows(qry)
        q_speed = windows(speed_shift(qry, sign))
        nqc = save_segments(q_clean, clipdir, cid, 'query', 'clean')
        save_segments([ambient_mix(s, bank, crng) for s in q_clean], clipdir, cid, 'query', 'noise')
        nqs = save_segments(q_speed, clipdir, cid, 'query', 'speed')
        save_segments([ambient_mix(s, bank, crng) for s in q_speed], clipdir, cid, 'query', 'speed_noise')

        man.append([cid, c['source'], c['file'].name, c['sr'], c['ch'], round(dur, 1), mode,
                    round(gs, 1), round(ge, 1), round(qs, 1), round(qe, 1),
                    sign, SPEED_PCT, SNR_DB, ng, nqc, nqs])
        print(f"  [{ci+1:2d}/{len(clips)}] {cid:24s} {sign:8s} gal={ng} qry={nqc}/{nqs}(speed)")

    with open(OUT_DIR / 'disc5_navy_manifest.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['clip_id', 'source', 'file', 'orig_sr', 'channels', 'dur_s', 'split_mode',
                    'gallery_start_s', 'gallery_end_s', 'query_start_s', 'query_end_s',
                    'doppler_sign', 'speed_pct', 'snr_db',
                    'n_seg_gallery', 'n_seg_query_clean', 'n_seg_query_speed'])
        w.writerows(man)

    n_app = sum(1 for r in man if r[11] == 'approach')
    print(f'\nwrote per-segment tensors for {len(man)} clips to {OUT_DIR}')
    print(f'Doppler split: {n_app} approach / {len(man) - n_app} recede (seed {SEED})')
    print('next: run disc5_navy_tensor_manifest.py, then tar + upload navy_tensors to Drive')

if __name__ == '__main__':
    main()
