# disc5_resample_arrange.py
# Reads the frozen manifest and, for every IARA + ShipsEar train/val identity clip,
# resamples the source WAV to 8 kHz mono and arranges it by source/vessel into a
# clean disc5_build tree, leaving the original source folders untouched. Reusable
# for any re-pull (new sources, re-staging ambient, different sample rate).
"""
DISC5 — Step 1: arrange + resample the training/validation clips
================================================================
Reads the FROZEN manifest (disc5_recording_manifest.csv from step 2) and, for
every IARA + ShipsEar identity recording (train and val), reads the source WAV,
**resamples to 8 kHz mono** (locked decision D12), and writes it into a clean
working tree:

    <outroot>/                      (default C:\\DISC5\\disc5_build)
        IARA/
            IARA_220/               vessels with >=2 recordings get a folder
                A-0008.wav
                A-0009.wav
            A-0001.wav              singleton hulls go loose (no one-file folders)
            ...
        ShipsEar/
            SHIPSEAR_mmsi_224084240/   Mar de Cangas (6 recordings)
                ...
            21__18_07_13_lanchaMotora.wav   singleton, loose
        ambient_manifest.csv        (only with --include-ambient) weather tags
                                    for the staged ambient clips

The ORIGINAL source folders (IARA\\, shipsEar\\) are never modified — read-only.
QianDao is intentionally EXCLUDED (its clips are 3 s, shorter than our 5 s window).
'ambient' rows are skipped unless --include-ambient is passed; with it, ambient is
staged flat under <source>/_ambient/ AND an ambient_manifest.csv is written joining
each clip to its weather (IARA rain/wind/sea-state from iara.xlsx; ShipsEar condition
parsed from the filename). Pass --iara-xlsx with --include-ambient for the IARA tags.

Source-file location logic (verified against the metadata):
  IARA : files are <Dataset>-<NNNN>.wav inside C:\\DISC5\\IARA\\<Dataset>\\ , where
         NNNN = (IARA_ID - first_ID_in_that_dataset + 1), zero-padded to 4.
         The manifest carries recording_id (= IARA ID) and collection (= Dataset).
  ShipsEar: the manifest's 'collection' column already holds the exact filename.

Usage:
  python disc5_step1_arrange_resample.py \
      --manifest C:\\DISC5\\disc5_build\\disc5_recording_manifest.csv \
      --iara-root C:\\DISC5\\IARA --shipsear-root C:\\DISC5\\shipsEar \
      --outroot C:\\DISC5\\disc5_build --sr 8000
"""

import argparse, sys, csv
from pathlib import Path
import pandas as pd


def safe_name(s):
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(s))


def iara_src_path(iara_root, collection, recording_id):
    """Map (Dataset, IARA ID) -> C:\\...\\IARA\\<Dataset>\\<Dataset>-<IARA_ID>.wav
    The number in the filename is the GLOBAL IARA ID, zero-padded to 4
    (e.g. B-0457.wav, not B-0001.wav)."""
    return Path(iara_root) / str(collection) / f"{collection}-{int(recording_id):04d}.wav"


def load_audio_resample(path, sr_target):
    """Load any-rate WAV -> mono float32 at sr_target. Prefers soundfile+resampy,
    falls back to librosa, then scipy. Returns (samples, sr) or raises."""
    import numpy as np
    try:
        import soundfile as sf
        y, sr = sf.read(str(path), always_2d=True)
        y = y.mean(axis=1).astype('float32')        # -> mono
        if sr != sr_target:
            try:
                import resampy
                y = resampy.resample(y, sr, sr_target)
            except ImportError:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(sr, sr_target)
                y = resample_poly(y, sr_target // g, sr // g).astype('float32')
        return y.astype('float32'), sr_target
    except Exception:
        import librosa                              # last-resort single dependency
        y, _ = librosa.load(str(path), sr=sr_target, mono=True)
        return y.astype('float32'), sr_target


def write_wav(path, y, sr):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype='PCM_16')


def rain_bucket(state, mm):
    """Coarse rain category from IARA 'Rain state' (preferred) or mm fallback."""
    s = str(state).strip().lower()
    if s in ('no rain', 'light', 'moderate', 'heavy', 'very heavy'):
        return s.replace(' ', '_')
    try:
        v = float(mm)
    except (TypeError, ValueError):
        return 'unknown'
    if v == 0: return 'no_rain'
    if v < 0.5: return 'light'
    if v < 2:   return 'moderate'
    return 'heavy'

def shipsear_ambient_condition(fname):
    """ShipsEar natural-noise condition from the filename (Spanish)."""
    f = str(fname).lower()
    for key, cond in [('lluvia', 'rain'), ('viento', 'wind'),
                      ('oleaje', 'waves'), ('corriente', 'current')]:
        if key in f:
            return cond
    return 'ambient'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--iara-root', required=True)
    ap.add_argument('--shipsear-root', required=True)
    ap.add_argument('--outroot', default=r'C:\DISC5\disc5_build')
    ap.add_argument('--sr', type=int, default=8000)
    ap.add_argument('--include-ambient', action='store_true')
    ap.add_argument('--iara-xlsx', default=None,
                    help="Path to iara.xlsx; required with --include-ambient to "
                         "attach rain/wind/sea-state to IARA ambient clips.")
    a = ap.parse_args()

    man = pd.read_csv(a.manifest)
    out = Path(a.outroot)

    # weather lookup for IARA ambient: IARA ID -> (rain_mm, rain_state, wind, sea_state)
    weather = {}
    if a.include_ambient and a.iara_xlsx:
        di = pd.read_excel(a.iara_xlsx, 'dataset_info'); di.columns = di.columns.map(lambda c: str(c).strip())
        for _, w in di.iterrows():
            weather[str(w['IARA ID'])] = (w.get('Rain'), w.get('Rain state'),
                                          w.get('Wind'), w.get('Sea state'))
    elif a.include_ambient and not a.iara_xlsx:
        print("WARNING: --include-ambient without --iara-xlsx; IARA ambient weather "
              "tags will be blank.", file=sys.stderr)

    ambient_rows = []   # for ambient_manifest.csv

    wanted = {'train', 'val'} | ({'ambient'} if a.include_ambient else set())
    rows = man[man['split'].isin(wanted) & man['source'].isin(['IARA', 'SHIPSEAR'])].copy()
    # vessels with >=2 recordings get their own folder; singletons go loose in the
    # source folder (avoids hundreds of one-file folders).
    rec_counts = rows.groupby('vessel_id')['recording_id'].transform('count')
    rows['multi'] = rec_counts >= 2

    log = []
    n_ok = n_skip = n_missing = 0
    for _, r in rows.iterrows():
        src_label = r['source']
        split = r['split']
        vessel = safe_name(r['vessel_id']) if r['multi'] else None  # folder only if >=2 recs

        if src_label == 'IARA':
            src = iara_src_path(a.iara_root, r['collection'], r['recording_id'])
            sub = 'IARA' if split != 'ambient' else 'IARA/_ambient'
            dst_name = src.name if src else None
        else:  # SHIPSEAR
            src = Path(a.shipsear_root) / str(r['collection'])   # collection == filename
            sub = 'ShipsEar' if split != 'ambient' else 'ShipsEar/_ambient'
            dst_name = src.name

        if src is None or not src.exists():
            n_missing += 1
            log.append([src_label, r['recording_id'], r['vessel_id'], split,
                        str(src), '', 'MISSING_SOURCE'])
            continue

        dst = (out / sub / vessel / dst_name) if vessel else (out / sub / dst_name)
        if dst.exists():
            n_skip += 1
            log.append([src_label, r['recording_id'], r['vessel_id'], split,
                        str(src), str(dst), 'EXISTS_SKIP'])
            continue
        try:
            y, sr = load_audio_resample(src, a.sr)
            write_wav(dst, y, sr)
            n_ok += 1
            log.append([src_label, r['recording_id'], r['vessel_id'], split,
                        str(src), str(dst), 'OK'])
            if split == 'ambient':
                if src_label == 'IARA':
                    rain_mm, rain_state, wind, sea = weather.get(str(r['recording_id']),
                                                                 (None, None, None, None))
                    cond = rain_bucket(rain_state, rain_mm)
                else:  # ShipsEar
                    rain_mm = rain_state = wind = sea = None
                    cond = shipsear_ambient_condition(dst_name)
                ambient_rows.append([str(dst), src_label, r['recording_id'], cond,
                                     rain_mm, rain_state, wind, sea])
        except Exception as e:
            n_missing += 1
            log.append([src_label, r['recording_id'], r['vessel_id'], split,
                        str(src), str(dst), f'ERROR:{e}'])

        if (n_ok + n_skip) % 100 == 0 and (n_ok + n_skip) > 0:
            print(f'  ...{n_ok} written, {n_skip} skipped, {n_missing} missing')

    # No _arrange_log.csv — it was bookkeeping only and nothing downstream needs it.
    # Missing/error files (if any) are listed to the console below.
    out.mkdir(parents=True, exist_ok=True)

    if a.include_ambient:
        with open(out / 'ambient_manifest.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['path', 'source', 'recording_id', 'condition',
                        'rain_mm', 'rain_state', 'wind', 'sea_state'])
            w.writerows(ambient_rows)

    print(f"\nDone. target_sr={a.sr} Hz mono")
    print(f"  written : {n_ok}")
    print(f"  skipped : {n_skip} (already present)")
    print(f"  missing/error: {n_missing}")
    print(f"  tree    : {out}\\<IARA|ShipsEar>\\  (>=2-rec vessels foldered, singletons loose)")
    if a.include_ambient:
        print(f"  ambient : {len(ambient_rows)} clips in <source>\\_ambient\\ + ambient_manifest.csv")
    if n_missing:
        print("  -- missing/error files --")
        for row in log:
            if row[6] != 'OK' and row[6] != 'EXISTS_SKIP':
                print(f"     {row[6]}: {row[4]}")


if __name__ == '__main__':
    main()
