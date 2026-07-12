# disc5_verify_resample.py
# Verifies the resampled clips in disc5_build against the sources: confirms counts
# match the arrange log, every output is 8 kHz mono, and each resampled clip's
# DURATION matches its source clip (catching truncation/corruption). Uses the
# src->dst mapping in _arrange_log.csv. Read-only; prints tables + writes a report.
#
# Usage:
#   python disc5_verify_resample.py --build-root C:\DISC5\disc5_build
#   (optional: --log C:\DISC5\disc5_build\_arrange_log.csv  --sr 8000)

import argparse
from pathlib import Path
import pandas as pd

try:
    import soundfile as sf
except ImportError:
    raise SystemExit("Need soundfile:  pip install soundfile")


def wav_info(path):
    """Return (samplerate, channels, duration_s) without loading audio, or None."""
    try:
        i = sf.info(str(path))
        return i.samplerate, i.channels, i.frames / i.samplerate if i.samplerate else 0.0
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-root', default=r'C:\DISC5\disc5_build')
    ap.add_argument('--log', default=None, help='default: <build-root>\\_arrange_log.csv')
    ap.add_argument('--sr', type=int, default=8000, help='expected output sample rate')
    ap.add_argument('--tol', type=float, default=0.05, help='duration match tolerance (s)')
    a = ap.parse_args()

    root = Path(a.build_root)
    log_path = Path(a.log) if a.log else root / '_arrange_log.csv'
    if not log_path.exists():
        raise SystemExit(f"arrange log not found: {log_path}")

    log = pd.read_csv(log_path, encoding='latin-1')
    print(f"=== arrange log: {len(log)} rows  ({log_path}) ===\n")

    # 1) status breakdown
    print("--- 1. Status breakdown ---")
    print(log['status'].value_counts().to_frame('count').to_string(), "\n")

    ok = log[log['status'] == 'OK'].copy()
    if ok.empty:
        print("No rows with status OK — nothing resampled? Stopping.")
        return

    # 2) counts by source x split (what was written)
    print("--- 2. Resampled clips by source x split ---")
    print(ok.groupby(['source', 'split']).size().to_frame('count').to_string(), "\n")

    # 3) read format + duration for each dst, and source duration for comparison
    rows = []
    for _, r in ok.iterrows():
        dst = wav_info(r['dst_path'])
        src = wav_info(r['src_path'])
        d_sr, d_ch, d_dur = dst if dst else (None, None, None)
        s_dur = src[2] if src else None
        rows.append({
            'source': r['source'], 'split': r['split'],
            'dst_sr': d_sr, 'dst_ch': d_ch,
            'dst_dur': round(d_dur, 3) if d_dur is not None else None,
            'src_dur': round(s_dur, 3) if s_dur is not None else None,
            'dur_diff': round(abs(d_dur - s_dur), 3) if (d_dur is not None and s_dur is not None) else None,
            'dst_path': r['dst_path'], 'src_path': r['src_path'],
        })
    df = pd.DataFrame(rows)

    n_dst_missing = df['dst_sr'].isna().sum()
    df_ok = df[df['dst_sr'].notna()].copy()

    # 4) format check
    print("--- 3. Format distribution (sr, channels) ---")
    fmt = df_ok.groupby(['dst_sr', 'dst_ch']).size().to_frame('count')
    print(fmt.to_string(), "\n")

    bad_fmt = df_ok[(df_ok['dst_sr'] != a.sr) | (df_ok['dst_ch'] != 1)]
    print(f"--- 4. Files NOT {a.sr} Hz mono (should be 0): {len(bad_fmt)} ---")
    if len(bad_fmt):
        print(bad_fmt[['dst_sr', 'dst_ch', 'dst_path']].head(20).to_string(index=False))
    if n_dst_missing:
        print(f"  ALSO: {n_dst_missing} resampled files could not be read (missing/corrupt).")
    print()

    # 5) duration preservation
    print(f"--- 5. Duration mismatch vs source (tol {a.tol}s) ---")
    cmp = df_ok[df_ok['dur_diff'].notna()]
    bad_dur = cmp[cmp['dur_diff'] > a.tol].sort_values('dur_diff', ascending=False)
    print(f"  clips compared: {len(cmp)} | mismatches > {a.tol}s: {len(bad_dur)}")
    if len(bad_dur):
        print(bad_dur[['source', 'dst_dur', 'src_dur', 'dur_diff', 'dst_path']].head(20).to_string(index=False))
    print()

    # 6) duration stats + shortest
    print("--- 6. Duration stats by source x split ---")
    stats = df_ok.groupby(['source', 'split'])['dst_dur'].agg(['count', 'min', 'max', 'sum'])
    stats['sum_hours'] = (stats['sum'] / 3600).round(2)
    print(stats.round(2).to_string(), "\n")

    print("--- 7. Shortest 10 clips (sanity: none ~0) ---")
    print(df_ok.nsmallest(10, 'dst_dur')[['source', 'split', 'dst_dur', 'dst_path']].to_string(index=False), "\n")

    # report file
    rep = root / '_verify_report.csv'
    df.to_csv(rep, index=False)

    # verdict
    total_hours = round(df_ok['dst_dur'].sum() / 3600, 2)
    print("=== VERDICT ===")
    print(f"  resampled clips read OK : {len(df_ok)}")
    print(f"  unreadable/missing dst  : {n_dst_missing}")
    print(f"  wrong format (not {a.sr} mono): {len(bad_fmt)}")
    print(f"  duration mismatches     : {len(bad_dur)}")
    print(f"  total audio retained    : {total_hours} h")
    print(f"  per-file report written : {rep}")
    if n_dst_missing == 0 and len(bad_fmt) == 0 and len(bad_dur) == 0:
        print("  PASS — counts/format/durations all consistent.")
    else:
        print("  CHECK — see flagged rows above and _verify_report.csv.")


if __name__ == '__main__':
    main()
