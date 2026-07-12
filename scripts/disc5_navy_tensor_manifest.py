# disc5_navy_tensor_manifest.py
# Builds the authoritative per-tensor manifest by SCANNING what is actually on disk: it
# walks OUT_DIR, loads every .npy (mmap, header only) to record its real shape/dtype,
# parses clip/piece/condition/segment from the path, joins the clip-level perturbation
# provenance (disc5_navy_manifest.csv), and flags any tensor that is not (1,1,40000) float32.
# Run after disc5_prep_navy.py. Writes disc5_navy_tensor_manifest.csv.

import csv
from pathlib import Path
from collections import Counter
import numpy as np

OUT_DIR  = Path(r'C:\DISC5\navy_tensors')
EXPECT_SHAPE = (1, 1, 40000)

def main():
    # clip-level provenance (Doppler sign, SNR, split mode) keyed by clip_id
    prov = {}
    clip_man = OUT_DIR / 'disc5_navy_manifest.csv'
    if clip_man.exists():
        with open(clip_man, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                prov[r['clip_id']] = r
    else:
        print(f'WARNING: {clip_man} not found -- provenance columns will be blank')

    files = sorted(p for p in OUT_DIR.rglob('*.npy'))
    rows, bad = [], []
    for p in files:
        stem = p.stem                       # <clip_id>__<piece>__<cond>__segNN
        try:
            clip_id, piece, cond, segtok = stem.rsplit('__', 3)
            seg_index = int(segtok[3:])
        except ValueError:
            bad.append((p.name, 'unparseable name')); continue
        try:
            arr = np.load(p, mmap_mode='r')
            shape, dtype = tuple(arr.shape), str(arr.dtype)
        except Exception as exc:
            bad.append((p.name, f'load error: {exc}')); continue
        ok = (shape == EXPECT_SHAPE and dtype == 'float32')
        if not ok:
            bad.append((p.name, f'shape={shape} dtype={dtype}'))
        pr = prov.get(clip_id, {})
        rows.append(dict(
            tensor_path=str(p.relative_to(OUT_DIR)).replace('\\', '/'),
            clip_id=clip_id, piece=piece, condition=cond, seg_index=seg_index,
            shape=str(shape), dtype=dtype, sr=8000, ok=int(ok),
            source=pr.get('source', ''), split_mode=pr.get('split_mode', ''),
            doppler_sign=pr.get('doppler_sign', '') if piece == 'query' and 'speed' in cond else '',
            speed_pct=pr.get('speed_pct', '') if piece == 'query' and 'speed' in cond else '',
            snr_db=pr.get('snr_db', '') if piece == 'query' and 'noise' in cond else '',
        ))

    cols = ['tensor_path', 'clip_id', 'source', 'piece', 'condition', 'seg_index',
            'shape', 'dtype', 'sr', 'ok', 'split_mode', 'doppler_sign', 'speed_pct', 'snr_db']
    out = OUT_DIR / 'disc5_navy_tensor_manifest.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print(f'scanned {len(files)} .npy files -> {len(rows)} rows  | {out}')
    print('by piece/condition:', dict(Counter(f"{r['piece']}/{r['condition']}" for r in rows)))
    print('distinct clips:', len({r['clip_id'] for r in rows}))
    n_bad = sum(1 for r in rows if not r['ok']) + len([b for b in bad if 'unparseable' in b[1] or 'load' in b[1]])
    if bad:
        print(f'PROBLEMS ({len(bad)}):')
        for name, why in bad[:20]:
            print(f'  {name}: {why}')
    else:
        print('all tensors are (1,1,40000) float32 -- clean')

if __name__ == '__main__':
    main()
