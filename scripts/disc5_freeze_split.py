# disc5_freeze_split.py
# Selects the hull-disjoint validation split across IARA + ShipsEar (a vessel with
# >=2 recordings of verifiable identity is val-eligible) and writes the FROZEN
# split + per-recording manifest that every later step reads. Build-order step 2.
"""
DISC5 — Validation split-freeze  (build-order step 2)
=====================================================
Selects the hull-disjoint validation set across IARA + ShipsEar and writes
ONE frozen artifact that every downstream step (segmentation, augmentation,
manifest, training) must read.  Run this BEFORE segmenting (locked decision D21).

Why this script exists (the load-bearing rules it enforces):
  D20  Hull-disjoint split — a validation hull appears in NO training data
       (no recording, no augmented copy). Split is by HULL, not by clip %.
  D21  Freeze BEFORE segmenting; write to file; all downstream steps read it.
  D22  Validation hulls must each have >=2 recordings so same-vs-different
       cosine can be measured.
  D25  Primary validation = IARA leave-one-vessel-out (many identities,
       real cross-passage). ShipsEar = within-type refinement.

Dataset-specific truths this script bakes in (verified from the metadata):
  IARA:
    - 645 hulls with real identity (Ship ID -> MMSI/IMO).
    - 186 have >=2 recordings  -> validation-eligible.
    - 2 acquisition platforms: Observatory (collections A-E) and Glider (F-H).
      Only 4 hulls appear on BOTH -> the sole internal cross-hardware signal.
    - CPA (closest-point-of-approach, => speed/aspect sampling) exists for
      collections A, C, F.   Background-dominant collections: E, H.
  ShipsEar:
    - Identity = MMSI parsed from the AIS-LINK column (falls back to name).
    - Only hulls with >=2 DISTINCT DAYS are genuine cross-passage; same-day
      Entra/Espera/Sale rows are state-variants of ONE docking session
      (same channel) and do NOT count as separate passages.
    - Generic, MMSI-less names ("motorboat","sailboat",...) are identity-
      ambiguous (could be different hulls) -> excluded from validation.
    - Net: ~5 genuine cross-passage hulls (all passenger ferries).

Outputs (written next to this script, or to --outdir):
  disc5_split_freeze.csv      one row per HULL: identity, source, type,
                              n_recordings, n_sessions, has_cpa,
                              on_both_hardware, split  (train|val)
  disc5_recording_manifest.csv one row per RECORDING/WAV: maps each file to its
                              vessel_id + split + flags (platform, has_cpa,
                              is_background, identity_confidence). This is what
                              segmentation reads.

Usage:
  python disc5_freeze_split.py \
      --iara /path/iara.xlsx --shipsear /path/shipsEar.xlsx \
      --n-iara-val 40 --both-hw-policy val --seed 1234
"""

import argparse, re, sys
from pathlib import Path
import pandas as pd

# ---- IARA collection -> platform / CPA / background (from summary.png) --------
IARA_PLATFORM   = {'A':'OS','B':'OS','C':'OS','D':'OS','E':'OS',
                   'F':'Glider','G':'Glider','H':'Glider'}
IARA_CPA_COLLS  = {'A','C','F'}          # collections where CPA is captured
IARA_BG_COLLS   = {'E','H'}              # background-dominant collections


# ============================== IARA ==========================================
def load_iara(path):
    sr = pd.read_excel(path, 'ship_recordings'); sr.columns = sr.columns.map(lambda c: str(c).strip())
    si = pd.read_excel(path, 'ship_info');       si.columns = si.columns.map(lambda c: str(c).strip())
    di = pd.read_excel(path, 'dataset_info');    di.columns = di.columns.map(lambda c: str(c).strip())

    for c in ['Qty recordings', 'With CPA', 'on OS', 'on Glider', 'on Both']:
        sr[c] = pd.to_numeric(sr[c], errors='coerce').fillna(0)

    types = si.set_index('Ship ID')['AIS TYPE SUMMARY'].to_dict()

    hulls = pd.DataFrame({
        'vessel_id'       : 'IARA:' + sr['Ship ID'].astype(str),
        'source'          : 'IARA',
        'ship_key'        : sr['Ship ID'].astype(str),
        'display_name'    : sr['Ship ID'].astype(str),
        'type'            : sr['Ship ID'].map(types).fillna('Unknown'),
        'n_recordings'    : sr['Qty recordings'].astype(int),
        'n_sessions'      : sr['Qty recordings'].astype(int),   # each IARA rec = separate passage
        'has_cpa'         : (sr['With CPA'] > 0),
        'on_both_hardware': (sr['on Both'] > 0),
        'identity_conf'   : 'mmsi',          # IARA identity is AIS-grade
    })
    return hulls, di


def select_iara_val(hulls, n_val, both_hw_policy, rng, val_prefer='rich'):
    """Pick ~n_val IARA validation hulls: eligible (>=2 recs), spanning types,
    preferring CPA hulls; honour the both-hardware policy.
    val_prefer='rich'     -> favour high-recording hulls (richer same/diff stats,
                             but spends more data on val).
    val_prefer='moderate' -> favour 2-5 recording hulls (keeps data-rich hulls
                             for training; val just needs >=2 to be measurable)."""
    elig = hulls[hulls['n_recordings'] >= 2].copy()      # the 186

    forced = pd.Index([])
    if both_hw_policy == 'val':
        forced = elig.index[elig['on_both_hardware']]    # force the 4 into val
    elif both_hw_policy == 'train':
        elig = elig[~elig['on_both_hardware']]           # keep the 4 out of val

    remaining = max(0, n_val - len(forced))
    pool = elig.drop(index=forced, errors='ignore')

    # proportional-by-type allocation with a floor of 1 per present type
    type_counts = pool['type'].value_counts()
    alloc = {}
    if remaining > 0 and len(type_counts):
        raw = (type_counts / type_counts.sum() * remaining)
        alloc = {t: max(1, int(round(v))) for t, v in raw.items()}
        # trim/pad to hit `remaining` exactly
        while sum(alloc.values()) > remaining:
            t = max(alloc, key=lambda k: alloc[k]); alloc[t] -= 1
            if alloc[t] == 0: del alloc[t]
        while sum(alloc.values()) < remaining:
            t = type_counts.index[rng.integers(len(type_counts))]; alloc[t] = alloc.get(t, 0) + 1

    chosen = list(forced)
    for t, k in alloc.items():
        cand = pool[pool['type'] == t]
        cand = cand.assign(_r=rng.random(len(cand)))
        if val_prefer == 'moderate':
            # distance from the sweet spot of ~3 recordings, then CPA, then random
            cand = cand.assign(_d=(cand['n_recordings'] - 3).abs())
            cand = cand.sort_values(['has_cpa', '_d', '_r'], ascending=[False, True, True])
        else:  # 'rich'
            cand = cand.sort_values(['has_cpa', 'n_recordings', '_r'],
                                    ascending=[False, False, True])
        chosen += list(cand.index[:k])

    hulls['split'] = 'train'
    hulls.loc[chosen, 'split'] = 'val'
    return hulls


# ============================ ShipsEar ========================================
# Generic type words: a Name equal to one of these (no number, no proper name)
# is a BARE label that does NOT identify an individual hull.
GENERIC = {'motorboat','sailboat','fishboat','mussel boat','pilot ship','tugboat',
           'yacht','zodiac','small yacht','high speed motorboat','dredger',
           'velero','lancha','boat'}

def _mmsi(x):
    m = re.search(r'mmsi=(\d+)', str(x), re.I);  return m.group(1) if m else None
def _state(n):
    m = re.search(r'\((.*?)\)', str(n));  return m.group(1).strip().lower() if m else ''
def _name_ident(name):
    """Identity key from the Name column.
    - a quoted proper name wins  (Motorboat "Duda" -> 'duda')
    - else strip trailing (state/condition) parenthetical, normalise
    Keeps numbers (Motorboat1 != Motorboat2), merges true repeats (Duda x2)."""
    n = str(name)
    q = re.search(r'"(.*?)"', n)
    if q: return q.group(1).strip().lower()
    n = re.sub(r'\(.*?\)', '', n)
    return re.sub(r'\s+', ' ', n).strip().lower()

def load_shipsear(path):
    df = pd.read_excel(path, 'Sheet1'); df.columns = df.columns.map(lambda c: str(c).strip())
    df['mmsi']  = df['AIS LINK'].apply(_mmsi)
    df['state'] = df['Name'].apply(_state)
    df['day']   = pd.to_datetime(df['Date'], errors='coerce').dt.date
    df['is_background'] = df['Type'].astype(str).str.contains('ambient', case=False, na=False)

    idn   = df['Name'].apply(_name_ident)
    quoted = df['Name'].str.contains(r'"', na=False)
    has_digit = idn.str.contains(r'\d', na=False)
    is_bare = (~quoted) & (~has_digit) & (idn.isin(GENERIC))
    # "Duda" = Spanish for "doubt": the DB creator's uncertain-identity flag, not a
    # vessel name. Treat as uncertain -> train-only singleton, never a merged hull.
    is_doubt = df['Name'].str.contains('duda', case=False, na=False)

    # identity_conf tiers:  mmsi  >  named (distinct proper/numbered)  >  uncertain
    conf = pd.Series('named', index=df.index)
    conf[is_bare | is_doubt] = 'uncertain'
    conf[df['mmsi'].notna()] = 'mmsi'
    df['identity_conf'] = conf

    # IDENTITY KEY by tier:
    #   mmsi      -> 'mmsi:<num>'  (verifiable hull; groups all its recordings)
    #   named     -> 'name:<idn>'  (groups true repeats e.g. Pirata de Salvora x3)
    #   uncertain -> 'rec:<ID>'    (SINGLETON: bare label or doubt-flagged; never
    #                               merged, train-only, can't anchor a val pair)
    uncertain = is_bare | is_doubt
    df['identity'] = 'name:' + idn
    df.loc[df['mmsi'].notna(), 'identity'] = 'mmsi:' + df['mmsi'].astype(str)
    df.loc[uncertain,          'identity'] = 'rec:'  + df['ID'].astype(str)

    df['session_id'] = 'SHIPSEAR:' + df['identity'] + '#d' + df['day'].astype(str)
    df['base'] = idn
    return df

def build_shipsear_hulls(df):
    g = (df.groupby('identity')
           .agg(display_name=('base', 'first'), type=('Type', 'first'),
                n_recordings=('ID', 'count'), n_sessions=('day', 'nunique'),
                identity_conf=('identity_conf', 'first'),
                is_background=('is_background', 'any'))
           .reset_index())
    g['vessel_id'] = 'SHIPSEAR:' + g['identity'].astype(str)
    g['source'] = 'SHIPSEAR'; g['has_cpa'] = False; g['on_both_hardware'] = False

    # Validation-eligible ShipsEar: >=2 recordings of the SAME vessel, identity
    # confidence mmsi OR named (NOT uncertain: bare label or 'Duda' doubt-flag),
    # non-ambient.
    g['split'] = 'train'
    val = (g['n_recordings'] >= 2) & (g['identity_conf'].isin(['mmsi', 'named'])) & (~g['is_background'])
    g.loc[val, 'split'] = 'val'
    g.loc[g['is_background'], 'split'] = 'ambient'   # ambient pool, not identity
    return g


# ============================ Manifests =======================================
def shipsear_recording_rows(df, hull_split):
    df = df.copy()
    df['vessel_id'] = 'SHIPSEAR:' + df['identity'].astype(str)
    df['recording_id'] = df['ID'].astype(str)
    df['split'] = df['vessel_id'].map(hull_split).fillna('train')
    df['platform'] = 'ShipsEar_array'; df['has_cpa'] = False
    df['source'] = 'SHIPSEAR'
    return df.rename(columns={'Filename': 'collection'})[
        ['recording_id', 'vessel_id', 'source', 'collection', 'platform',
         'has_cpa', 'is_background', 'identity_conf', 'session_id', 'state', 'split']]


# ============================== main ==========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iara', required=True)
    ap.add_argument('--shipsear', required=True)
    ap.add_argument('--n-iara-val', type=int, default=40)
    ap.add_argument('--both-hw-policy', choices=['val', 'train'], default='val',
                    help="Where to send the 4 OS+Glider hulls. 'val' = measure "
                         "hardware transfer; 'train' = learn channel invariance.")
    ap.add_argument('--val-prefer', choices=['rich', 'moderate'], default='moderate',
                    help="Recording-count bias for val hulls. 'moderate' (default) "
                         "keeps data-rich hulls in training; 'rich' maximises "
                         "same/diff pair statistics in validation.")
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--outdir', default='disc5_build',
                    help="Clean output folder for manifests, kept SEPARATE from the "
                         "source audio folders. Created if absent. Later build steps "
                         "(resampled_8k/, segments/, shards/) live here too.")
    a = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(a.seed)
    out = Path(a.outdir).resolve(); out.mkdir(parents=True, exist_ok=True)

    # IARA
    iara_hulls, di = load_iara(a.iara)
    iara_hulls = iara_hulls.reset_index(drop=True)
    iara_hulls = select_iara_val(iara_hulls, a.n_iara_val, a.both_hw_policy, rng, a.val_prefer)

    # ShipsEar
    se_df = load_shipsear(a.shipsear)
    se_hulls = build_shipsear_hulls(se_df)

    # ---- frozen hull-level split -------------------------------------------
    cols = ['vessel_id', 'source', 'display_name', 'type', 'n_recordings',
            'n_sessions', 'has_cpa', 'on_both_hardware', 'identity_conf', 'split']
    freeze = pd.concat([iara_hulls[cols], se_hulls[cols]], ignore_index=True)
    freeze.to_csv(out / 'disc5_split_freeze.csv', index=False)

    # ---- recording-level manifest ------------------------------------------
    hull_split = freeze.set_index('vessel_id')['split'].to_dict()
    di2 = di.assign(source='IARA')
    di2['vessel_id'] = 'IARA:' + di2['Ship ID'].astype(str)
    di2['recording_id'] = di2['IARA ID'].astype(str)
    di2['collection'] = di2['Dataset']
    di2['platform'] = di2['Dataset'].map(IARA_PLATFORM)
    di2['has_cpa'] = di2['Dataset'].isin(IARA_CPA_COLLS)
    di2['is_background'] = di2['Dataset'].isin(IARA_BG_COLLS)
    di2['identity_conf'] = 'mmsi'
    di2['split'] = di2['vessel_id'].map(hull_split).fillna('train')
    di2.loc[di2['is_background'] & (di2['split'] == 'train'), 'split'] = 'ambient'
    # IARA: each recording is its own independent passage -> its own session.
    di2['session_id'] = di2['vessel_id'] + '#r' + di2['recording_id']
    di2['state'] = ''  # IARA has no operating-state label
    irec = di2[['recording_id', 'vessel_id', 'source', 'collection', 'platform',
                'has_cpa', 'is_background', 'identity_conf', 'session_id', 'state', 'split']]

    srec = shipsear_recording_rows(se_df, hull_split)
    manifest = pd.concat([irec, srec], ignore_index=True)
    manifest.to_csv(out / 'disc5_recording_manifest.csv', index=False)

    # ---- summary ------------------------------------------------------------
    def line(s): print(s)
    line("\n=== DISC5 split freeze ===")
    line(f"seed={a.seed}  n_iara_val={a.n_iara_val}  both_hw_policy={a.both_hw_policy}")
    for src in ['IARA', 'SHIPSEAR']:
        s = freeze[freeze.source == src]
        line(f"\n[{src}] hulls: {len(s)}  |  val: {(s.split=='val').sum()}  "
             f"train: {(s.split=='train').sum()}  ambient: {(s.split=='ambient').sum()}")
        v = s[s.split == 'val']
        if len(v):
            line(f"   val type spread: {dict(v['type'].value_counts())}")
            line(f"   val hulls with CPA: {int(v['has_cpa'].sum())}/{len(v)}")
            if v['on_both_hardware'].any():
                line(f"   val incl. both-hardware hulls: {int(v['on_both_hardware'].sum())}")
    line(f"\nRecordings: {len(manifest)}  "
         f"(val {(manifest.split=='val').sum()}, train {(manifest.split=='train').sum()}, "
         f"ambient {(manifest.split=='ambient').sum()})")
    # speed-probe availability: val recordings sharing a session (same-day variants)
    vm = manifest[manifest.split == 'val']
    sess_sizes = vm.groupby('session_id').size()
    n_speed_sessions = int((sess_sizes >= 2).sum())
    line(f"Val speed-probe sessions (>=2 same-session recs, e.g. Enter/Wait/Leave): "
         f"{n_speed_sessions}")
    line(f"Val cross-passage hulls (>=2 sessions): "
         f"{int((freeze[(freeze.split=='val')]['n_sessions']>=2).sum())}")
    line(f"\nWrote:\n  {out/'disc5_split_freeze.csv'}\n  {out/'disc5_recording_manifest.csv'}")
    line("\nNEXT: eyeball disc5_split_freeze.csv, then treat it as FROZEN — "
         "every later step reads it; do not regenerate after segmenting.")


if __name__ == '__main__':
    main()
