# DISC5 Training — Directory Structure

Layout expected by `notebooks/DISC5_Training_100.ipynb` (its Cell-2 `CFG` block)
and produced by the build pipeline. Two sides: a **build machine** (local,
CPU-only, where audio becomes tensors) and the **training environment** (GPU,
where tensors become checkpoints). They exchange exactly one large file: the
tensor tar.

## 1. Build machine

```
<work>/
├── disc5_freeze_split.py            build-order step 2 (split + manifest)
├── disc5_build_tensor_tar.py        archive builder (this repo, scripts/)
└── disc5_build/                     all generated artefacts, SEPARATE from
    │                                the source-audio folders
    ├── disc5_split_freeze.csv       frozen hull-level split (one row per hull)
    ├── disc5_recording_manifest.csv one row per recording: vessel_id, split,
    │                                platform, has_cpa, session_id, ...
    ├── resampled_8k/                8 kHz mono WAV mirror of the source audio
    ├── tensors/                     5 s float32 segments as .npy
    │   ├── tensor_index.csv         one row per tensor: path, recording_id,
    │   │                            seg_index, shape, dtype, sr
    │   └── <collection>/<recording>/seg_XXXX.npy
    └── disc5_tensors.tar            single-file transfer artefact
```

Source audio (IARA / ShipsEar distributions) stays wherever it was downloaded;
scripts take its location as arguments and never write into it.

## 2. Training environment (as in the notebook CFG)

The notebook was run on Colab with Google Drive as persistent storage; the CFG
keys map one-to-one onto this layout and can be re-pointed at any equivalent
paths:

```
MyDrive/
├── DISC5/
│   ├── disc5_tensors.tar            uploaded archive (drive_tar)
│   ├── disc5_recording_manifest.csv (drive_manifest)
│   ├── tensors/                     optional extracted copy (drive_tensors)
│   └── epoch_sets/                  per-epoch sampling CSVs (drive_epoch_sets)
└── DISC5_Checkpoints/               checkpoints + run history (drive_ckpt_dir)

/content/disc5/tensors               fast local SSD copy, extracted from the tar
                                     at session start (ssd_root)
/content/output                      scratch outputs for the session (output_dir)
```

Session start sequence (notebook cells): mount Drive → extract `drive_tar` to
`ssd_root` with the per-file progress loop → load manifest + epoch sets → train,
checkpointing every epoch to `drive_ckpt_dir`.

## 3. Ordering rule

The split freeze (step 2) runs **before** any segmentation, and every later
stage — segmentation, tensor generation, epoch-set construction, training —
reads `disc5_recording_manifest.csv` rather than re-deriving membership. The
split is by hull, not by clip percentage, and is never regenerated after
segmentation has begun.
