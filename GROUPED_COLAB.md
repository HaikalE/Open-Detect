# OpenDetect grouped split — 6 September 2026

This is a new controlled replication variant, not a claim to reproduce the authors'
unreleased partitions exactly. Base: `a07eff6` (`codex/replication-fixes`), including
decoder v2 and the validation-threshold numerical fix. This integration does not
modify the model, losses, augmentation, hyperparameters, or byte preprocessing.

## What is installed

`data/grouped.py` reads the 40 original audited index files from `grouped_manifests/`.
It verifies source NPZ hashes, index hashes, row concatenation order, class roles,
complete row coverage and exact-image separation. Training and evaluation use the
same index file, recorded in checkpoint and metrics. The original six NPZ files
are unchanged. Duplicate rows within one partition retain their original weights.

The immutable bundle still uses the historical protocol ID ending in `v1-preview`
and REPORT says `training_enabled: false`: those describe the original artifact
creation stage. This branch explicitly enables consumption of that artifact through
`--split_manifest_dir`. No indices are regenerated during training.

| Scenarios | Source | Unknown role |
|---|---|---|
| A-1, A-2, A-3 | USTC | Withheld USTC classes from `data/splits.py` |
| B-1, B-2, B-3 | Malicious TLS | Withheld Malicious TLS classes |
| C-1 | Combined | USTC unknown, Malicious TLS known |
| C-2 | Combined | Malicious TLS unknown, USTC known |

There is no C-3 in these eight scenarios. Each uses seeds 2022–2026, 100 epochs
per seed, **five repeated grouped 80:10:10 holdouts**, not five disjoint KFold tests.
Ratios apply to unique-image groups; row counts may differ slightly. Known classes
are split into train/validation/test. Unknown classes are grouped independently;
only their test partition is evaluated. `unknown_unused` is never used for training
or threshold calibration. Threshold remains calibrated using known validation only.

## Colab

Open one `OpenDetect_<scenario>_GROUPED.ipynb` from the new Drive folder
`THESIS IMPLEMENTASI/OpenDetect_GROUPED_2026-09-06`, select GPU, and Run all.
Dataset source: `THESIS IMPLEMENTASI/data/dataset`; no dataset upload/replacement needed.
The notebook installs an isolated Python 3.10 environment with pinned requirements,
verifies source hashes and runs CPU tests before launching. The launcher uses a pinned
Git commit, not whichever branch tip happens to exist later.

Output: `OpenDetect_GROUPED_2026-09-06/outputs/<scenario>/`. Old output directories
are rejected, never overwritten. Begin from epoch 1 for this new split, even if the
same NPZ was already used in an old experiment. Resume is only for this grouped run.

By default each Run all has a three-hour training budget. The limit is checked after
a complete epoch; evaluation/setup/checkpoint I/O can extend wall time beyond that
budget. If paused, Run all again. The scientific 100 epochs and five seeds are not
shortened. Do not run the same scenario in two runtimes concurrently.

## Recovery and evidence

After each epoch the trainer writes alternating `last.pt` / `last.backup.pt` and
their hash manifests to Drive, then `progress.json`. Saved state includes model,
optimizer, scheduler, random states, DataLoader generator and best checkpoint.
Only the newest valid committed slot is restored; a corrupt newest slot falls back
to the previous valid one. No valid recovery state means fail closed, except a truly
fresh run. An old best checkpoint alone is insufficient to resume training.

Data, code, manifest, environment and training configuration must match. A change of
GPU model is warned about, and numerical bitwise equivalence across GPUs is not
promised. Runtime death during an epoch repeats that epoch. Drive/FUSE is not
transactional: both slots can still be lost if remote synchronization never finishes.
This does not bypass Colab limits or guarantee execution after closing the browser.

Completion markers are written only after 100 epochs and successful evaluation.
`save_model/` retains the best-validation checkpoint, `results/` contains metrics and
raw scores, and `logs/` contains train/test logs. `per_run.csv` lists completed seeds;
`summary.json` is produced only for all five verified seeds (mean and sample SD).
Resume slots are retained, so budget several GB of Drive space per scenario.

## Local/server invocation

```bash
python run_grouped.py --scenario B-1 --output /new/empty/grouped/B-1 --session_hours 3
```

Provide the unchanged NPZ pair under `data/dataset`. Run again with the same command
to resume. This CLI is the trainer's launcher, not a Google Colab provisioning CLI.
No server orchestrator or old cloud runtime is modified by this package.

To build pinned notebook artifacts from a committed checkout:

```bash
python build_grouped_colabs.py --commit FULL_COMMIT --output /new/notebooks
python -m unittest discover -s tests -v
```

Full NPZ partition validation is opt-in via `OPENDETECT_QA_DATA_ROOT` pointing to
the parent of the `dataset` directory. Tests include all 40 indices, tiny real-ResNet
grouped training/evaluation, CPU fault injection and mocked orchestration. They are
not full 100-epoch GPU validation. The CPU recovery test uses a tiny mock network to
check optimizer/scheduler/RNG equality and corrupted-slot fallback.

## Interpretation limits

Exact-image grouping fixes measured overlap of identical 1024-byte images. It does
not prove distinct PCAP flows/captures, resolve missing PCAP provenance or label
semantics, repair bytes in distributed NPZ files, or force counts to match the paper.
It is not selected to improve any seed. Report all five seeds, including poor ones.
Do not merge these metrics with results from the previous row-wise experiment.
