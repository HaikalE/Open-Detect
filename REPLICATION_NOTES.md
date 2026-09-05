# Replication fixes, 2026-09-05

Base: `2554fba5e3871c36f4d2439af68db6388e754e1b` on
`codex/paper-alignment`. This is a separate implementation version, not a claim
that the paper's numerical results have been reproduced.

## Confirmed defects

1. **Decoder final residual block was bypassed.** `layer1(x1)` was calculated,
   but the final convolution consumed `x1` instead of that result. The block's
   parameters received no reconstruction gradient. The regression test fails
   with `grad is None` on the base commit. Decoder v2 consumes the block output.
   This changes training and requires a new run from initialization. It is
   consistent with the mirror ResNet18 decoder described in paper VI-A.2,
   but does not establish that the authors' experiments used this exact fix.
2. **Threshold rounding violated the requested acceptance.** The threshold was
   advanced by one float64 ULP, then compared with float32 arrays. NumPy could
   round it back to the selected score. For 3,258 validation samples the base
   implementation accepts 3,095 (94.9969%) rather than the required at least
   3,096. Comparisons now consistently use float64 and reject invalid scores.
   The strict rule `score < threshold` remains (paper IV-D, Eqs. 21-22).
   This small boundary error does not explain the very large seed-2024 drop.
3. **Header-only preprocessing could contain payload.** With `keep_payload=False`,
   application bytes were not removed before taking the first 80 header bytes.
   Extraction now separates network/transport and application bytes structurally,
   including parsed application layers without Scapy Raw. Packet copies preserve
   the input capture, and excluded packets no longer consume an eight-packet slot.
   Paper IV-B (p. 6) separates header and payload, excludes ARP/DHCP and masks IPs.
   These fixes only affect future PCAP conversion, not existing NPZ contents.

Three regression tests were executed against the base behavior and failed for
these specific reasons before the fixes. The tests cover behavior, not accuracy.

## Evidence recorded by this version

- Dataset file SHA-256, shapes, dtypes, class counts and differences from paper
  Table II. No data is duplicated, resampled or fabricated to match the table.
- Source hashes and Git commit, training hyperparameters, best epoch and decoder
  version in new checkpoints; checkpoint SHA-256 in evaluation output.
- Evaluation refuses a different dataset fingerprint, scenario, fold or split seed.
- Binary confusion counts, the existing binary unknown-positive `open_f1`, plus
  explicitly named binary macro and weighted F1. These additional F1 values must
  not be chosen after looking for whichever is closest to the paper.
- The five-run launcher exports raw validation/test scores to `.scores.npz` for
  diagnosis without another training run. Test scores must not tune the threshold.
- Python random is seeded along with NumPy and Torch. Non-finite training loss
  fails explicitly instead of silently continuing.

## What is still unresolved

The supplied USTC NPZ files contain 34,585 samples. Their U2 and U12 counts are
434 and 72; Table II reports 2,000 for each (total 38,079). Hash agreement proves
file identity, not equivalence to the paper's PCAP processing or capture provenance.
The per-class label mapping also remains dependent on the dataset's documentation.

The supplied Malicious TLS files contain 115,636 samples, exactly twice each
class count listed in Table II (57,818 total). The combined files contain 150,221
samples. Counts alone cannot establish whether augmentation, duplicated flows,
or another preparation step explains this; no samples have been removed or
added by this patch. Original capture provenance remains necessary to resolve it.

The base branch implements five independently seeded stratified 80:10:10 splits.
The paper states 8:1:1 in VI-A.2 and "5-fold" in table captions, without enough
detail to reconstruct its split indices. This version preserves repeated 80:10:10
splits and identifies the protocol explicitly. It does not silently substitute
classical KFold or claim five disjoint test folds.

The paper's exact F1 averaging and evaluation composition still need confirmation.
`open_f1` remains binary unknown-positive; `closed_f1` remains weighted over known
classes. Correct classification of a known sample is different from merely
accepting it as known. AUROC and confusion counts help interpret this distinction.

Seed 2024 was poor in both A1 and A2. The fixes are not proven to be the cause of
that result or its remedy. Keep that seed in the official old-version aggregate.
Do not select seeds, alter lambda, tune thresholds on test data, or change inputs
just to approach published results.

## Running and comparing versions

Finish the user's running A3/B1 on the old pinned commit. Existing Colab/Kaggle
notebooks embed source hashes and a separate resume adapter; they **do not update
automatically** when this branch is pushed. Changing only their commit string is
insufficient. Regenerate and test launchers/adapters before using them with v2.

New training defaults to `save_model_v2`; results default to `results_v2`.
Existing training checkpoints cause a refusal rather than being overwritten.
Use a new experiment directory for each complete run set. The command-line
`train.py` still saves the best model; it does not implement epoch resume itself.
Epoch resume needs the separately tested notebook adapter. Never resume an old
optimizer/model state into the corrected decoder training path.

```bash
python provenance.py --dset USTC
python -m unittest discover -s tests -v
python run_5fold.py --dset USTC --split 0 --save_dir experiments/v2/A-1/save --results_dir experiments/v2/A-1/results
```

Legacy checkpoints without `decoder_version` are loaded with the historical v1
decoder for evaluation, so they are not silently reinterpreted as v2. Evaluation
uses the corrected threshold precision and labels output accordingly. It warns
when a legacy checkpoint has no training dataset fingerprint. Preserve original
JSON results and write re-evaluations into a separate results directory.

Do not pool A1/A2/A3/B1 from the old version with later scenarios from v2 into
one same-implementation replication table. Label versions separately; if a full
v2 comparison is needed, re-run all affected scenarios under v2.

## Validation scope

On 2026-09-05, all 21 unit/integration tests passed on Windows CPU using
Python 3.12, Torch 2.5.1 and NumPy 2.4.6. Tests include one real ResNet training
epoch on small synthetic images, checkpoint save/load, evaluation, score export,
legacy decoder compatibility and rejection of mismatched data or mixed results.
Synthetic test metrics are not experiment results. All six supplied local NPZs
also passed the shape, byte-range and label-space checks; count discrepancies
are documented above. The GitHub Actions workflow separately runs Python 3.10
with the repository's pinned requirements and CPU Torch wheels; consult its
actual run status before treating that environment as verified.

No full 100-epoch dataset experiment, multi-seed accuracy replication or v2
notebook epoch-resume test has been completed by this patch.

## Literature source

Meng et al., *Detection of Unknown Attacks Through Encrypted Traffic: A Gaussian
Prototype-Aided Variational Autoencoder Framework*, IEEE TIFS, 2025,
[doi:10.1109/TIFS.2025.3612141](https://doi.org/10.1109/TIFS.2025.3612141).
Reviewed the user's 16-page accepted-manuscript PDF, pp. 6-11, and completed
A1/A2 notebook outputs. Published accuracy is not a pass/fail unit test.
