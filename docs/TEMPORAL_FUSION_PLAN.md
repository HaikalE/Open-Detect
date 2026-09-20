# Temporal fusion: controlled extension, not a new replication claim

Branch: `codex/temporal-fusion`. Existing replication code/checkpoints remain unchanged.
Do not tag current HEAD as the trained baseline: recover the actual training
commit/config/split hashes from result provenance first. Proposed baseline tag:
`baseline-replication-v1`, only after that verification.

## Hypothesis and representation

Test whether packet timing adds unknown-rejection information beyond the image
branch and additional model capacity. BiGRU models sequence order; explicit IAT
adds elapsed-time information. These are complementary, not synonymous.

Initial aligned input: the SAME first eight eligible packets used by the current
`data/Preprocessing/utils.py`: image 32x32, 80 header + 48 payload bytes per packet.
Sequence columns: transport-payload length, direction relative to the first
eligible packet's sender, inter-arrival time in seconds (first IAT = 0).
Use retained packet timestamps, document filtering semantics, retain lengths/masks;
padding must not be processed as real packets. Never derive timing from image bytes.
Normalize size/IAT with training-only statistics; choose log1p(IAT) with documented
seconds scale; negative/nonfinite times are audit failures, not silently clamped.
TCP window is optional future ablation, not part of the initial three-feature claim.
32-packet temporal input is a separate observation-budget experiment, not an equal
comparison against an eight-packet image. BiGRU is offline over the observed window;
do not claim causal per-packet real-time inference without latency evaluation.

## Required data gate

Full manifest: stable sample ID, capture ID, sessionization rule/session ID,
packet/window indices, image SHA, sequence length, label namespace, split/fold.
Keep all views/windows/duplicates of the same source session in one split.
Audit source/capture overlap appropriate to the evaluation claim; identical-image
grouping alone does not establish flow- or capture-independent generalization.
Verify raw capture labels and timestamp precision. A hash match is only a candidate:
truncated images can collide across flows; combined datasets can repeat source rows.
If rebuilding changes samples or splits, retrain B0 on the exact same paired cohort.
Do not compare a new fusion cohort to old scores as if only architecture changed.

## Architecture after gate

Existing image ResNet pooled512 -> projection128; BiGRU(hidden64/direction) ->
projection128; concatenate256 -> mu/logvar128. Keep Gaussian class prototypes,
image reconstruction decoder and image lateral connections. Keep initial objective
and minimum-KL rejection scoring; calibrate threshold on known validation only.
Do not reuse baseline thresholds after changing the representation. Unknown test
classes must not tune architecture, normalization, threshold or early stopping.
Explicitly decide how image augmentations interact with paired sequence semantics;
do not silently change baseline augmentation only for one arm.

## Experiments and acceptance

Start one fixed scenario/fold from the existing protocol (choice follows data gate),
small smoke run before full-budget pilot. B0: image-only on paired data. E1: fusion
with length+direction. E2: E1+IAT. C1: parameter-matched image-only capacity control.
Keep samples, splits, seeds, observation horizon and training budget matched.
Later: same-seed repeated runs, timestamp/ordering perturbation and temporal-only
baseline with its objective stated explicitly (no image reconstruction available).
Report closed accuracy/F1, AUROC, open metrics/known acceptance/unknown rejection,
validation calibration, parameters, epoch time and peak memory. No guaranteed gain.
Pilot passes engineering correctness and fair comparison, not an arbitrary accuracy
increase. Inspect generalization with locked tests only after design selection.

## File map

| Stage | Files | Change |
|---|---|---|
| P0 implemented | `pilot/audit.py`, `pilot/build_colab.py`, `pilot/01_AUDIT_DATA_CPU.ipynb`, `tests/test_pilot_audit.py` | Read-only source audit, bounded downloads, separate report upload |
| P1 planned | `data/Preprocessing/build_paired_dataset.py`, `data/paired.py` | Source manifest, session/window alignment, leakage checks and paired loader |
| P2 planned | `networks/temporal.py`, `networks/fusion.py`, `model_fusion.py` | Mask-aware BiGRU, concat projections, original probabilistic head/loss |
| P2 planned | `networks/resnet.py` | Optional pooled-feature interface while preserving original forward behavior |
| P2 planned | `train_fusion.py`, `run_fusion.py`, `tests/test_temporal_fusion.py` | Paired training, shapes/gradients/padding/resume tests |
| P2 planned | `test.py`, `provenance.py`, `resume_support.py` | Minimal adapter if needed; feature schema/normalizer/manifest hashes, compatible resume |
| P2/P3 planned | `pilot/02_TRAIN_FUSION.ipynb`, evaluation notebook | Create after P1, separate training and CPU upload; no change to existing scenario Colabs |

## PDF references and boundaries

- `TSP_CMES_83669.pdf`, section 3.3 / Table 4 and section 3.4: feature taxonomy,
  packet size/direction versus timing/periodicity and non-one-to-one associations.
  Survey justification only, not a prescribed BiGRU architecture or unknown-attack proof.
- `MalDIST_From_Encrypted_Traffic_Classification_to_Malware_Traffic_Detection_and_Classification.pdf`,
  section III, PDF pp.2–3 / Fig.2: packet metadata BiGRU branch, 32 packets,
  direction/size/IAT/window; source inspiration for multimodal fusion. Its binary/
  family classification evaluation is NOT OpenDetect unknown-class rejection.
- `j__jnca___encrypted_traffic_classification_via_multimodal_multitask_deep_learning.pdf`
  (DISTILLER), PDF p.7 feature definitions, p.9 input construction: transport-payload
  length, direction, IAT, TCP window, truncation/padding. Do not call payload length
  total on-wire packet length. Do not copy its ports/statistics or 32 packets blindly.
- OpenDetect implementation: `model.py`, `networks/resnet.py`, `test.py`,
  `data/grouped.py`, `data/Preprocessing/utils.py`: preserve the current objective,
  latent/prototype scoring and document the pairing-related protocol changes.

Enough literature for a data/engineering pilot. Before final thesis experiments,
confirm a dedicated open-set evaluation reference and frozen protocol; more broad
paper collecting is not a prerequisite for P0. No official public MalDIST repository
has been established here; absence of a found repo is not proof none exists.
