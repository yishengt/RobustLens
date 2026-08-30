# Final Gap Audit — AI Image Forensics Repository

Audit date: 30 August 2026

Scope: current working tree, existing checkpoint, non-training work only

Checkpoint: Bombek1 SigLIP2+DINOv2, 740,371,777 parameters

Repository validation: 470 tests passed, 337 subtests passed, 1 skipped

Status key:

- ✅ Complete and verified
- ⚠️ Partial
- ❌ Missing
- 🐛 Broken or misleading
- 🧪 Implemented but insufficiently tested
- ⏸️ Deferred
- 🚫 Not verifiable with the available data

## 1. Executive summary

### What the system actually is

The production path is a **binary pixel classifier with a deterministic
robustness and crop-analysis harness**. The checkpoint has two visual backbones
(SigLIP2 and DINOv2), but their features are concatenated into one classifier
head and produce one synthetic-origin logit. They are not separately calibrated
forensic channels and do not provide camera, provenance, transformation, or
localized-edit verdicts.

`DetectionPipeline.analyse_image()` in `src/pipeline/pipeline.py` runs one model
over the clean image, 14 generated transformations, and optional crops. The
shipped fusion is `0.7 * clean + 0.3 * mean(transformed)`. Patch scores are from
the same classifier and were removed from probability fusion after a negative
ablation. Spectral and residual statistics are optional diagnostics with proven
zero effect on `pred`.

### Plain answers

- **What works today?** Checkpoint loading, validation, CPU/MPS inference,
  deterministic single-transform and chain scoring, score consistency,
  crop-risk visualization, per-view Platt calibration, held-out robustness
  reporting, safe abstain-only mechanics, batch JSON output, data quarantine,
  and the Streamlit upload-ready UI.
- **What is partial?** Local-edit sensitivity, patch localization, evidence
  fusion, abstention validation, final-score calibration, shortcut analysis,
  failure analysis, stress testing, and device/size coverage.
- **What is missing?** Camera-origin evidence, EXIF/C2PA provenance parsing and
  verification, transformation detection, wavelets, spatial autocorrelation,
  cross-channel forensics, spatial patch grouping, edit masks/segmentation, a
  patch-count benchmark, unseen-camera evaluation, and a true same-source
  representation benchmark.
- **What cannot be established?** Dataset-source holdout (checkpoint training
  data is not recorded), unseen-generator performance (SID_Set has no generator
  identity), unseen-camera performance, CUDA behavior, and population-level
  chain robustness from the 12-image cache.
- **Can this be described as a multi-channel forensic detector?** **No.** Calling
  clean, transformed, and crop scores independent channels would rename repeated
  views of one classifier, not create independent evidence.
- **Smallest trustworthy claim:** this repository runs an existing binary
  AI-image classifier and reproducibly measures how its score changes under a
  fixed set of transformations and crops. On the cached sample it detects fully
  synthetic images much better than locally tampered images. It does not
  establish camera origin, provenance, edit localization, unseen-generator
  generalization, or dataset-source generalization.

Fine-tuning, consistency-loss training, training ablations, and post-training
recalibration were deliberately not run. Their empirical claims remain deferred.

## 2. Evidence standard and execution trace

Comments, YAML keys, and README claims were not accepted as implementation
evidence. A capability counts only when it is reachable from the actual path,
has a real value rather than a renamed score, and is supported by a test or
artifact.

The traced path is:

1. `validation.py:load_validated_image()` validates and decodes to RGB.
2. `transformations.py:generate_variants()` creates the clean view plus 14
   seeded variants. It applies transformations; it does not detect them.
3. `prediction.py:predict_variants()` sends every view through the same model.
4. `patches.py:analyse_patches()` sends crops through that same model.
5. `consistency.py:compute_consistency()` summarizes score spread.
6. `frequency.py:extract_features()` runs only when explicitly enabled;
   `frequency_probability()` always returns `None` in this build.
7. `fusion.py:fuse_predictions()` combines same-classifier view scores. The
   shipped mode excludes patch and frequency evidence.
8. `abstention.py:evaluate_abstention()` can replace a label with `Uncertain`
   but never flips it. The shipped policy is disabled.
9. `confidence.py:compute_confidence()` derives a heuristic score from
   decisiveness, agreement, and consistency.
10. `PipelineResult.as_simple_dict()` emits exactly `{image_path, pred}`.

The actual Bombek model path is
`src/models/bombek_siglip2_dinov2.py:forward_with_features()`: SigLIP2 and
DINOv2 features are concatenated and passed through one classification head.
Separate preprocessing does not make the two branches separate forensic claims.

## 3. Complete requirement matrix

| # | Requirement | Status | Executable evidence | Gap / conclusion |
|---:|---|---|---|---|
| 1 | Independent synthetic, camera, provenance, transformation, and localized-edit channels | 🐛 Broken or misleading | `BombekSigLIP2DINOv2Detector.forward_with_features`; `DetectionPipeline.analyse_image` | One synthetic-origin classifier exists. Transforms and crops reuse it. Camera and provenance channels do not exist. |
| 2 | Spectral, wavelet, residual, autocorrelation, and cross-channel features execute and affect inference | ⚠️ Partial | `frequency.py:extract_features`; `PipelineBehaviourTest.test_diagnostic_frequency_features_have_zero_effect_on_prediction` | FFT/DCT/high-pass/residual diagnostics execute only when enabled and have zero effect. Wavelet, autocorrelation, and cross-channel features are missing. |
| 3 | Missing camera/metadata evidence must not count as AI evidence | ✅ Complete and verified | `preprocessing.py` strips metadata; `test_metadata_presence_cannot_change_the_prediction`; missing fusion terms are dropped and weights renormalized | Passes, though partly vacuous because no camera channel exists. Metadata is displayed but never scored. |
| 4 | 4/9/16 patches, spatial grouping, crop robustness, and local-edit detection | ⚠️ Partial | `patches.py:generate_patch_boxes`; 4/9/16 budget test; `outputs/patch_ablation/ablation.json` | Budgets and crop heatmaps work. No neighboring-patch grouping, edit masks, region IoU, or validated localization. Tampered image recall is 0.455. |
| 5 | Transformations detected rather than merely applied | 🐛 Broken or misleading | `transformations.py:build_transform_specs` and `generate_variants` | JPEG, blur, resize, noise, color, and crop are generated for testing. No input-history or transformation detector exists. |
| 6 | Transformation-invariant training empirically demonstrated | ⏸️ Deferred | Loss/training code and unit plumbing exist under `src/finetune/`; no experiment was run in this scope | Unit tests cannot establish empirical benefit. Classification-only versus consistency-loss results are absent. |
| 7 | Fusion combines genuinely independent signals | 🐛 Broken or misleading | `fusion.py:fuse_predictions`; shipped `fusion.mode: rgb_transform` | It reweights clean/transformed outputs from one classifier. Patch output is also the same classifier; frequency cannot supply a probability. |
| 8 | Abstention is reachable, safe, calibrated, and chain-validated | ⚠️ Partial | `abstention.py:evaluate_abstention`; `tests/test_abstention.py`; `outputs/abstention/abstention_thresholds.json`; chain artifact | Rules are reachable in tests and never flip class. Validation selected no active rule; config is disabled; chain fitting was not performed; observed chain abstention is 0%. |
| 9 | Confidence calibration on held-out and transformed data | ⚠️ Partial | `outputs/calibration_robustness/calibration_robustness.json`; `outputs/confidence/confidence_report.json` | Per-view class probabilities are evaluated on 72 held-out images and 14 transforms. The final fused score has no post-fusion calibrator, and heuristic confidence reliability is not evaluated per transformation. |
| 10 | EXIF/C2PA or other provenance parsed and cryptographically verified | ❌ Missing | `ImageMetadata` contains filename/path/type/size/dimensions/mode only; no parser or verifier imports | No EXIF claim extraction, C2PA manifest parsing, signature validation, trust store, or provenance output. |
| 11 | Structured fields are computed and `{image_path, pred}` is preserved | ⚠️ Partial | `PipelineResult.as_simple_dict`, `as_detailed_dict`; batch contract tests; Streamlit download path | Simple contract is exact. Detailed diagnostics are computed or `null`. Missing forensic channels are not fabricated. Sensitivity and patch-risk fields are not edit history/segmentation. |
| 12 | Required benchmark families exist | ⚠️ Partial | Protocol, patch-mode, chain, calibration, format, and data-quality scripts/artifacts | Single-transform robustness and leakage are strongest. Patch-count, unseen-camera, true unseen-generator, and same-source representation benchmarks are absent or unverifiable. |
| 13 | Failure analysis covers high-confidence FP/FN, local edits, and should-abstain cases | ⚠️ Partial | `outputs/protocol_calibrated/metrics.json:failures`; example PNGs | FP/FN examples and tampered-family errors exist. There is no explicit high-confidence failure table, region-level local-edit error analysis, or should-have-abstained report. |
| 14 | Runtime, dependencies, malformed/color/size cases, CPU, MPS, and GPU verified | ⚠️ Partial | protocol runtime; validation/preprocessing/pipeline tests; live Streamlit; CPU format artifact | Guards and common formats are tested. Real CPU and MPS paths run, but the format CLI crashes on MPS; very-large inference and CUDA are not verified. |

## 4. Forensic component matrix

| Component | Status | Actual behavior |
|---|---|---|
| Synthetic-origin model | ⚠️ Partial | One 740.37 M-parameter dual-backbone binary classifier; empirically evaluated only on small local samples. |
| SigLIP2 branch | ✅ Verified plumbing | Supplies features to the shared head; no standalone forensic probability or calibration. |
| DINOv2 branch | ✅ Verified plumbing | Supplies features to the shared head; no standalone forensic probability or calibration. |
| Camera-origin / CFA / PRNU / ISP | ❌ Missing | No implementation. |
| Provenance / EXIF claims / C2PA | ❌ Missing | No implementation or cryptographic verification. |
| Transformation detector | ❌ Missing | Transformations are generated, not inferred from input. |
| Localized-edit detector | ⚠️ Partial | Same classifier scores crops; heatmap is an attention/risk aid, not an edit detector. |
| FFT radial profile / high-frequency ratio | ⚠️ Diagnostic only | Runs only when enabled; reported as descriptive values; zero effect on `pred`. |
| DCT energy ratio | ⚠️ Diagnostic only | Optional SciPy statistic; zero effect on `pred`. |
| Laplacian high-pass / box-blur residual | ⚠️ Diagnostic only | Reduced to scalar energy/std on luminance; zero effect on `pred`. |
| Wavelet | ❌ Missing | No DWT/wavelet code. |
| Spatial autocorrelation | ❌ Missing | No implementation. |
| Cross-channel forensics | ❌ Missing | Frequency path converts to luminance; no RGB relationship features. |
| Independent evidence fusion | ❌ Missing | Current fusion combines correlated views of one classifier. |

## 5. Patch, transformation, fusion, and abstention findings

### Patch analysis

`generate_patch_boxes()` accepts arbitrary positive caps. Focused tests now pin
4, 9, and 16 and verify unique spatial boxes. That is implementation coverage,
not an empirical patch-count comparison. The cached 60-image ablation compares
**modes**, not those three counts:

| Mode | F1 | Recall | FPR | Mean forward passes | Delta F1 vs off |
|---|---:|---:|---:|---:|---:|
| Off | 0.7761 | 0.650 | 0.050 | 0.0 | — |
| Coarse | 0.7576 | 0.625 | 0.050 | 4.0 | −0.0185 |
| Full | 0.7576 | 0.625 | 0.050 | 12.0 | −0.0185 |
| Top-k | 0.7576 | 0.625 | 0.050 | 12.0 | −0.0185 |

The evidence supports the shipped decision to keep patches as explainability
only (`fusion.mode: rgb_transform`). It does not establish local-edit
localization accuracy.

### Transformations and chains

The 14 single transformations are deterministic at seed 1234 and evaluated at
one frozen threshold. They do not identify what happened to an uploaded image.

The cached chain evaluation contains only 12 images, so it is a stress smoke
test:

| Condition | Mean score drift | AI-image drift | Accuracy | Recall | F1 | Abstention |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 0.0000 | 0.0000 | 0.667 | 0.625 | 0.714 | 0% |
| Screenshot + reshare | −0.0576 | −0.0334 | 0.750 | 0.625 | 0.769 | 0% |
| Generation depth 5 | **−0.0956** | **−0.0925** | 0.667 | **0.500** | 0.667 | 0% |
| Generation depth 10 | −0.0870 | −0.0815 | 0.667 | **0.500** | 0.667 | 0% |

Longer degradation pushes scores toward “authentic.” This is false-negative
pressure, not transformation detection.

### Fusion

On the 72-image held-out sample, whole-image AUC is 0.856771 and shipped
whole-plus-transform AUC is 0.857639, a gain of 0.000868. This is a small
same-model ensemble effect. It is not evidence of independent-channel fusion.

### Abstention

The unit-tested safety invariant is good: a firing rule only changes the label
to `Uncertain` and retains `label_before_abstention`; it cannot output the
opposite class. Practical validation is negative:

- fit split: 48; held out: 72;
- pre-registered bars: at least 1.5× error enrichment and at most 35%
  abstention;
- selected active rules: none;
- held-out coverage: 100%; abstention: 0%; error rate: 0.1806;
- chain validation was not part of threshold fitting.

The system **can** abstain in controlled tests, but the current deployed config
does not, and no validated policy passed the stated bars.

## 6. Calibration and structured output

The Platt calibrator was fitted on 48 clean validation scores. On the disjoint
72-image held-out split:

| Condition | ECE | Accuracy | AUC |
|---|---:|---:|---:|
| Clean | 0.1721 | 0.8194 | 0.8568 |
| Worst transformed ECE (`noise_s0.02`) | 0.1868 | 0.8333 | 0.8568 |

ECE degradation is 0.0148, inside the pre-registered 0.05 bar. This verifies
the **per-view class-probability calibrator** at hackathon scale. It does not
verify a post-fusion calibration model, because fusion occurs after each view
is calibrated. It also does not prove that the heuristic `ConfidenceReport`
score is calibrated under each transformation.

The public simple output remains exactly:

```json
{"image_path": "path/to/image.jpg", "pred": 0.84}
```

No timestamp or diagnostic key enters this two-key record. Detailed output now
distinguishes:

- `raw_probability`: clean-view raw model score;
- `calibrated_probability`: clean-view calibrated score, otherwise `null`;
- `final_probability` / `pred`: fused score;
- `probability_kind`: whether fusion consumed calibrated per-view values.

`estimated_manipulation_severity` is computed from transformation sensitivity,
not recovered edit history. `highest_risk_region` is a highest-scoring crop,
not an edit mask. Missing camera/provenance channels are omitted rather than
filled with copies of `pred`.

## 7. Benchmark inventory

| Benchmark | Status | Evidence / result |
|---|---|---|
| Single-transform robustness | ✅ Complete at current sample size | `outputs/protocol_calibrated/metrics.json`, 72 held out, 14 transforms, fixed threshold 0.69; clean accuracy 0.8194, worst 0.7778. |
| Chain stress | 🧪 Implemented but insufficiently tested | `outputs/chains/chain_metrics.json`, deterministic, n=12. |
| Patch modes | ✅ Complete for modes | `outputs/patch_ablation/ablation.json`, n=60; no mode improved F1/recall. |
| Patch counts 4/9/16 | ❌ Missing empirically | Geometry is tested; no three-way performance/runtime table. |
| Local-edit family | ⚠️ Partial | Tampered-vs-authentic n=46; image-level recall 0.4545; no masks or localization metric. |
| Unseen generator | 🚫 Not verifiable | SID_Set lacks per-generator IDs. `validate_generator_generalisation_config()` fails closed when labels/holdouts are absent. |
| Unseen camera | 🚫 Not verifiable | No camera model labels and no camera channel. |
| Dataset-source holdout | 🚫 Not verifiable | Cached checkpoint metadata does not identify training data; SID_Set overlap cannot be excluded. |
| File-format shortcut | ⚠️ Partial | `outputs/data_quality/file_format_shortcut.json`, 16+16 CPU run: AUC 1.0 before/after JPEG, 31/32 decisions preserved, mean absolute drift 0.0192. |
| Leakage/quarantine | ✅ Complete for exact hashes/groups | 25,337 files; 0 exact cross-split duplicates; 6 contradictory pairs; 12 files quarantined. |
| Near duplicates | ⚠️ Partial | One dHash-distance-2 cross-split pair; no semantic embedding audit. |
| Malformed/format stress | ✅ Verified in tests | Empty, corrupt, truncated, unsupported, grayscale, RGBA, size and pixel limits. |
| Same-source consistency | ❌ Missing | Existing `consistency_score` is within-image transform score spread, not a same-source representation benchmark. |

### File-format interpretation

Uniform JPEG re-encoding retained AUC 1.0, which weakens the hypothesis that
the classifier is simply reading PNG-vs-JPEG containers. One decision changed
and the mean absolute score drift was 0.0192. The control does **not** eliminate
compression-history, resolution, aspect ratio, subject, border, watermark,
color-profile, or generator-style shortcuts.

## 8. Failure analysis

The held-out calibrated protocol has one false positive and twelve false
negatives at threshold 0.69. The false positive is a real image scoring 0.8196.
All twelve false negatives are locally tampered images; the tampered-family
recall is 10/22 = 0.4545, versus 26/26 for fully synthetic images.

This is useful but incomplete:

- failure images are exported in `outputs/protocol_calibrated/examples/`;
- failure records do not explicitly rank by the heuristic confidence score;
- no edit masks exist, so localized false-region/IoU analysis is impossible;
- there is no table of errors that a candidate abstention rule should have
  withdrawn;
- because no abstention candidate passed, no abstained-error enrichment exists
  beyond the negative 0% result.

## 9. Runtime and operational verification

| Area | Status | Evidence / limitation |
|---|---|---|
| Runtime | ✅ Verified on MPS sample | Protocol cache: 120 images, 15 versions + 12 patches, 27 forwards/image, mean 13.3424 s/image, total 1601.09 s. |
| Dependencies | ⚠️ Partial | `requirements.txt`, setup doctor, import checks. HF processors may attempt Hub HEAD requests unless cached/offline. Versions use lower bounds rather than a lock file. |
| Malformed files | ✅ Verified | Corrupt/truncated/empty/unsupported files raise clear validation errors; batch mode records failures. |
| Grayscale | ✅ Verified | Validation conversion and mock end-to-end pipeline test. |
| Alpha channel | 🧪 Insufficiently tested | RGBA validation/preprocessing tests pass; no real-checkpoint RGBA benchmark. Alpha is discarded during RGB conversion. |
| Very small images | ✅ Guard verified | Below configured minimum is rejected; single-patch cases are safely skipped. |
| Very large images | 🧪 Insufficiently tested | Max-side/pixel/file-size guards are unit-tested; a boundary-size real inference was not run. |
| CPU | ✅ Verified | Mock end-to-end tests plus real 16+16 format control. |
| Apple MPS | ⚠️ Partial | Real Streamlit load and cached protocol succeed; the format CLI exited 139 on a 2-image valid smoke run. |
| CUDA GPU | 🚫 Not verifiable | No CUDA device is available in this environment. |

The Streamlit app was opened in a browser and rendered the real checkpoint as
`bombek_siglip2_dinov2`, 740.37 M parameters, device `mps (Apple Metal)`, and
14 transformations. It reached the upload-ready state. Automated browser file
upload could not proceed because the Chrome extension lacks file-URL access;
the rendered patch wording is separately exercised by a behavioral test.

### Observed verification failures

These were retained rather than converted into passes:

- `scripts/setup.py --check` exited 1 because the required sample file
  `data/cifake_sample/0000.jpg` is missing. The working tree already records
  that deletion; it was not restored or discarded during this audit.
- The valid two-image real-checkpoint format smoke command exited 139 on MPS.
  The equivalent CPU command and the full 16+16 CPU control succeeded.
- Browser automation reached the upload-ready UI, but Chrome file upload was
  blocked because the ChatGPT browser extension lacks file-URL access.
- One JPEG re-encode idempotence test was skipped because decoded pixels were
  not identical on this Pillow/platform combination. The byte-hash and other
  pixel-hash quarantine tests passed.

## 10. Critical missing capabilities

1. Genuine independent camera-origin, provenance, transformation, and
   localized-edit channels.
2. Camera forensics (CFA, PRNU/sensor, ISP traces) with an explicit
   “unavailable” state that contributes no negative evidence.
3. EXIF claim parsing and C2PA signature/trust verification.
4. Transformation detection rather than only synthetic test generation.
5. Validated local-edit localization with masks, spatial grouping, and
   region-level metrics.
6. Wavelet, spatial-autocorrelation, and cross-channel feature families.
7. A genuinely independent fusion model; the current views are correlated.
8. A validated chain-fitted abstention policy that meets fixed bars.
9. Post-fusion calibration and transformed reliability for the heuristic
   confidence score.
10. True unseen-generator, unseen-camera, patch-count, and same-source
    representation benchmarks.
11. Complete failure categories, especially should-have-abstained cases.
12. Reproducible CUDA verification and resolution of the MPS shortcut-CLI crash.

## 11. Evaluation and dataset risks

- **Unknown checkpoint training data:** source overlap with SID_Set cannot be
  ruled out. This blocks dataset-source generalization claims.
- **Small protocol:** 48 validation and 72 held out; confidence intervals are
  not reported.
- **Very small chain cache:** n=12, unsuitable for population conclusions or
  abstention fitting.
- **No generator/camera identities:** unseen-generator and unseen-camera claims
  cannot be evaluated.
- **Local-edit label quality:** six byte-identical pairs carry opposing labels
  (five train, one test). Both sides are quarantined rather than guessed.
- **One near duplicate:** dHash distance 2 across train/test remains a potential
  semantic leakage risk.
- **No edit masks:** all 25,337 local-edit files lack matching masks, so crop
  heatmaps cannot be validated as edit localization.
- **Format confound:** fully synthetic examples are PNG while real examples are
  JPEG/MPO in the inspected SID_Set sample. The JPEG control weakens but does
  not eliminate associated compression-history shortcuts.
- **Threshold sample size:** the 0.69 threshold and abstention sweeps use only
  48 validation images.
- **Cached artifact provenance:** several JSON files contain absolute local
  paths and do not cryptographically bind checkpoint, code revision, and data.
- **Cached protocol metadata mismatch:** re-analysis from cached scores writes
  the current `patches.mode: coarse` configuration (cap 4) while retaining the
  original cached runtime summary of 12 patches/image. Whole-image and
  transformation score fields remain explicit, but the regenerated protocol
  file is not accepted as patch-mode provenance.

## 12. BEFORE versus AFTER for this final audit

| Before | After | Verification |
|---|---|---|
| Detailed JSON always called the fused score `calibrated_probability`, even without a calibrator | Clean-view calibrated value is reported only when present; fused score has an explicit `probability_kind` | Calibration and pipeline tests |
| Metadata neutrality was stated but not pinned end to end | Identical pixels with/without attached EXIF-like metadata produce exactly the same prediction | Focused pipeline test |
| Generic patch cap accepted 4/9/16 but no focused guard named them | 4, 9, and 16 unique spatial budgets are tested | Focused patch geometry test |
| File-format control lived mainly as a scratch/README result | Real-checkpoint 16+16 CPU artifact persisted at `outputs/data_quality/file_format_shortcut.json` | Frozen-threshold CLI run |
| Audit mixed old and current claims and called derived scores channels | This document traces the production path and marks missing/deferred/unverifiable work explicitly | Source, tests, artifacts, commands below |
| Fine-tuning infrastructure could be mistaken for empirical completion | Training-dependent Tasks 2, 3, and 10 are explicitly deferred | No training command was run |

No model training, fine-tuning, consistency-loss experiment, ablation training,
or post-training recalibration was run during this audit.

## 13. Reproducible validation commands

```bash
# Environment and repository validation
./.venv/bin/python scripts/setup.py --check
./.venv/bin/python -m pytest -q
./.venv/bin/ruff check src scripts app.py tests
./.venv/bin/python -m compileall -q src scripts app.py tests model.py
git diff --check

# Deterministic non-training evaluations
./.venv/bin/python scripts/audit_dataset_quality.py
./.venv/bin/python scripts/evaluate_chains.py --reuse-scores
./.venv/bin/python scripts/evaluate_calibration_robustness.py

# Real-checkpoint file-container control on CPU
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  ./.venv/bin/python scripts/evaluate_format_shortcuts.py \
  --authentic-dir data/extracted/sid_set/real \
  --synthetic-dir data/extracted/sid_set/ai_generated \
  --per-class-limit 16 --output-format JPEG --device cpu

# Demo
./.venv/bin/streamlit run app.py
```

The full protocol can be re-analyzed from cached scores without model forwards;
do not treat it as a new independent sample. The command also serializes the
current configuration beside old scores, so compare configuration metadata to
the cached runtime before using it as implementation provenance:

```bash
./.venv/bin/python scripts/evaluate_protocol.py \
  --reuse-scores --output-dir outputs/protocol_calibrated \
  --calibration outputs/calibration.json --operating-point balanced
```

## 14. Tasks 1–11 status

| Task | Status | Final audit conclusion |
|---:|---|---|
| 1 — pipeline validation/integration | ✅ Complete for non-training path | Real checkpoint/UI startup, exact JSON contract, calibration wording, and current inference path verified. |
| 2 — local-edit fine-tuning | ⏸️ Deferred | Infrastructure may exist; no training run or empirical result is claimed. |
| 3 — transformation-invariant objective/ablation | ⏸️ Deferred | Loss plumbing is not empirical evidence. Classification-only versus consistency-loss table is absent. |
| 4 — abstention | ⚠️ Partial | Safe abstain-only mechanics pass; no candidate met fixed bars and chain fitting is absent. |
| 5 — spectral cleanup | ✅ Complete for diagnostic-only scope | FFT/DCT/residual diagnostics are optional and tested to have zero probability effect. Not a forensic channel. |
| 6 — transformation/chain robustness | ✅ Complete at current sample sizes | 14 transforms on n=72 and deterministic chains on n=12, one frozen threshold. |
| 7 — generator generalization | 🚫 Not verifiable | SID_Set lacks per-generator labels; configuration fails closed. |
| 8 — repository tests | ✅ Complete for implemented behavior | Full suite, lint, compilation, imports, leakage, shortcut, safety, UI, and contract checks run. |
| 9 — demo/UI explainability | ✅ Complete | Real upload-ready UI verified; patch map explicitly disclaims segmentation/proof; unavailable Grad-CAM section removed. |
| 10 — post-fine-tuning recalibration | ⏸️ Deferred | No fine-tuned model exists in scope; no post-training calibration was run. |
| 11 — audit/reproducibility | ✅ Complete | This matrix, negative findings, artifacts, commands, risks, and roadmap are recorded. |

## 15. Recommended roadmap

### P0 — truthfulness and decision safety

1. Keep the public claim to “binary classifier plus robustness diagnostics”; do
   not call derived view scores independent forensic channels.
2. Record checkpoint training dataset, version/hash, calibration hash, and
   evaluation data manifest so source overlap can be audited.
3. Build a larger chain-validation split and fit abstention there without
   lowering the 1.5× enrichment / 35% abstention bars.
4. Diagnose the observed MPS `evaluate_format_shortcuts.py` exit 139 and add
   a real-checkpoint MPS smoke gate.
5. If fine-tuning resumes, quarantine conflicts first and report the deferred
   classification-only versus consistency-loss experiment honestly.

### P1 — missing acceptance criteria

1. Acquire local-edit masks and implement region-level validation, spatial
   grouping, isolated-patch suppression, and 4/9/16 empirical comparisons.
2. Add explicit post-fusion calibration and evaluate both final score and
   heuristic confidence on held-out transformed/chain data.
3. Add high-confidence FP/FN and should-have-abstained failure tables.
4. Extend shortcut controls to dimensions, aspect ratio, JPEG history,
   watermarks, borders, color profiles, content, and source.
5. Add a dataset with generator and camera identities; do not substitute
   generation-process families.

### P2 — independent forensic channels

1. Add camera-origin analysis with a true unavailable state and no penalty for
   missing evidence.
2. Add provenance parsing and cryptographic C2PA verification as a channel
   independent of pixels.
3. Prototype wavelet, autocorrelation, cross-channel, and spectral models only
   behind held-out ablations proving incremental value and robustness.
4. Add a genuine same-source representation-consistency benchmark.

### P3 — production hardening

1. Lock dependency versions and support offline processor loading explicitly.
2. Run boundary-size memory tests and real-checkpoint RGBA/grayscale tests.
3. Verify CUDA determinism/performance and publish per-device runtime/memory.
4. Add artifact manifests with code, checkpoint, config, data, seed, and command
   hashes.

## Final conclusion

The full specification is **not complete**. The repository is a careful and
useful robustness wrapper around one classifier, not a multi-channel forensic
detector. Its strongest defensible evidence is the executable inference path,
fixed-threshold transform evaluation, held-out per-view calibration, exact JSON
contract, and honest negative results for patches and abstention. Its most
important weaknesses are local edits, missing independent channels, unknown
checkpoint training provenance, and the absence of validated abstention under
compound degradation.

---

# Addendum — remaining-task completion

Everything in this section was executed in this tree. Each claim carries an
evidence grade: **Verified** (ran here, reproducible), **Smoke-tested** (ran end
to end on a deliberately tiny sample — proves the machinery, not the science),
**Blocked** (prerequisite missing).

> **Threshold provenance.** Results below use the frozen calibrated threshold
> **0.69** unless stated. The fine-tuning comparison and consistency ablation
> use **0.5**, because a fine-tuned head has no fitted threshold of its own and
> reusing 0.69 would import a calibration fitted for a different model. Earlier
> protocol numbers at **0.42** are *historical* and must not be compared with
> either.

## Local-edit fine-tuning — Smoke-tested, not adopted

Head-only run completed end to end on `configs/lora_finetune_smoke.yaml`.

| Split | Images | Groups | Subgroups |
|---|---|---|---|
| train | 68 | 25 | authentic 24, minor_edit 27, **synthetic replay 17** |
| validation | 17 | 8 | authentic 8, minor_edit 9 |
| test | 20 | 8 | authentic 8, minor_edit 12 |

Replay mixture realised **exactly 0.250** of the train set and is **train-only** —
validation and test contain no replay images.

**Bug found and fixed:** the replay mixture was sized against the *pre-limit*
train count (requesting 6,780 images for a 68-image run) and was added *before*
the group limit. Since every replay image is its own group, the limit then
discarded essentially all of them — the mixture silently contributed 0 images.
The mixture is now sized and added after quarantine and the group limit settle
the local-edit set. `max_groups_per_split` now explicitly caps **local-edit
source groups only**; replay is governed by `synthetic_mixture_fraction`.

### Integrity checks — Verified

| Check | Result |
|---|---|
| Adapter save → reload determinism | **0.0** max score difference across two independent reloads |
| Adapter actually changes predictions | Yes, max difference **0.2329** |
| Original checkpoint SHA-256 | `caae0c00…30d4b` — **unchanged** before and after training |
| Second LoRA adapter added | No — existing adapter tensors reused |

### Original vs fine-tuned — Smoke-tested (threshold 0.5, n=20 held out)

| Metric | Original | Fine-tuned | Δ |
|---|---|---|---|
| accuracy | 0.6000 | 0.6000 | +0.0000 |
| balanced accuracy | 0.5000 | 0.5000 | +0.0000 |
| F1 | 0.7500 | 0.7500 | +0.0000 |
| recall | 1.0000 | 1.0000 | +0.0000 |
| **AUROC** | 0.5104 | **0.3542** | **−0.1562** |
| FPR | 1.0000 | 1.0000 | +0.0000 |
| FNR | 0.0000 | 0.0000 | +0.0000 |

**Decision: do NOT adopt.** Ranking quality got worse and nothing improved. Both
models collapse to predicting every image AI-generated at threshold 0.5 on this
20-image set — which is what 68 training images should be expected to produce.
**These are machinery numbers, not evidence about the method.**

## Consistency-loss ablation — Smoke-tested

Three runs differing **only** in the loss. All variants saw two identically
generated transformed views per image, including the baseline, so the comparison
is controlled.

| Variant | F1 | Recall | FPR | AUROC | Balanced acc | Runtime |
|---|---|---|---|---|---|---|
| classification only | 0.7500 | 1.0000 | 1.0000 | 0.4479 | 0.5000 | 250.0 s |
| + logit-MSE consistency | 0.7500 | 1.0000 | 1.0000 | 0.4375 | 0.5000 | 122.5 s |
| + symmetric-KL consistency | 0.7500 | 1.0000 | 1.0000 | 0.4479 | 0.5000 | 113.5 s |

**Decision: keep classification-only.** Neither variant reached the
pre-registered +0.01 gain in F1 or recall. The loss code is retained and tested
but stays **disabled by default**.

*Limitation:* at 68 training images this ablation cannot detect a real effect.
It shows the machinery is correct and controlled, nothing more.

## Abstention fitted on chains — Verified

The single-transformation sweep selected nothing: drift and consistency rules
fired on **0.000** of images at every threshold, because one transformation
barely moves a score. Refitting on compound chains, with the **same
pre-registered bars** (≥1.5× error enrichment, ≤35% abstention rate):

| | Chain validation (n=24) | Chain held out (n=24) | Held out, abstention off |
|---|---|---|---|
| abstention rate | 0.292 | **0.125** | 0.000 |
| accuracy among answered | 0.765 | **0.810** | 0.750 |
| error enrichment | 1.90× | **2.67×** | — |

**Three rules accepted and frozen** into `configs/config.yaml`:
`borderline_margin 0.02`, `min_consistency 0.50`, `min_agreement 0.70`.

Rules that still never reached the bar are left at settings they **cannot fire
at** (`max_transformed_drift 1.0`, `boundary_crossing_fraction 2.0`) rather than
tuned to look active. **The bars were not lowered.** Abstention is now
`enabled: true`.

## Leakage and shortcut tests — Verified

`tests/test_leakage_confounds.py` (16 tests) plus the existing format probe.
Covers exact duplicates, near-duplicates, group leakage, transformed-copy
leakage, conflicting labels, and confounds in file format, resolution, aspect
ratio and compression history, plus filename leakage.

**Bug found and fixed:** `dataset_confounds` used `zip()` on paths and labels,
which silently truncated to the shorter list and dropped images from the audit.
A length mismatch is now an error.

The SID_Set format confound is pinned by a test against the real data
(full_synthetic **100% PNG**, real **0% PNG**) so the documented conclusion
cannot drift from what the data looks like. Re-encoding both classes to a common
format left AUC and separation unchanged, so the confound is present in the data
but is **not** what the detector reads.

## Patch explainability — Verified

The demo shows the required sentence verbatim:

> "A highlighted region is a suspicious region that influenced the model's
> score. It is not proof of AI editing, a segmentation mask, or a reconstruction
> of editing history."

It also states that patch evidence carries **zero weight** in both probability
and confidence. Coverage is displayed, unmeasured areas stay untinted, patches
are skipped for confident images, and Grad-CAM unavailability is explained.
Pinned by `tests/test_app_contract.py`.

## Recalibration — Blocked

No fine-tuned model was adopted, so there is nothing to recalibrate. The
existing calibration and the frozen **0.69** threshold stand unchanged. Threshold
`0.69` was **not** automatically reused for the fine-tuned comparison — that used
0.5, since a calibration fitted for one model does not transfer to another.

## Remaining limitations

- Every fine-tuning number here is **smoke scale** (68 train / 20 test images). A
  real run needs the full 20,330-image train split, which is a multi-hour job
  that was not started without approval.
- The consistency ablation cannot resolve a real effect at this sample size.
- Chain abstention was fitted on 48 images split 24/24. The held-out gain
  (0.750 → 0.810) rests on 24 images and needs confirmation at larger scale.
- `moderate_edit` and `transformed` subgroups are **empty** in the current
  dataset — the escher-vismin export provides only one edit severity, so
  minor/moderate cannot yet be reported separately.
- Masks remain absent for all 25,337 images, so heatmap-overlap evaluation
  against ground-truth edit regions is still impossible.
- True unseen-generator generalisation remains **not established**.
