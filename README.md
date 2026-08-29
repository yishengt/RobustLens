# Robust Detection of AI-Generated Images Under Real-World Transformations

An **inference-only** pipeline that estimates whether an image is AI-generated,
and — more importantly — whether that estimate *survives* the things that
happen to images in the real world: JPEG recompression, blurring, downscaling,
noise, colour edits and cropping.

The project loads an existing trained checkpoint. It does not train models.

---

## Problem statement

Images shared online are rarely pristine. By the time a picture has been
uploaded, recompressed by a platform, screenshotted and reposted, many
AI-detection models that scored well on clean benchmarks collapse — they had
learned fragile, high-frequency cues that compression destroys.

This project addresses that directly. For every input image it also builds
14 transformed versions, classifies all of them, and reports both a fused
prediction and a **transformation-consistency score** that says how stable the
verdict actually was. A confident-looking score with low consistency is a
warning sign, and the pipeline surfaces it rather than hiding it.

---

## Pipeline

```
                        Input image
                             │
                    ┌────────▼────────┐
                    │   Validation    │  exists · format · not corrupted
                    │                 │  dimensions · opens · RGB-convertible
                    └────────┬────────┘
                             │  full-resolution RGB image + metadata
                    ┌────────▼────────┐
                    │  Preprocessing  │  RGB · resize · normalize · tensor
                    │                 │  identical for every image version
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌────────▼─────────┐
│  Whole-image   │  │  Patch-level    │  │  Transformation  │
│   detection    │  │   detection     │  │     testing      │
│                │  │ overlapping     │  │ JPEG ×4 · blur ×3│
│  p(AI) for the │  │ tiles → per-    │  │ resize ×2 · noise│
│  entire frame  │  │ patch p(AI)     │  │ ×3 · jitter·crop │
└───────┬────────┘  └────────┬────────┘  └────────┬─────────┘
        │                    │                    │
        │            heatmap + top regions   14 transformed
        │                    │               predictions
        │                    │                    │
        │                    │           ┌────────▼────────┐
        │                    │           │   Consistency   │
        │                    │           │ mean·min·max·std│
        │                    │           │ range · score   │
        │                    │           └────────┬────────┘
        │                    │                    │
        └──────────┬─────────┴────────────────────┘
                   │
          ┌────────▼─────────┐
          │  Fusion          │  0.60 · whole image
          │                  │  0.20 · transformed mean
          │                  │  0.20 · patch evidence
          └────────┬─────────┘  (missing terms are dropped and
                   │             the rest renormalised)
          ┌────────▼─────────┐
          │   Confidence     │  decisiveness · version agreement
          │                  │  consistency · patch agreement
          └────────┬─────────┘
                   │
          ┌────────▼─────────┐
          │ Explainability   │  patch heatmap · charts
          └────────┬─────────┘  (Grad-CAM where the model supports it)
                   │
           ┌───────▼────────┐
           │  Final output  │  label · p(AI) · p(real) · confidence
           │                │  consistency · highest-risk region
           └────────────────┘
```

Transformations are applied to the **original full-resolution image**, then put
through exactly the same preprocessing as the original. This mirrors how images
degrade in the wild, rather than degrading an already-downsampled thumbnail.

### Why whole-image detection is not enough

A whole-image score answers *"is this picture synthetic?"*. It is much weaker at
*"was one region of this otherwise real photo replaced?"* — a single edited
object is averaged away across the whole frame, so a locally manipulated image
can score as confidently authentic.

Patch-level detection addresses that directly: the image is tiled into
overlapping patches, each patch goes through the **same** model and
preprocessing as the full image, and the per-patch scores are reconstructed
into a heatmap. A locally edited region can then raise the fused score even
when the frame as a whole looks authentic.

A hot patch means the detector responded strongly to that region. It is a
**potentially manipulated region worth a human look — not proof that the region
was edited.**

---

## Quickstart

Every command below is copy-pastable from the repository root and calls
`./.venv/bin/python` directly, so you never need to activate the virtualenv.

### 1. Install dependencies

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

### 2. Download the detector checkpoint

2.11 GB, roughly 8 minutes. `-C -` resumes if the connection drops, so it is
safe to re-run.

```bash
mkdir -p models/pretrained
curl -L --fail --progress-bar -C - \
  https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2/resolve/main/pytorch_model.pt \
  -o models/pretrained/pytorch_model.pt
```

Check it arrived intact — the size must be exactly `2105483083`:

```bash
ls -l models/pretrained/pytorch_model.pt
```

### 3. Run your first prediction

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir data/cifake_sample \
  --checkpoint models/pretrained/pytorch_model.pt \
  --config configs/config.yaml \
  --no-transformations \
  --output outputs/cifake_predictions.json

cat outputs/cifake_predictions.json
```

Expected shape (`pred` is the AI-generated probability):

```json
[
  {
    "image_path": "/abs/path/to/data/cifake_sample/0000.jpg",
    "pred": 0.009768
  }
]
```

`image_path` is absolute by default. Add `--relative-paths` to write paths
relative to `--input-dir` instead:

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir data/cifake_sample \
  --checkpoint models/pretrained/pytorch_model.pt \
  --no-transformations \
  --relative-paths \
  --output outputs/cifake_predictions.json
```

No config edit is needed. The loader identifies this checkpoint from its tensor
signature and selects `bombek_siglip2_dinov2` automatically, overriding
`model.name` in the config.

### 4. Launch the web demo

```bash
./.venv/bin/streamlit run app.py
```

**One manual step:** the sidebar defaults to `checkpoints/best.pt`, which does
not exist, so the app opens on *"Model setup required"*. Paste this into the
sidebar's **Model checkpoint** field:

```
models/pretrained/pytorch_model.pt
```

To make the default work instead, link it once:

```bash
ln -s ../models/pretrained/pytorch_model.pt checkpoints/best.pt
```

---

## Common tasks

All paths are relative to the repository root.

### Score a folder of images

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir path/to/your/images \
  --checkpoint models/pretrained/pytorch_model.pt \
  --no-transformations \
  --output outputs/predictions.json
```

### Get the full report (label, confidence, per-transformation scores)

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir path/to/your/images \
  --checkpoint models/pretrained/pytorch_model.pt \
  --no-transformations \
  --output outputs/predictions.json \
  --detailed-output outputs/detailed.json
```

### Run the full robustness sweep

Drops `--no-transformations`, so all 15 image versions are scored and
`transform_consistency` becomes meaningful. Roughly 110 s per image on CPU.

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir data/cifake_sample \
  --checkpoint models/pretrained/pytorch_model.pt \
  --output outputs/predictions.json \
  --detailed-output outputs/detailed.json
```

### Get labelled evaluation data

Download a ~2 GB sample of SID_Set, then unpack it into class folders:

```bash
./.venv/bin/python scripts/download_dataset.py --split validation --shards 4 --yes
./.venv/bin/python scripts/extract_dataset.py
```

That writes `data/extracted/sid_set/{real,ai_generated}/` plus a `labels.json`
manifest. For a smaller balanced sample:

```bash
./.venv/bin/python scripts/extract_dataset.py --per-class-limit 100
```

### Benchmark accuracy and robustness

```bash
./.venv/bin/python scripts/evaluate_dataset.py \
  --data-dir data/sid_set \
  --checkpoint models/pretrained/pytorch_model.pt \
  --limit 200 \
  --output outputs/benchmark.json
```

### Run the checks

```bash
./.venv/bin/python -m pytest tests/ -q
./.venv/bin/python -m ruff check src scripts app.py tests
```

---

## Runtime expectations

The detector is 740 M parameters, so it is not instant on CPU:

| Operation | CPU time |
|---|---|
| One-time model load | ~14 s |
| Per image, `--no-transformations` | ~7 s |
| Per image, full 15-version sweep | ~110 s |

`--device cpu` is the tested path. `--device mps` (Apple GPU) and
`--device cuda` are wired up but **untested with this checkpoint's bfloat16
SigLIP weights** — if either errors or returns NaNs, fall back to CPU.

Two behaviours to expect, both intentional:

- **Grad-CAM reports "unavailable"** for this model. Attribution would have to
  cross two LoRA-adapted branches at different resolutions with different token
  grids, so the pipeline returns an explanation instead of a misleading
  heatmap. Probability, label, confidence and consistency are unaffected.
- **`explainability` is `null` in batch JSON.** Batch mode skips it for speed;
  the Streamlit path populates it.

---

## Installation

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

Activating the virtualenv is optional; every command in this README calls
`./.venv/bin/python` directly. If you prefer to activate it:

```bash
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

Requires Python 3.9+. Torch is selected automatically for your platform;
CUDA, Apple MPS and CPU are all supported (`inference.device: auto`).

---

## Model checkpoint setup

**A trained checkpoint is required.** Without one, every entry point stops with
an actionable setup error — the pipeline never fabricates predictions.

Place your checkpoint at `checkpoints/best.pt`, or pass `--checkpoint`.
Full format details, supported architectures and the preprocessing contract are
in [`models/README.md`](models/README.md).

Supported backbones, all far below the 2B parameter limit:

| `model.name`      | Parameters | Grad-CAM layer |
|-------------------|-----------:|----------------|
| `efficientnet_b0` |     ~5.3 M | `features[-1]` |
| `resnet18`        |    ~11.7 M | `layer4`       |
| `convnext_tiny`   |    ~28.6 M | `features[-1]` |
| `dual_backbone`   |     ~740 M | two processor-specific inputs |
| `bombek_siglip2_dinov2` | 740.4 M | unavailable (two LoRA branches) |

To test the plumbing before a real checkpoint exists:

```bash
./.venv/bin/python scripts/make_dummy_checkpoint.py --output checkpoints/dummy.pt
```

This writes **untrained** weights. It proves the wiring works; its predictions
are meaningless. It defaults to `resnet18` because a randomly initialised
EfficientNet-B0 is numerically dead in eval mode — its feature map collapses to
a standard deviation of ~1e-14 and every image returns an identical score,
which would mask real bugs.

For a trained SigLIP2 + DINOv2 checkpoint, set `model.name: dual_backbone` in
`configs/config.yaml` (or ensure the checkpoint records that architecture).
The loader uses the separate SigLIP2 and DINOv2 image processors automatically;
see [`models/README.md`](models/README.md) for the checkpoint contract.

### Real pretrained detector (Bombek1 SigLIP2 + DINOv2 LoRA)

The one trained checkpoint this repo is wired to use out of the box comes from
[Bombek1/ai-image-detector-siglip-dinov2](https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2).
Download it yourself (2.11 GB):

```bash
mkdir -p models/pretrained
curl -L --fail --progress-bar \
  https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2/resolve/main/pytorch_model.pt \
  -o models/pretrained/pytorch_model.pt
```

Then run it with the stock config — the loader recognises the checkpoint from
its tensor signature and selects the right architecture automatically:

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir data/cifake_sample \
  --checkpoint models/pretrained/pytorch_model.pt \
  --config configs/config.yaml \
  --no-transformations \
  --output outputs/cifake_predictions.json
```

This is a **different architecture** from the native `dual_backbone`
(PEFT-wrapped SigLIP2, timm DINOv2, LoRA on both branches, a 512/256 head).
It is registered separately as `bombek_siglip2_dinov2` and loads strictly;
see [`models/README.md`](models/README.md) for the full key-level comparison.

---

## Streamlit demo

```bash
./.venv/bin/streamlit run app.py
```

The sidebar defaults to `checkpoints/best.pt`. If your checkpoint is elsewhere,
paste its path into the **Model checkpoint** field, for example:

```
models/pretrained/pytorch_model.pt
```

Upload one JPG/JPEG/PNG/WEBP image to see:

- the original image and its metadata
- the final classification and the confidence statement
- AI-generated and real-image probabilities
- the confidence level and the transformation-consistency score
- a per-version prediction table and bar chart
- a drift chart showing which transformations moved the score most
- a **patch-risk heatmap** with the highest-risk regions outlined and tabulated
- a Grad-CAM heatmap (or a clear message if unavailable)
- a **Download detailed JSON** button
- the compact report schema and the full JSON result

The demo shows an explicit notice when probabilities are uncalibrated, and
clear messages for a missing checkpoint, an unsupported or corrupted image, and
any unavailable explainability component.

---

## Batch inference

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir path/to/images \
  --checkpoint models/pretrained/pytorch_model.pt \
  --output outputs/predictions.json
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--detailed-output PATH` | also write the detailed JSON report |
| `--no-transformations` | score the original image only (much faster) |
| `--device {auto,cpu,cuda,mps}` | override device selection |
| `--batch-size N` | override `inference.batch_size` |
| `--threshold F` | override the binary decision threshold |
| `--limit N` | process only the first N images |
| `--relative-paths` | write paths relative to `--input-dir` |
| `--on-error {skip,fallback}` | omit failed images, or record `--fallback-pred` |
| `--frequency` | enable the optional frequency module |

Exit codes: `0` success, `1` inference failure, `2` config error,
`3` model setup error, `4` input error.

### JSON output format

The simple format is the submission contract — `pred` is the AI-generated
probability in `[0, 1]`:

```json
[
  {
    "image_path": "images/example.jpg",
    "pred": 0.84
  }
]
```

The detailed format (`--detailed-output`) is a **superset**: it keeps every
field above and adds the report schema plus the full diagnostic detail.

The report fields, as surfaced in the demo's "Compact report schema" panel:

```json
{
  "image_path": "images/example.jpg",
  "raw_probability": 0.86,
  "final_probability": 0.84,
  "real_probability": 0.16,
  "label": "Likely AI-generated",
  "confidence": "high",
  "transformation_consistency": 0.84,
  "estimated_manipulation_severity": "medium",
  "highest_risk_region": {
    "index": 1, "x": 320, "y": 80, "width": 160, "height": 160, "score": 0.91
  },
  "per_transformation_predictions": {
    "clean": 0.86, "jpeg_q30": 0.79, "blur_s2": 0.76, "center_crop_80": 0.82
  }
}
```

`raw_probability` is the uncalibrated model score; `final_probability` is the
fused value and always equals `pred`. `per_transformation_predictions` reports
the untransformed image as `clean`. `highest_risk_region` is `null` when patch
analysis did not run.

The detailed record additionally carries `patch_analysis` (every patch's
coordinates and score, `heatmap_coverage`, and the settings used),
`fusion_detail` (weights, components and any `fallback_reason`),
`confidence_detail`, `consistency_detail`, `metadata`, `explainability` and
`errors`.

The remaining legacy fields:

```json
[
  {
    "image_path": "images/example.jpg",
    "pred": 0.84,
    "label": "Likely AI-generated",
    "confidence": "High",
    "real_probability": 0.16,
    "transform_consistency": 0.93,
    "transformations": {
      "jpeg_q90": 0.85, "jpeg_q70": 0.83, "jpeg_q50": 0.81, "jpeg_q30": 0.78,
      "blur_s0.5": 0.84, "blur_s1": 0.82, "blur_s2": 0.75,
      "resize_0.5x": 0.83, "resize_0.25x": 0.79,
      "noise_s0.02": 0.84, "noise_s0.05": 0.82, "noise_s0.1": 0.77,
      "color_jitter": 0.85, "center_crop_80": 0.80
    },
    "errors": [],
    "original_prediction": {"...": "..."},
    "predictions": ["... every version ..."],
    "consistency_detail": {"...": "..."},
    "fusion_detail": {"...": "..."},
    "confidence_detail": {"...": "..."},
    "metadata": {"...": "..."},
    "explainability": {"...": "..."}
  }
]
```

Images that fail validation appear in the detailed output with `"pred": null`
and a populated `errors` array. A single bad file never aborts a batch run.

---

## Patch-level analysis

Tiles the image into overlapping patches, scores each through the same model and
preprocessing as the full image, and reconstructs a risk heatmap.

### Cost

**Every patch is a full forward pass.** With the 740 M-parameter detector that
is ~7 s per patch on CPU, so a 12-patch image takes about two minutes. Because
of this:

- batch inference has patches **off by default**; opt in with `--patches`
- `patches.max_patches` caps the grid; larger grids are evenly subsampled
- the Streamlit demo runs them on the single uploaded image

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir path/to/images \
  --checkpoint models/pretrained/pytorch_model.pt \
  --patches \
  --output outputs/predictions.json \
  --detailed-output outputs/detailed.json
```

### Settings

All under `patches` in `configs/config.yaml`:

| Key | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | master switch |
| `patch_size` | `256` | window size in original-image pixels |
| `stride` | `192` | `< patch_size` produces overlapping patches |
| `min_patch_size` | `64` | below this the image is too small to tile |
| `max_patches` | `12` | hard cap; larger grids are evenly subsampled |
| `top_k` | `3` | how many highest-risk patches to report |
| `heatmap_threshold` | `0.5` | highlight cut-off in the demo |
| `evidence_statistic` | `top_k_mean` | `top_k_mean`, `max` or `mean` |

`evidence_statistic` chooses the single number fed to fusion. `top_k_mean` is
the default because a bare `max` over many patches drifts upward simply as the
maximum of many noisy scores, which inflates false positives on authentic
images.

### When patch analysis is skipped

It degrades to a clear message and the whole-image result is unaffected when:

- `patches.enabled` is false;
- the image is smaller than `min_patch_size` on either side;
- the image fits in a **single** patch — one whole-image tile would only repeat
  the whole-image score, and feeding that back as independent "patch evidence"
  would double-count one measurement and make patch agreement trivially 1.0;
- a patch forward pass fails.

### Reading the heatmap

Regions no patch scored are left **untinted**, not drawn cold. An unmeasured
area must not read as "confidently authentic", so the demo reports the covered
fraction whenever it is below 100 %.

A warm patch means the detector responded strongly there. That is evidence
about where the model looked — **not proof that the region was edited.**

---

## Transformation details

All 14 transformations are configurable under `transformations` in
`configs/config.yaml`.

| Transformation | Values | Simulates |
|---|---|---|
| JPEG compression | quality 90, 70, 50, 30 | platform recompression |
| Gaussian blur | sigma 0.5, 1.0, 2.0 | soft focus, re-encoding |
| Downscale → upscale | 0.5×, 0.25× | thumbnailing and re-upload |
| Gaussian noise | sigma 0.02, 0.05, 0.10 | sensor noise, transmission |
| Colour jitter | ±20% brightness/contrast/saturation | filters and light edits |
| Centre crop | keep 80% of each side | cropped reposts |

Stochastic transforms (noise, jitter) derive their random state from
`transformations.seed` plus the transform name, so results are reproducible and
independent of ordering.

---

## Configuration

Everything lives in [`configs/config.yaml`](configs/config.yaml). No paths are
machine-specific; relative paths resolve against the project root.

| Section | Controls |
|---|---|
| `data` | image size, supported extensions |
| `validation` | min/max dimensions, pixel budget, file-size cap |
| `model` | architecture, class count, parameter limit |
| `normalization` | preprocessing mean and standard deviation |
| `transformations` | which transforms to generate, and their parameters |
| `labels` | the three-way label boundaries |
| `consistency` | how spread is converted to a stability score |
| `fusion` | fusion mode and weights |
| `frequency` | the optional frequency module (off by default) |
| `confidence` | confidence weights and High/Medium/Low bands |
| `explainability` | Grad-CAM toggle and overlay strength |
| `inference` | batch size, threshold, device |
| `calibration` | Platt calibration file, FPR target, selected-threshold policy |

### Label bands

| AI probability | Label |
|---|---|
| 0.00 – 0.39 | Likely authentic |
| 0.40 – 0.59 | Uncertain |
| 0.60 – 1.00 | Likely AI-generated |

### Fusion

Default mode, `whole_patch_transform`:

```
final = 0.60 · whole_image
      + 0.20 · mean(transformed versions)
      + 0.20 · patch_evidence
```

Whole-image scoring carries the most weight because it is what the model was
trained to produce. The transformed mean rewards predictions that survive
redistribution. The patch term lets a locally edited region raise the score on
an image that looks authentic overall.

**Every term degrades safely.** If patch analysis is unavailable — disabled, the
image is too small to tile, or a patch pass failed — its weight is dropped and
the remaining weights are renormalised (0.75 / 0.25) rather than treating the
missing evidence as a zero. The same applies when there are no transformed
versions. The reason is recorded in `fusion_detail.fallback_reason`.

Two other modes remain available. `rgb_transform` is the simpler two-term
formula (`0.7 · original + 0.3 · mean(transformed)`), and `frequency` adds a
frequency-model term when one is configured; without it, fusion falls back and
records why.

All weights live under `fusion` in `configs/config.yaml`.

### Confidence

```
score = 0.35 · decisiveness
      + 0.25 · version agreement
      + 0.25 · transformation consistency
      + 0.15 · patch agreement
```

*Decisiveness* is `2 × |p − 0.5|`. *Version agreement* is the share of image
versions landing on the same side of the threshold as the original.
*Consistency* is the transformation-consistency score. *Patch agreement* is the
share of patches agreeing with the whole-image verdict.

Scores map to High (≥0.70), Medium (≥0.45) and Low. An `Uncertain` verdict is
never reported as High confidence.

When patch analysis did not run, its weight is dropped and the rest are
renormalised — a missing signal never counts as disagreement.

### Uncertainty and the three-way decision

The pipeline deliberately supports three outcomes rather than a binary call:

| AI probability | Label |
|---|---|
| below `labels.authentic_max` | Likely authentic |
| in between | **Uncertain — send for human review** |
| at or above `labels.ai_min` | Likely AI-generated |

The defaults (0.40 / 0.60) are **interface defaults, not statistically derived
thresholds**. Replace them with values fitted on labelled validation data:

```bash
./.venv/bin/python scripts/calibrate_threshold.py \
  --data-dir data/sid_set \
  --checkpoint models/pretrained/pytorch_model.pt \
  --target-fpr 0.01
```

That fits Platt calibration on clean validation images, selects a threshold at
a target false-positive rate, and writes both to `outputs/calibration.json`.
Until you do this, the reported probability is an **uncalibrated model score**
and the demo says so.

### Estimated manipulation severity

`estimated_manipulation_severity` reports how much the score moved under the
transformation battery — `low`, `medium` or `high`. It is a statement about
**observable transformation sensitivity** only. It does **not** claim how many
times an image was edited, re-uploaded, or passed through a platform; that
history is not recoverable from a single image.

## Optional frequency / noise analysis

`src/pipeline/frequency.py` extracts FFT radial-energy profiles, DCT energy
ratios, high-pass filter energy and noise residual statistics. It is **disabled
by default** (`frequency.enabled: false`).

It is deliberately conservative: descriptive features are always available, but
`frequency_probability()` returns `None` unless a trained frequency classifier
is configured, and the reason is recorded in the result's `errors`. The module
never invents a score, and importing it never breaks the pipeline — SciPy is
optional and the DCT features are simply skipped without it.

---

## Benchmarking against a labelled dataset

Optional tooling for measuring real accuracy and robustness using
[SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) (public, CC-BY-4.0,
no token required).

```bash
# See what is available (140 GB total, so download selectively)
./.venv/bin/python scripts/download_dataset.py --list

# ~2 GB evaluation sample: 4 validation shards, ~3,500 images
./.venv/bin/python scripts/download_dataset.py --split validation --shards 4

# Benchmark a checkpoint
./.venv/bin/python scripts/evaluate_dataset.py \
    --data-dir data/sid_set \
  --checkpoint models/pretrained/pytorch_model.pt \
    --limit 300 \
    --output outputs/benchmark.json
```

SID_Set labels three classes — `0` real, `1` fully synthetic, `2` tampered —
mapped to the binary target as `0 → real`, `1`/`2` → AI-generated. The report
gives accuracy, AUC, precision/recall/F1, a per-class breakdown, and a
robustness table ranking transformations by how much accuracy they cost.

Evaluation defaults to the **validation** split. Do not evaluate on data a
model was trained on — the resulting numbers are meaningless.

## Probability Calibration and Threshold Selection

Raw sigmoid/softmax scores are often overconfident, so a raw score of 0.5 is
not automatically a reliable decision boundary. This project uses Platt
scaling: a two-parameter logistic mapping is fitted on the model's raw scores
from clean labelled validation images. The fitted parameters are saved and
loaded during inference. Calibration does not use transformed images, test
images, or demo uploads.

Fit once, against a checkpoint you have already downloaded:

```bash
./.venv/bin/python scripts/calibrate_threshold.py \
  --data-dir data/sid_set \
  --checkpoint models/pretrained/pytorch_model.pt \
  --config configs/config.yaml \
  --output outputs/calibration.json \
  --report outputs/calibration_report.json
```

The command accepts only the SID_Set `validation` shard split. It evaluates
thresholds from 0.01 through 0.99 and saves three operating points: a balanced
threshold (Youden's J), the lowest threshold meeting a 1% false-positive-rate
target when possible, and a highest-recall threshold. It then evaluates the
balanced threshold unchanged on clean images and every configured
transformation. If the 1% target is not achievable, the report records that
fact and selects the closest available candidate.

After calibration, set `calibration.enabled: true` in `configs/config.yaml`.
The saved balanced threshold is then loaded automatically and used consistently
for every transformation; no transformation receives its own retuned threshold.
The detailed report includes accuracy, precision, recall, F1, specificity,
false-positive/negative rates, balanced accuracy, Youden's J, ROC-AUC,
calibrated probability statistics, reliability diagrams, ECE, Brier score,
confidence-bin accuracy, ROC/precision-recall curves, and a clean-versus-
transformed chart.

The methodology is motivated by *Your AI-Generated Image Detector Can Secretly
Achieve SOTA Accuracy, If Calibrated*, *Fixed-Threshold Evaluation of a Hybrid
CNN-ViT for AI-Generated Image Detection Across Photos and Art*, and *GenImage:
A Million-Scale Benchmark for Detecting AI-Generated Image*. No threshold is
copied from those papers; all selected values come from this project's own
labelled validation data. Thresholds may need to be refit when the data
distribution or checkpoint changes.

---

## Error handling

| Situation | Behaviour |
|---|---|
| File missing, empty, or not a file | `ImageValidationError` naming the file |
| Unsupported extension | rejected, listing the supported types |
| Corrupted or truncated image | rejected with the decode error |
| Dimensions out of range | rejected, stating the limit |
| Missing/unreadable checkpoint | `ModelSetupError` pointing at `models/README.md` |
| Checkpoint/architecture mismatch | error naming the expected architecture |
| Model at or above 2B parameters | refused at load time |
| A transformation fails | skipped, recorded in `errors`, analysis continues |
| Grad-CAM incompatible | clear message; predictions unaffected |
| Frequency model unavailable | fusion falls back and records the reason |
| One bad file in a batch | recorded and skipped; the run continues |

---

## Testing

```bash
./.venv/bin/python -m pytest tests/ -q
./.venv/bin/python -m ruff check src scripts app.py tests
```

195 tests cover validation (valid, invalid, corrupted, truncated, RGB
conversion), preprocessing (resizing, normalization, determinism), every
transformation, model-loading failures, prediction ranges, score fusion,
confidence calculation, the JSON output formats, batch processing and the
evaluation metrics.

**Mock vs real inference.** The suite runs without a real checkpoint. Tests
that need a model build an **untrained** one on the fly and verify plumbing
only — shapes, ranges, error handling and output contracts. They say nothing
about detection accuracy. Real quality can only be measured with a trained
checkpoint via `scripts/evaluate_dataset.py`. `tests/test_smoke.py` needs
neither a checkpoint nor the torch stack.

---

## Project structure

```
.
├── README.md
├── requirements.txt
├── app.py                          # Streamlit demo
├── configs/config.yaml             # all tunable settings
├── models/README.md                # checkpoint setup guide
├── checkpoints/                    # place best.pt here (git-ignored)
├── outputs/                        # generated JSON and images
├── scripts/
│   ├── run_inference.py            # batch inference CLI
│   ├── calibrate_threshold.py      # validation-only calibration CLI
│   ├── evaluate_dataset.py         # benchmark against labelled data
│   ├── download_dataset.py         # fetch SID_Set shards
│   └── make_dummy_checkpoint.py    # untrained checkpoint for smoke tests
├── src/
│   ├── pipeline/
│   │   ├── validation.py           # stage 2
│   │   ├── preprocessing.py        # stage 3
│   │   ├── transformations.py      # stage 4
│   │   ├── model_loader.py         # stage 5
│   │   ├── prediction.py           # stages 6-7
│   │   ├── consistency.py          # stage 7
│   │   ├── fusion.py               # stage 8
│   │   ├── frequency.py            # stage 9 (optional)
│   │   ├── confidence.py           # stage 10
│   │   ├── explainability.py       # stage 11
│   │   └── pipeline.py             # orchestrator
│   ├── inference/batch_inference.py
│   ├── evaluation/                 # calibration, metrics, SID_Set reader, benchmark
│   ├── utils/{device,config}.py
│   ├── data/                       # legacy albumentations helpers
│   ├── models/classifier.py        # legacy model builder
│   └── explainability/gradcam.py   # legacy standalone Grad-CAM CLI
└── tests/
```

`src/data/`, `src/models/` and `src/explainability/` predate the
`src/pipeline/` stack and are preserved for reference. They need
`albumentations` and `opencv-python-headless` (commented out in
`requirements.txt`); the pipeline itself does not.

---

## Scope and limitations

Please read this section before quoting any number from this system.

- **Patch analysis localises attention, not editing.** A hot patch shows where
  the detector responded strongly. It does **not** prove that region was
  edited, and the system does not segment or verify edits.
- **Processing history is not recoverable.** `estimated_manipulation_severity`
  describes how much the score moved under our transformation battery. The
  system cannot reconstruct how many platforms, uploads, editing tools or
  compression passes an image actually went through, and does not claim to.
- **Probabilities are uncalibrated by default.** Until
  `scripts/calibrate_threshold.py` has been run on labelled validation data,
  the reported number is a raw model score, and the 0.40 / 0.60 label bands are
  interface defaults rather than thresholds derived from data.
- **The WildFake demonstration subset must not be used** for training, model
  selection, or threshold selection. It is reserved for evaluation.
- **Image-level detection only.** The pipeline returns one probability per
  image. It does not localise manipulated regions, and it does not process
  video or audio.
- **Predictions are confidence estimates, not certainty.** Output is worded as
  "Likely AI-generated" or "Likely authentic", never as proof. A high score is
  evidence, not a verdict, and should not be the sole basis for a consequential
  decision about a person or a piece of content.
- **Hackathon-scale proof of concept.** Not a production moderation system, and
  deliberately not a platform-wide one.
- **The model must have fewer than 2 billion parameters**, enforced at load
  time. The supported backbones are 5–29 M parameters.
- **Accuracy depends entirely on the supplied checkpoint.** This repository
  contributes robustness measurement and a reproducible pipeline, not a model.
- **The WildFake validation subset must not be used for training.** It is
  reserved for evaluation; training on it invalidates the results.
- **Consistency is not correctness.** A model can be consistently wrong. A high
  consistency score means the prediction was stable under transformation, not
  that it was right.
- **Grad-CAM shows where the model looked**, which is not the same as showing
  where an image was manipulated.
- **Transformations approximate real-world degradation.** They simulate
  compression, redistribution and editing, but no fixed list covers every
  pipeline an image may pass through.

---

## Future improvements

- Train a frequency-domain classifier so the optional fusion mode carries a
  genuine calibrated signal instead of falling back.
- Test-time augmentation ensembling with learned rather than fixed weights.
- Localisation of manipulated regions, using the segmentation masks SID_Set
  already ships.
- Adversarial and unseen-generator robustness testing.
- Model ensembling across backbones, still within the parameter budget.
- ONNX export and quantisation for faster batch throughput.

---

## Team contributions

| Member | Role | Contribution |
|---|---|---|
| _(name)_ | Pipeline engineering | _(validation, preprocessing, transformations)_ |
| _(name)_ | Modelling | _(architecture selection, training, checkpointing)_ |
| _(name)_ | Evaluation | _(benchmarking, robustness analysis, metrics)_ |
| _(name)_ | Demo & UX | _(Streamlit app, explainability, charts)_ |
| _(name)_ | Documentation | _(README, configuration, presentation)_ |

---

## Acknowledgements

- [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) (CC-BY-4.0) from
  the SIDA project, which incorporates material from COCO, OpenImages V7 and
  Flickr30k.
- torchvision for the pretrained backbone architectures.
