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

### One command

```bash
python3 scripts/setup.py --all
```

That creates the virtualenv, installs `requirements.txt`, downloads the 2.11 GB
model checkpoint (resuming if interrupted) and fetches a ~2 GB evaluation
sample. It prompts before each large download; add `--yes` to skip prompts.
Everything is idempotent, so re-running it is safe.

Just the code and the model, no dataset:

```bash
python3 scripts/setup.py
```

### Check what you have

```bash
python3 scripts/setup.py --check
```

```
Project status
------------------------------------------------------------------
  OK       Virtualenv           Python 3.14.2
  OK       Dependencies         all present
  OK       Model checkpoint     2.0 GB, size verified
  OK       Sample image         bundled
  optional Evaluation dataset   not downloaded
  optional Calibration          not fitted
------------------------------------------------------------------
```

Anything missing prints the exact command that fixes it. Exit code is `0` when
everything required is present, `1` otherwise, so it works in CI.

### Run it

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir data/cifake_sample \
  --checkpoint models/pretrained/pytorch_model.pt \
  --no-transformations \
  --output outputs/predictions.json

cat outputs/predictions.json
```

`pred` is the AI-generated probability:

```json
[
  {
    "image_path": "/abs/path/to/data/cifake_sample/0000.jpg",
    "pred": 0.009768
  }
]
```

Paths are absolute by default; add `--relative-paths` for paths relative to
`--input-dir`. No config edit is needed — the loader recognises the checkpoint
from its tensor signature and selects `bombek_siglip2_dinov2` automatically.

### Web demo

```bash
./.venv/bin/streamlit run app.py
```

It defaults to the checkpoint `setup.py` downloads, so it works immediately.

### On Apple Silicon

Add `--device mps` to any command — it is about **8.9× faster** than CPU and
was verified to produce matching probabilities.

---

## Manual setup

If you would rather not use `setup.py`, or it fails on your platform:

### 1. Dependencies

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

Verified from scratch: a clean clone plus a clean virtualenv installed only
from `requirements.txt` runs the full 375-test suite green.

### 2. Model checkpoint (2.11 GB)

```bash
mkdir -p models/pretrained
curl -L --fail --progress-bar -C - \
  https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2/resolve/main/pytorch_model.pt \
  -o models/pretrained/pytorch_model.pt
```

The size must be exactly `2105483083` bytes:

```bash
ls -l models/pretrained/pytorch_model.pt
```

### 3. Evaluation dataset (optional, ~2 GB)

Only needed to reproduce the evaluation numbers; inference works without it.

```bash
./.venv/bin/python scripts/download_dataset.py --split validation --shards 4 --yes
./.venv/bin/python scripts/extract_dataset.py          # unpack to image files
```

### 4. Calibration (optional)

Needs the dataset. Without it, scores are uncalibrated and the demo says so.

```bash
./.venv/bin/python scripts/evaluate_confidence.py \
  --save-calibration outputs/calibration.json
```

---

## What is and is not in the repository

| | In git? | How to get it |
|---|---|---|
| All source, tests, configs | yes | `git clone` |
| Sample image (`data/cifake_sample/`) | yes | `git clone` |
| Evaluation results (`outputs/protocol/`, `outputs/patch_ablation/`) | yes | `git clone` |
| **Model checkpoint** (2.11 GB) | **no** | `scripts/setup.py --checkpoint` |
| **Dataset** (~2 GB) | **no** | `scripts/setup.py --dataset` |
| **Calibration parameters** | **no** | `scripts/evaluate_confidence.py` |

The three large or machine-specific items are git-ignored deliberately: GitHub
rejects files over 100 MB, and git keeps blobs forever once added. Every one of
them has a one-command fetch, and the pipeline degrades with a clear message
rather than crashing when any is absent.

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

The detector is 740 M parameters. **Use `--device mps` on Apple Silicon** — it is
about 8.9× faster than CPU and was verified to produce matching probabilities
(0.0108 vs 0.0110 on the same image, no NaNs).

| Operation | CPU | MPS |
|---|---|---|
| One-time model load | ~14 s | ~15 s |
| One forward pass | ~6.7 s | ~0.75 s |
| Per image, `--no-transformations` | ~7 s | ~1 s |
| Per image, full 15-version sweep | ~110 s | ~12 s |
| Per image, 15 versions + 12 patches | ~195 s | ~20 s |

Measured on this machine with the Bombek1 checkpoint; your numbers will differ.
`--device cuda` is wired up but has not been tested here. If a device errors or
returns NaNs, fall back to `--device cpu`.

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

The default is `models/pretrained/pytorch_model.pt`, where `scripts/setup.py`
puts it. Override with `--checkpoint` or `paths.checkpoint_dir`.
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

The sidebar defaults to `models/pretrained/pytorch_model.pt`, so the demo works
with no manual step after `scripts/setup.py`. Point it elsewhere by editing the
**Model checkpoint** field.

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

### Modes

Patch cost is one forward pass per patch, so the mode decides the bill:

| `patches.mode` | Passes / image | What it does |
|---|---:|---|
| `off` | 0 | no patch scoring |
| `coarse` | ~4 | one small grid — cheapest useful map |
| `full` | ~12 | one dense grid, capped by `max_patches` |
| `top_k` | ~12 | coarse grid, then refine only the top-k regions |
| **`uncertain_only`** (default) | **0 or ~12** | runs `base_mode` **only** when the whole-image score falls inside `uncertain_band`; confident images stop early and spend nothing |

`uncertain_only` is the default because the detector is confident on most
images, and patch scoring on an already-decided image buys nothing.

### Cost controls

- **Early stopping** — `uncertain_only` skips patches entirely outside the
  uncertain band.
- **Whole-image reuse** — a patch box covering the whole image reuses the
  whole-image score instead of recomputing it; reuses are counted in
  `patch_analysis.reused_scores`.
- **Caching** — a box requested twice (as `top_k` refinement can) is served
  from cache.
- **Batched inference** under `torch.inference_mode()`, with the model loaded
  once and reused.
- **`--device mps`** is ~8.9× faster than CPU.
- `patches.max_patches` caps the grid; larger grids are evenly subsampled.
- Batch inference has patches **off by default**; opt in with `--patches`.

Every run reports its own cost: `forward_passes`, `reused_scores`, `seconds`,
`peak_memory_mb` and heatmap `coverage`.

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
| `mode` | `uncertain_only` | `off`, `coarse`, `full`, `top_k`, `uncertain_only` |
| `base_mode` | `full` | what `uncertain_only` runs when it runs |
| `uncertain_band` | `[0.2, 0.8]` | whole-image scores that trigger patch analysis |
| `coarse_max_patches` | `4` | cap for `coarse` mode |
| `refine_factor` | `2` | `top_k` splits each chosen patch into 2×2 |
| `batch_size` | `null` | falls back to `inference.batch_size` |
| `device` | `null` | falls back to the model's device |
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

### Ablation: does patch analysis earn its cost?

```bash
./.venv/bin/python scripts/ablate_patches.py \
  --checkpoint models/pretrained/pytorch_model.pt \
  --limit 60 --per-class-limit 20 --device mps \
  --output-dir outputs/patch_ablation
```

Runs every mode against the same whole-image score at one frozen threshold and
records, per mode: accuracy, balanced accuracy, F1, recall, FPR, AUROC, mean
forward passes, seconds per image, heatmap coverage, peak memory, and **how
many final decisions actually changed**.

It ends with a verdict computed from the numbers: a mode keeps scoring status
only if it improves F1 or recall **without** raising the false-positive rate.
Otherwise the report says to demote patch analysis to an explainability
feature. Outputs land in `outputs/patch_ablation/` as `ablation.json`,
`modes.csv` and `per_image.csv`.

### Ablation result: patch analysis is explainability, not scoring

Measured on **60 SID_Set images**, threshold 0.42 frozen across all modes
(`outputs/patch_ablation/`):

| Mode | Accuracy | F1 | Recall | FPR | AUROC | passes/img | s/img | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **whole-image only** | **0.750** | **0.776** | **0.650** | 0.050 | **0.844** | 0 | 0.56 | — |
| coarse | 0.733 | 0.758 | 0.625 | 0.050 | 0.838 | 4.0 | 2.49 | 27.8 % |
| full | 0.733 | 0.758 | 0.625 | 0.050 | 0.811 | 12.0 | 6.26 | 70.2 % |
| top_k | 0.733 | 0.758 | 0.625 | 0.050 | 0.818 | 12.0 | 6.41 | 27.8 % |
| uncertain_only | 0.733 | 0.758 | 0.625 | 0.050 | 0.844 | **1.6** | **1.33** | 67.9 % |

**Every mode scored worse than whole-image alone** — F1 −0.019, recall −0.025,
accuracy −0.017 — with **no** reduction in false positives. Only one decision
out of 60 changed, and it changed for the worse.

**Patch evidence has therefore been removed from scoring.** `fusion.mode` is
now `rgb_transform`; patch analysis stays enabled purely for the heatmap. This
is not a preference — the ablation applies a rule fixed before the run (a mode
keeps scoring status only if it improves F1 or recall without raising FPR), and
an earlier independent 120-image protocol run reached the same conclusion.
Re-run `scripts/ablate_patches.py` after any checkpoint change to re-test;
`fusion.whole_patch` weights remain configured for that.

Why it fails here, from the cached scores: on authentic images patch evidence
averages **0.297** against a whole-image score of **0.081** (the
max-of-many-noisy-patches inflation), while on tampered images it barely
exceeds the whole-image score and correlates with it at **r = 0.83** — largely
redundant. The likely real fix is a checkpoint trained with locally edited
examples, not a better patch aggregation.

**Cost.** `uncertain_only` is the cheapest useful mode at **1.6 passes and
1.33 s per image — 4.7× cheaper than `full`** — because it skips patches
entirely on confidently-scored images, and it is the only mode that leaves
AUROC unchanged. It is the default.

One caveat: this ablation measured patch evidence's effect on the **probability**
only. Patch agreement still contributes 0.15 to the *confidence* score; that
contribution has not been separately measured.

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

### What a highlighted region is, and is not

A warm patch is a **suspicious region that influenced the model's score**.

It is **not**:

- proof that the region was AI-edited, or edited at all;
- a segmentation of a manipulated area;
- a reconstruction of the image's editing or redistribution history — that
  history is not recoverable from a single image, and this system does not
  attempt to infer it.

Treat highlighted regions as **a pointer for human review**, and read them
alongside the whole-image score rather than instead of it.

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

Default mode, `rgb_transform`:

```
final = 0.7 · whole_image + 0.3 · mean(transformed versions)
```

Patch evidence carries **no** fusion weight: an ablation found it degraded F1
and recall (see *Ablation result* under Patch-level analysis). The three-term
mode `whole_patch_transform` remains available for re-testing:

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

## Track 5 evaluation protocol

A complete robustness evaluation: every official transformation, a four-way
system ablation, failure analysis and runtime — all at **one fixed threshold**.

```bash
./.venv/bin/python scripts/evaluate_protocol.py \
  --checkpoint models/pretrained/pytorch_model.pt \
  --data-dir data/sid_set \
  --limit 120 \
  --per-class-limit 40 \
  --device mps \
  --output-dir outputs/protocol
```

Re-analyse a finished run in seconds without re-scoring anything:

```bash
./.venv/bin/python scripts/evaluate_protocol.py --reuse-scores
```

### How it works

One scoring pass per image records the clean score, all 14 transformed-version
scores and the patch evidence, then caches them to `outputs/protocol/scores.json`.
Every analysis is derived from that cache, so the four systems are compared on
**identical forward passes** rather than separate runs, and re-analysis needs no
GPU time.

### Threshold discipline

The decision threshold is selected **once**, on the **clean scores of a
validation split only**, then frozen and applied unchanged to every
transformation and every system variant. It is never retuned per condition —
doing so would report the best case for each transformation rather than the
performance an operator would actually get.

The validation and test splits are disjoint, assigned deterministically by
hashing the image id, and **stratified by class** so both splits always contain
authentic and AI-generated images. Change the sizes with
`--validation-fraction` and `--split-seed`, and the operating point with
`--target-fpr`.

### Conditions evaluated

Clean, plus every transformation the challenge specifies: JPEG quality 90/70/50/30,
Gaussian blur σ 0.5/1.0/2.0, resize 0.5×/0.25× with upscale, Gaussian noise
σ 0.02/0.05/0.10, colour jitter ±20 %, and centre crop retaining 80 %.

### Metrics reported

Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, FPR, FNR and the
confusion matrix for every condition, plus:

```
robustness_drop  = clean_score - transformed_score
robustness_ratio = transformed_score / clean_score
worst_case       = min(score) across all transformations
```

Average-transformed and worst-case figures are reported separately, because an
average hides a single catastrophic transformation.

### System ablation

Four systems on the same images and the same frozen threshold:

| Variant | Formula |
|---|---|
| Whole-image only | clean score |
| Whole + transformations | `0.7 · clean + 0.3 · mean(transformed)` |
| Whole + patch analysis | `0.75 · clean + 0.25 · patch evidence` |
| Complete fused system | `0.6 · clean + 0.2 · transformed + 0.2 · patch` |

### Outputs

```
outputs/protocol/
├── metrics.json          every number, plus the config snapshot and claims
├── scores.json           the raw per-image cache (re-analysable)
├── tables/
│   ├── per_transformation.csv
│   ├── system_ablation.csv
│   ├── robustness_summary.csv
│   └── generation_families.csv
├── charts/
│   ├── robustness_accuracy.png       clean vs each transformation
│   ├── severity_accuracy.png         performance against severity
│   ├── confusion_matrices.png        one per system variant
│   ├── confidence_distributions.png  authentic vs AI, threshold marked
│   ├── variant_comparison.png        the four-way ablation
│   └── generation_families.png       per-family performance
└── examples/
    ├── false_positives.png           authentic images flagged as AI
    └── false_negatives.png           AI images reported as authentic
```

### What these results may and may not claim

**Dataset-source holdout — genuine.** The checkpoint records training on
OpenFake; evaluation runs on SID_Set, a different dataset with different
collection procedures. Cross-dataset generalisation may be claimed at the
sample size actually evaluated.

**Unseen generator — NOT established.** SID_Set does not publish per-generator
labels. The protocol reports `full_synthetic` and `tampered` as *generation-process
families*, which is a **proxy** showing whether performance transfers across
generation processes within one dataset. It is **not** a generator holdout, and
no unseen-generator claim is made. A genuine claim needs a dataset with
per-generator labels where the evaluated generator was excluded from model
development.

**Sample size.** Figures come from the number of images actually scored, which
is a hackathon-scale sample rather than the full split. Every table records its
own `count`; treat wide confidence intervals accordingly.

**The WildFake demonstration subset is not used** for training, model selection
or threshold selection.

---

## Confidence, calibration and thresholds — measured

All from cached scores in `outputs/protocol/scores.json`; reproduce with:

```bash
./.venv/bin/python scripts/evaluate_confidence.py --method platt --target-fpr 0.01 \
  --save-calibration outputs/calibration.json
```

120 SID_Set images, **48 validation / 72 test, stratified and disjoint**.
Calibration and thresholds are fitted on the **clean validation scores only**.

### Patch agreement removed from confidence

Variants compared at a fixed threshold, with the term weighted 0.15:

| Variant | AUROC(confidence vs correctness) | Correct−incorrect gap | Confidence ECE |
|---|---:|---:|---:|
| **A. no patch agreement** | **0.7684** | **0.1450** | 0.1307 |
| B. patch agreement always | 0.7415 | 0.1177 | 0.1348 |
| C. patch agreement when reliable | 0.7474 | 0.1209 | 0.1256 |

Accuracy, F1 and FPR were **identical** across all three (0.792 / 0.819 / 0.042),
which confirms the term never touched the predicted probability.

Including patch agreement made confidence **worse** at separating correct from
incorrect predictions. Gating it to "reliable" patches did not rescue it. Under
the rule fixed before the run — retain only if AUROC improves by >0.01 without
shrinking the gap — **patch agreement is removed**:
`confidence.patch_agreement_weight: 0.0`. Confidence is now
`0.4·decisiveness + 0.3·version agreement + 0.3·consistency`. Patch agreement is
still computed and shown in the UI; it simply carries no weight.

### Calibration

Platt scaling on clean validation scores:

| | ECE | Brier |
|---|---:|---:|
| Uncalibrated | 0.2602 | 0.2210 |
| **Platt** | **0.1327** | **0.1540** |
| Temperature | 0.1653 | 0.1799 |

Calibration is monotonic, so **AUROC is unchanged** (0.8568) — it fixes
probability values, not ranking.

### Threshold selection, and why the F1-optimal one was rejected

All 99 thresholds 0.01–0.99 evaluated on calibrated clean validation scores.
Measured on the **held-out test split**:

| Operating point | Threshold | Accuracy | F1 | FPR | Worst-case |
|---|---:|---:|---:|---:|---:|
| **balanced** (Youden J) | **0.69** | **0.819** | **0.847** | **0.042** | **0.778** |
| low-FPR | 0.78 | 0.778 | 0.805 | 0.042 | 0.736 |
| F1-optimal | 0.53 | 0.750 | 0.816 | 0.417 | 0.681 |
| high-recall | 0.01 | — | — | 1.000 | — |

The F1-optimal threshold maximised F1 on the 48-image validation split and then
**transferred badly** — FPR 0.417 on test. That is what a small validation set
does to threshold selection, and it is why `balanced` is the shipped default.

The FPR ≤ 1 % target was met on validation (FPR 0.000 at t=0.78) but **not on
test** (0.042). Reported as achieved-in-validation-only, not as a guarantee.

### Fixed-threshold robustness (calibrated, threshold 0.69)

| | Accuracy | F1 | AUROC | FPR |
|---|---:|---:|---:|---:|
| Clean | 0.819 | 0.847 | 0.857 | 0.042 |
| Average transformed | 0.809 | | | |
| Worst case (JPEG q50) | 0.778 | | | |

Largest drop **0.042**; the threshold is never retuned for any transformation.

### System comparison

| System | Accuracy | F1 | AUROC | FPR |
|---|---:|---:|---:|---:|
| 1. Whole-image RGB | 0.819 | 0.847 | 0.8568 | 0.042 |
| 2. Whole + transformations | 0.819 | 0.847 | 0.8576 | 0.042 |
| 3. Whole + transformations, calibrated | 0.819 | 0.847 | 0.8568 | 0.042 |
| 4. Final system (patch excluded from scoring) | 0.819 | 0.847 | 0.8576 | 0.042 |

Systems 2 and 4 are numerically identical because the final system *is*
`rgb_transform` with patch analysis excluded from scoring — stated rather than
presented as a separate result. Calibration changes probability values, not the
ranking or the decisions at a matched operating point. `rgb_transform` is
retained: nothing measurably beat it.

### Two bugs found and fixed during this work

- **Platt fitting diverged** on this data (slope ≈ 1.6 × 10⁸), collapsing every
  calibrated score to exactly 0.0 or 1.0. That made all four operating points
  identical and Brier *worse* than uncalibrated. Fixed with a backtracking line
  search plus Platt target smoothing; both are covered by tests.
- **A missing calibration file killed the pipeline.** `outputs/` is git-ignored,
  so a fresh clone with calibration enabled could not run at all. It now falls
  back to uncalibrated scores with an explicit notice.

---

## Measured results

From `outputs/protocol/metrics.json`. **120 SID_Set validation images** (48 for
threshold selection, **72 held-out test**), Bombek1 checkpoint, threshold
**0.42** fixed on clean validation data and never retuned. Reproduce with
`scripts/evaluate_protocol.py --reuse-scores`.

This is a hackathon-scale sample. Treat these as indicative, not definitive.

### Robustness: strong

| | Accuracy | Balanced acc | F1 | AUROC | FPR |
|---|---:|---:|---:|---:|---:|
| Clean | 0.806 | 0.844 | 0.833 | 0.857 | 0.042 |
| Average transformed | 0.800 | — | — | — | — |
| **Worst case** (JPEG q50) | **0.764** | 0.802 | 0.795 | 0.837 | 0.083 |

Largest drop **0.042**; robustness ratio ≥ **0.948** on every transformation.
Four transformations slightly *improved* accuracy (centre crop 0.847, and
JPEG q90 / blur σ1 / resize 0.5× at 0.819). No transformation caused a
collapse — the headline robustness claim is supported.

### The real weakness: locally tampered images

| Family | n | Accuracy | Recall | AUROC | mean p(AI) |
|---|---:|---:|---:|---:|---:|
| Fully synthetic | 26 | 0.980 | 1.000 | 1.000 | 0.998 |
| **Tampered** | 22 | **0.696** | **0.409** | **0.678** | 0.389 |

**13 of 14 errors were false negatives, and all the worst were tampered
images.** The detector is near-perfect on fully synthetic images and weak on
locally edited ones — unsurprising, since the checkpoint was trained on
OpenFake, a fully-synthetic corpus.

### Patch analysis did not fix it — negative result

All four systems scored **identically** (accuracy 0.806, F1 0.833, FPR 0.042);
only AUROC moved, and the fused system was slightly **worse**:

| System | Accuracy | F1 | AUROC |
|---|---:|---:|---:|
| Whole-image only | 0.806 | 0.833 | **0.857** |
| Whole + transformations | 0.806 | 0.833 | 0.863 |
| Whole + patches | 0.806 | 0.833 | 0.853 |
| Complete fused | 0.806 | 0.833 | 0.852 |

Diagnosis from the cached scores:

- On **authentic** images patch evidence averages **0.297** against a clean
  score of **0.081** — the max-of-many-noisy-patches inflation, which pushes
  authentic images *toward* the AI side.
- On **tampered** images patch evidence (0.417) barely exceeds the clean score
  (0.383), and the two correlate at **r = 0.83** — it is largely redundant
  rather than new information.
- Of the 13 tampered images the whole-image model missed, patch evidence
  rescued only **2**.

The gain on tampered images and the inflation on authentic ones roughly cancel.
**On this evidence the patch term does not earn its 0.2 weight**, and the
default `fusion.mode: whole_patch_transform` is not justified by these numbers.
It has deliberately **not** been changed — retuning fusion weights on 72 test
images would be fitting to the sample. Re-measure at larger n before deciding;
a detector trained with locally-edited examples is the more likely real fix.

### Runtime

13.34 s per image on MPS for 27 forward passes (1 clean + 14 transformed + 12
patches). Roughly 1 s per forward pass end to end.

### What these numbers do not establish

Cross-**generator** generalisation. SID_Set publishes no per-generator labels,
so the family split above is a proxy. The **dataset-source holdout is genuine**
(trained on OpenFake, evaluated on SID_Set).

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
not automatically a reliable decision boundary.

Two calibrators are available, both fitted on the model's raw scores from
**clean labelled validation images only**:

| `--method` | Parameters | Effect |
|---|---|---|
| `platt` (default) | slope + intercept | Full logistic remap; can shift the decision boundary |
| `temperature` | slope only (intercept pinned at 0) | Sharpens or softens confidence; **cannot reorder scores or move the boundary on its own** |

Temperature scaling is the standard single-parameter choice when a model ranks
well but is over-confident. Its fitted temperature `T` (where slope = 1/T) is
recorded in the saved parameters. The fitted parameters are saved and loaded
during inference. Calibration never uses transformed images, test images, or
demo uploads.

Fit once, against a checkpoint you have already downloaded:

```bash
./.venv/bin/python scripts/calibrate_threshold.py \
  --data-dir data/sid_set \
  --checkpoint models/pretrained/pytorch_model.pt \
  --config configs/config.yaml \
  --output outputs/calibration.json \
  --report outputs/calibration_report.json
```

Add `--method temperature` to fit the single-parameter calibrator instead.

The command accepts only the SID_Set `validation` shard split. It evaluates
all 99 thresholds from 0.01 through 0.99, computing accuracy, precision,
recall, F1, balanced accuracy, FPR, FNR and Youden's J at each, and saves
**four operating points**:

| Operating point | Selection rule | Use when |
|---|---|---|
| `balanced` | maximises Youden's J | general use; the default |
| `f1_optimal` | maximises F1 | precision and recall matter equally |
| `low_false_positive` | lowest threshold with FPR ≤ target (default 1 %) | false accusations are costly |
| `high_recall` | maximises recall | missing an AI image is costly |

It then evaluates the chosen threshold **unchanged** on clean images and every
configured transformation. If the FPR target is not achievable the report
records that fact and selects the closest available candidate.

After calibration, set `calibration.enabled: true` in `configs/config.yaml` and
pick the operating point:

```yaml
calibration:
  enabled: true
  path: outputs/calibration.json
  method: platt                # platt | temperature
  operating_point: balanced    # balanced | f1_optimal | low_false_positive | high_recall
  target_false_positive_rate: 0.01
  uncertainty_margin: 0.10     # sets the Uncertain band around the threshold
```

The saved threshold is then loaded automatically and used consistently for
every transformation; **no transformation receives its own retuned threshold.**
The detailed report includes accuracy, precision, recall, F1, specificity,
false-positive/negative rates, balanced accuracy, Youden's J, ROC-AUC,
calibrated probability statistics, reliability diagrams, ECE, Brier score,
confidence-bin accuracy, ROC/precision-recall curves, and a clean-versus-
transformed chart.

### Four states the demo keeps distinct

Presenting a raw score beside a default threshold as though both came from data
would overstate what the system knows, so the demo labels each explicitly:

| State | Shown as |
|---|---|
| Calibrated probability | green — names the method and fitted temperature |
| Uncalibrated model score | amber — "ranks images but is not a calibrated probability" |
| Data-derived threshold | green — names the operating point it came from |
| Interface default threshold | amber — "not derived from labelled validation data" |

`DetectionPipeline.calibration_status()` returns this programmatically.

### When labelled validation data is unavailable

No calibration metrics are invented. Without labelled clean validation images
`scripts/calibrate_threshold.py` cannot run, the pipeline keeps the 0.5
interface default with the 0.40 / 0.60 label bands, and both the demo and
`calibration_status()` report them as interface defaults. The missing
requirement is **labelled clean validation images with both authentic and
AI-generated examples**; threshold selection raises rather than proceeding if
the validation split contains only one class.

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
