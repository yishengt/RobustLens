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
                    │    (stage 2)    │  dimensions · opens · RGB-convertible
                    └────────┬────────┘
                             │  full-resolution RGB image + metadata
                    ┌────────▼─────────────┐
                    │  Transformation      │  JPEG ×4 · blur ×3 · resize ×2
                    │  generation (st. 4)  │  noise ×3 · jitter · crop
                    └────────┬─────────────┘
                             │  15 image versions (original + 14)
                    ┌────────▼────────┐
                    │  Preprocessing  │  RGB · 224×224 · normalize · tensor
                    │    (stage 3)    │  identical for every version
                    └────────┬────────┘
                             │
                    ┌────────▼─────────────┐
                    │  Feature extraction  │  lightweight CNN backbone
                    │  + classification    │  → p(AI) per version
                    │  (stages 5-6)        │
                    └────────┬─────────────┘
                             │  15 predictions
         ┌───────────────────┼───────────────────┐
         │                   │                   │
┌────────▼────────┐ ┌────────▼────────┐ ┌────────▼────────┐
│  Consistency    │ │  Optional freq. │ │  Explainability │
│  (stage 7)      │ │  analysis (9)   │ │  (stage 11)     │
│ mean·min·max    │ │ FFT·DCT·hipass  │ │ Grad-CAM heatmap│
│ std·range·score │ │ noise residual  │ │ + charts        │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └─────────┬─────────┘                   │
                   │                             │
          ┌────────▼────────┐                    │
          │  Fusion (st. 8) │  0.7·original + 0.3·mean(transformed)
          └────────┬────────┘                    │
                   │                             │
          ┌────────▼────────┐                    │
          │ Confidence (10) │  decisiveness · agreement · consistency
          └────────┬────────┘                    │
                   └──────────┬──────────────────┘
                              │
                       ┌──────▼──────┐
                       │ Final output│  label · p(AI) · p(real)
                       │             │  confidence · consistency
                       └─────────────┘
```

Every transformation is applied to the **original full-resolution image**, then
put through exactly the same preprocessing as the original. This mirrors how
images degrade in the wild, rather than degrading an already-downsampled thumbnail.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
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

To test the plumbing before a real checkpoint exists:

```bash
python scripts/make_dummy_checkpoint.py --output checkpoints/dummy.pt
```

This writes **untrained** weights. It proves the wiring works; its predictions
are meaningless. It defaults to `resnet18` because a randomly initialised
EfficientNet-B0 is numerically dead in eval mode — its feature map collapses to
a standard deviation of ~1e-14 and every image returns an identical score,
which would mask real bugs.

---

## Streamlit demo

```bash
streamlit run app.py
```

Upload one JPG/JPEG/PNG/WEBP image to see:

- the original image and its metadata
- the final classification and the confidence statement
- AI-generated and real-image probabilities
- the confidence level and the transformation-consistency score
- a per-version prediction table and bar chart
- a drift chart showing which transformations moved the score most
- a Grad-CAM heatmap (or a clear message if unavailable)
- the full JSON result

---

## Batch inference

```bash
python scripts/run_inference.py \
    --input-dir path/to/images \
    --checkpoint checkpoints/best.pt \
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

The detailed format (`--detailed-output`) adds:

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

### Label bands

| AI probability | Label |
|---|---|
| 0.00 – 0.39 | Likely authentic |
| 0.40 – 0.59 | Uncertain |
| 0.60 – 1.00 | Likely AI-generated |

### Fusion

Default:

```
final = 0.7 × original + 0.3 × mean(transformed)
```

Optional frequency mode (`fusion.mode: frequency`):

```
final = 0.5 × rgb + 0.3 × frequency + 0.2 × transformation_consistency
```

Frequency mode never activates on its own. Without a trained frequency
classifier it falls back to the default formula and records the reason in
`fusion_detail.fallback_reason`. Note that the consistency term in this formula
is a stability measure, not a probability of being AI-generated; it is included
because the task specifies these weights, but treat it as a tie-breaker rather
than evidence.

### Confidence

```
score = 0.4 × decisiveness + 0.3 × agreement + 0.3 × consistency
```

where *decisiveness* is `2 × |p − 0.5|`, *agreement* is the share of versions
landing on the same side of the threshold as the original, and *consistency* is
the transformation-consistency score. Scores map to High (≥0.70),
Medium (≥0.45) and Low. An `Uncertain` verdict is never reported as High.

---

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
python scripts/download_dataset.py --list

# ~2 GB evaluation sample: 4 validation shards, ~3,500 images
python scripts/download_dataset.py --split validation --shards 4

# Benchmark a checkpoint
python scripts/evaluate_dataset.py \
    --data-dir data/sid_set \
    --checkpoint checkpoints/best.pt \
    --limit 300 \
    --output outputs/benchmark.json
```

SID_Set labels three classes — `0` real, `1` fully synthetic, `2` tampered —
mapped to the binary target as `0 → real`, `1`/`2` → AI-generated. The report
gives accuracy, AUC, precision/recall/F1, a per-class breakdown, and a
robustness table ranking transformations by how much accuracy they cost.

Evaluation defaults to the **validation** split. Do not evaluate on data a
model was trained on — the resulting numbers are meaningless.

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
python -m pytest tests/ -q          # or: python -m unittest discover -s tests
```

188 tests cover validation (valid, invalid, corrupted, truncated, RGB
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
│   ├── evaluation/                 # metrics, SID_Set reader, benchmark
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
- Calibrate probabilities (temperature scaling or isotonic regression) against
  a held-out set, so the label bands mean the same thing across checkpoints.
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
