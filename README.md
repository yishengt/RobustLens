# Robust Detection of AI-Generated Images

This repository is a lightweight, modular foundation for classifying images as
real (`0`) or AI-generated (`1`). The training pipeline includes configurable
real-world transformations so the model can be evaluated on both clean and
degraded images.

No model training is performed as part of this scaffold.

## Project layout

```text
.
├── README.md
├── requirements.txt
├── configs/config.yaml
├── data/
│   ├── raw/{real,ai_generated}/
│   ├── processed/
│   └── splits/
├── src/
│   ├── data/{dataset.py,augmentations.py}
│   ├── features/engineering.py
│   ├── models/classifier.py
│   ├── training/train.py
│   ├── evaluation/{evaluate.py,robustness_test.py,error_analysis.py,materialize_transforms.py}
│   ├── explainability/gradcam.py
│   ├── inference/predict.py
│   └── utils/{config.py,checkpoint.py,seed.py}
├── scripts/run_inference.py
├── app.py
└── tests/test_smoke.py
```

## Dataset placement

Place approved datasets under the two class directories below. The loader
searches recursively and accepts common image extensions.

```text
data/raw/
├── real/
│   ├── SID_Set/
│   ├── CIFAKE_real/
│   ├── WildFake_real/
│   └── other_approved_real/
└── ai_generated/
    ├── SID_Set_generated/
    ├── CIFAKE_fake/
    ├── WildFake_generated/
    └── other_approved_generated/
```

Use the exact real/generated labels supplied by each dataset. For datasets
whose folders do not naturally match this layout, copy or symlink the images
into `real/` and `ai_generated/`, or update `configs/config.yaml` to point at
your own two class roots. Keep dataset licenses, attribution, and any
train/test restrictions with the downloaded data. In particular, verify the
terms for SID Set, CIFAKE, WildFake, and any other approved dataset before
redistributing or publishing results.

The split file is created at `data/splits/split.json` on the first training or
evaluation run. It stores image paths and labels so later runs use the same
train/validation split. Delete it or pass `--force-resplit` to the training
command to create a new split.

## Installation

Use Python 3.10+ in a virtual environment and install the capability set:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses torchvision's EfficientNet-B0 by default. It has one binary
output logit and is far below the 2B-parameter limit. Set
`model.pretrained: true` if pretrained torchvision weights are available; the
first such run may download weights.

## Configuration

Edit `configs/config.yaml` for data paths, image size, batch size, optimizer,
device, checkpoint locations, and augmentation probabilities. The robustness
suite evaluates these exact cases by default:

- JPEG quality 90, 70, 50, and 30
- Gaussian blur sigma 0.5, 1.0, and 2.0
- Downscale to 0.5x and 0.25x, then upscale to the model input size
- Gaussian noise sigma 0.02, 0.05, and 0.10 (fractions of the 0-1 image range)
- Brightness, contrast, and saturation jitter up to +/-20%
- Centre crop of 80%, then resize to the model input size

## Commands

Run the dependency-free smoke checks and source compilation:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts app.py
```

Train and save `checkpoints/best.pt` plus `checkpoints/last.pt`:

```bash
python -m src.training.train --config configs/config.yaml
```

Evaluate the clean validation set and write metrics JSON:

```bash
python -m src.evaluation.evaluate \
  --config configs/config.yaml \
  --checkpoint checkpoints/best.pt \
  --output outputs/metrics.json
```

Run clean-versus-transformation robustness evaluation:

```bash
python -m src.evaluation.robustness_test \
  --config configs/config.yaml \
  --checkpoint checkpoints/best.pt \
  --output outputs/robustness.json
```

Create your own viewable transformed test images and a manifest. This keeps a
materialized copy of each robustness case for presentations, qualitative
inspection, and repeatable offline tests:

```bash
python -m src.evaluation.materialize_transforms \
  --config configs/config.yaml \
  --input-dir data/raw/real \
  --output-dir data/processed/robustness/real
```

The command can be run separately for `data/raw/ai_generated`. Use
`--cases jpeg_30 noise_0.1 center_crop_80` to materialize only selected cases.

Review per-image errors and pixel-derived diagnostic features:

```bash
python -m src.evaluation.error_analysis \
  --config configs/config.yaml \
  --checkpoint checkpoints/best.pt \
  --output outputs/error_analysis.csv
```

The CSV includes false positives, false negatives, confidence, and features
such as colour statistics, edge density, entropy, and Laplacian variance. These
are image-pixel features for analysis; the detector does not use EXIF data,
filenames, watermarks, or other metadata as its decision rule.

Generate an image-only Grad-CAM explanation for the AI-generated logit:

```bash
python -m src.explainability.gradcam \
  --config configs/config.yaml \
  --checkpoint checkpoints/best.pt \
  --image path/to/image.jpg \
  --output outputs/gradcam.png
```

Run batch inference. `pred` is always the probability that an image is
AI-generated, not the probability that it is real:

```bash
python scripts/run_inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best.pt \
  --input-dir path/to/images \
  --output outputs/predictions.json
```

The output is a JSON list in this form:

```json
[
  {
    "image_path": "path/to/image.jpg",
    "pred": 0.91
  }
]
```

Start the Streamlit demo after a checkpoint exists:

```bash
streamlit run app.py
```

The demo reports AI-generated probability, real-image probability,
classification label, and confidence. Missing directories, missing
checkpoints, unreadable images, unsupported files, and invalid split entries
raise actionable errors in the CLI or are shown in the demo.

## Design notes

- Labels are fixed as real `0` and AI-generated `1` throughout the project.
- The model is an image-only EfficientNet-B0 binary classifier. It does not
  process video or audio and does not rely on metadata, watermarks, or
  filenames.
- Augmentations are implemented with Albumentations and use OpenCV/NumPy for
  the custom downscale/upscale and Gaussian-noise operations.
- Feature engineering provides transparent pixel-level diagnostic signals for
  error analysis; Grad-CAM highlights spatial regions contributing to the
  AI-generated logit. Neither is presented as proof of provenance.
- Validation is deterministic; colour-jitter and Gaussian-noise robustness
  cases are stochastic and can be reproduced by setting the configured seed.
- Checkpoints contain model settings, optimizer state, epoch, and validation
  metrics, which makes the inference and evaluation entry points reproducible.
- Deployment assumptions are intentionally hackathon-scale: one local
  Streamlit process, a local checkpoint, supported still-image uploads, and
  CPU-compatible inference. Uploads are processed in memory by the demo and
  are not persisted by this project. The threshold is configurable and the
  confidence score is not calibrated by default.
- This is a proof of concept, not a production moderation platform or a claim
  that the detector generalizes to unseen generators, transformations, or
  datasets. Track dataset provenance, test for leakage, and inspect errors
  before making any higher-stakes use.
