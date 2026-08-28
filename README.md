# Robust AI-Generated Image Detection

Inference-only pipeline for estimating whether an image is AI-generated under
common real-world transformations. The project loads an existing checkpoint;
it does not train models or use filenames, EXIF metadata, or watermarks.

## Pipeline

```text
Input image → Validation → Preprocessing → Transformations →
Model inference → Consistency check → Prediction output → Explainability
```

## Project layout

```text
.
├── app.py                         # Streamlit upload demo
├── configs/config.yaml            # Inference and transformation settings
├── scripts/run_inference.py       # Batch JSON CLI
├── src/
│   ├── data/
│   │   ├── dataset.py             # Discovery, validation, RGB decoding
│   │   └── augmentations.py       # Configurable robustness transformations
│   ├── explainability/gradcam.py  # Optional Grad-CAM image overlay
│   ├── inference/predict.py       # Checkpoint loading and prediction
│   ├── models/classifier.py        # EfficientNet-B0/ConvNeXt-Tiny builder
│   └── utils/
│       ├── checkpoint.py          # Safe checkpoint restoration
│       └── config.py              # YAML and path helpers
├── tests/test_smoke.py
├── requirements.txt
├── checkpoints/                   # Place an existing .pt checkpoint here
├── data/                          # User datasets; never modified by the CLI
└── outputs/                       # Generated JSON and explanation images
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The default model is EfficientNet-B0 with one binary output logit. A checkpoint
is required; missing or invalid checkpoints produce an actionable error.

## Batch inference

```bash
python3 scripts/run_inference.py \
  --config configs/config.yaml \
  --checkpoint checkpoints/best.pt \
  --input-dir path/to/images \
  --output outputs/predictions.json
```

The output is a JSON list. `pred` is the AI-generated probability in the range
0–1:

```json
[
  {"image_path": "path/to/image.jpg", "pred": 0.91}
]
```

Supported formats are JPG, JPEG, PNG, and WEBP. Images are validated, decoded
as RGB, resized to 224×224, normalized, and converted to PyTorch tensors.

## Explainability

After installing dependencies and placing a checkpoint, generate a Grad-CAM
overlay with:

```bash
python3 -m src.explainability.gradcam \
  --config configs/config.yaml \
  --checkpoint checkpoints/best.pt \
  --image path/to/image.jpg \
  --output outputs/gradcam.png
```

If the selected model has no convolutional feature layer, Grad-CAM reports a
clear compatibility error rather than failing silently.

## Streamlit demo

```bash
streamlit run app.py
```

The demo accepts one image upload and displays the original image, AI and real
probabilities, classification, and confidence. It stops with a setup error if
the configured checkpoint is missing.

## Checks

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src scripts app.py
python3 scripts/run_inference.py --help
```
