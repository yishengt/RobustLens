# Model checkpoints

This project is **inference-only**. It does not train models — it loads a
checkpoint you provide. If no checkpoint is present, every entry point stops
with a setup error instead of returning fabricated predictions.

## Where to put the checkpoint

The default path is `checkpoints/best.pt`, relative to the project root:

```
checkpoints/
└── best.pt
```

Override it anywhere:

```bash
python scripts/run_inference.py --checkpoint /path/to/your.pt --input-dir images/
```

or by editing `paths.checkpoint_dir` in `configs/config.yaml`. `checkpoints/*.pt`
is git-ignored, so weights are never committed.

## Expected checkpoint format

A `torch.save` file containing either a bare `state_dict` or a dict with one of
the keys `model_state_dict`, `state_dict`, `model`, `net`, or `weights`:

```python
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_name": "efficientnet_b0",   # optional; overrides model.name in the config
        "num_classes": 1,                   # optional; also inferred from the head shape
    },
    "checkpoints/best.pt",
)
```

`module.` and `model.` key prefixes (from `DataParallel` or Lightning) are
stripped automatically.

## Supported architectures

| `model.name`      | Parameters | Grad-CAM layer |
|-------------------|-----------:|----------------|
| `efficientnet_b0` |      ~5.3 M | `features[-1]` |
| `resnet18`        |     ~11.7 M | `layer4`       |
| `convnext_tiny`   |     ~28.6 M | `features[-1]` |

All are far below the 2-billion-parameter competition limit, which
`src/pipeline/model_loader.py` enforces at load time.

## Classification head

Both binary head styles are supported and detected automatically from the
checkpoint's head shape:

- **1 output logit** (`num_classes: 1`) — read through a sigmoid.
- **2 output logits** (`num_classes: 2`) — read through a softmax; the column
  used as the AI-generated class is `model.ai_class_index` (default `1`).

In both cases the pipeline expects **higher output = more likely AI-generated**.
If your model was trained the other way around, flip `model.ai_class_index` for
a 2-class head, or re-export the 1-logit head with a negated final layer.

## Preprocessing contract

The checkpoint must have been trained with preprocessing that matches
`configs/config.yaml`, otherwise the scores will be miscalibrated:

- RGB, resized to **224 × 224** (bilinear)
- pixels scaled to `[0, 1]`
- normalized with ImageNet statistics
  (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`)

Adjust `data.image_size` and the `normalization` block if your model differs.

## Testing without a trained checkpoint

To exercise the CLI, the demo and the tests before a real checkpoint exists:

```bash
python scripts/make_dummy_checkpoint.py --output checkpoints/dummy.pt
```

This writes **randomly initialised weights**. It verifies that the plumbing
works end to end; its predictions are meaningless and must never be reported as
detection results.

## Training

Training is out of scope for this repository. When you do train a model, note
the competition rule that the **WildFake validation subset must not be used for
training** — it is reserved for evaluation.
