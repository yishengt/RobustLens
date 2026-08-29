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
| `dual_backbone`   |      ~740 M | two processor-specific inputs |
| `bombek_siglip2_dinov2` | 740,371,777 | unavailable (two LoRA branches) |

All are far below the 2-billion-parameter competition limit, which
`src/pipeline/model_loader.py` enforces at load time.

## Dual SigLIP2 + DINOv2 checkpoint

Set `model.name: dual_backbone` to load a checkpoint produced by
`src/models/dual_backbone.py`. The loader constructs the frozen
`google/siglip2-so400m-patch14-384` and `facebook/dinov2-large` towers, loads
their separate Hugging Face image processors, and passes both
processor-specific tensors through the model for every original and
transformed image.

The checkpoint must contain the complete `DualBackboneDetector` state dict and
must record `model_name: dual_backbone` (or use that value in the config). This
architecture requires `transformers` and access to the model configuration and
weights in the local Hugging Face cache or on the Hugging Face Hub. Grad-CAM is
reported as unavailable for it because the existing single-input Grad-CAM
implementation cannot attribute both branches safely.

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

---

## External checkpoint: Bombek1 SigLIP2 + DINOv2 (LoRA)

A real pretrained detector published at
<https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2>
(public, ungated). This is the only trained checkpoint this repository is
wired to use out of the box.

### Download (2.11 GB — run this yourself)

```bash
mkdir -p models/pretrained
curl -L --fail --progress-bar \
  https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2/resolve/main/pytorch_model.pt \
  -o models/pretrained/pytorch_model.pt
```

Or with the Hugging Face CLI:

```bash
hf download Bombek1/ai-image-detector-siglip-dinov2 pytorch_model.pt \
  --local-dir models/pretrained
```

Exact size: **2,105,483,083 bytes**. `models/pretrained/pytorch_model.pt` is
covered by the global `*.pt` rule in `.gitignore`, so it is never committed.

### Running it

```bash
./.venv/bin/python scripts/run_inference.py \
  --input-dir data/cifake_sample \
  --checkpoint models/pretrained/pytorch_model.pt \
  --config configs/config.yaml \
  --no-transformations \
  --output outputs/cifake_predictions.json
```

No config change is needed. The loader identifies this checkpoint from its
tensor signature and overrides `model.name`, so the stock config works. To
select it explicitly instead, set `model.name: bombek_siglip2_dinov2`.

Requires `peft` and `timm` (both in `requirements.txt`).

### Why it is a separate architecture, not `dual_backbone`

These two are **not** interchangeable, and no key renaming can make them so.
Verified by constructing both locally:

| | native `dual_backbone` | `bombek_siglip2_dinov2` |
|---|---|---|
| SigLIP2 keys | `siglip.encoder.layers.N.…` | `siglip.base_model.model.encoder.layers.N.…` (PEFT) |
| DINOv2 source | transformers `Dinov2Model` | timm `vit_large_patch14_dinov2.lvd142m` |
| DINOv2 keys | `dinov2.embeddings.…` / `dinov2.encoder.layer.N.…` | `dinov2.blocks.N.attn.qkv.original.…` |
| LoRA tensors | none | 108 SigLIP + 48 DINOv2 |
| head | `head.{0,2,5}`, 2176 → 3584 → 1 | `classifier.head.{0,1,4,7}`, 2176 → 512 → 256 → 1 |
| DINOv2 input | 224 px, ImageNet norm | 392 px, ImageNet norm |
| SigLIP2 input | 384 px | 384 px |
| state-dict keys | 893 | 954 |

Loading is strict against the selected architecture. Pointing this checkpoint
at `dual_backbone` fails with a key-level diff rather than partially loading
and producing meaningless scores.

### Preprocessing

The two branches get different tensors, which is why `input_kind` is `dual`:

- **SigLIP2** — its own published `SiglipImageProcessor` at 384 px with SigLIP
  normalisation.
- **DINOv2** — the upstream torchvision pipeline at 392 px, bicubic resize,
  ImageNet normalisation, wrapped in `TorchvisionImageProcessor` so both
  branches share one call convention.

The 224 px `Preprocessor` used by the CNN paths is bypassed entirely.

### Grad-CAM

Reported as **unavailable** for this architecture. Attribution would have to
flow through two LoRA-adapted transformer branches at different resolutions
with different token grids; no single heatmap can represent both honestly, so
the pipeline returns a clear explanation instead of a misleading picture. All
other outputs — probability, label, confidence, consistency — are unaffected.

### Performance note

The model is 740 M parameters. On CPU a single image takes roughly 7 s, and a
full 15-version transformation sweep about 110 s. Use `--no-transformations`
for quick runs, or `--device mps` / `--device cuda` where available.
