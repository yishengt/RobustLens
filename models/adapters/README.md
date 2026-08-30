# LoRA adapters

**This directory is empty in a fresh clone, and that is deliberate.** Adapter
weights are git-ignored (`*.safetensors`, `*.pt`) because they are large binaries
that do not belong in version control.

## You almost certainly do not need these

Every adapter published here comes from an experiment that was **rejected on
measurement**. The production RobustLens pipeline uses the **unmodified base
checkpoint** and references no adapter anywhere — not in `configs/config.yaml`,
not in `app.py`, not in `scripts/run_inference.py`.

To run RobustLens:

```bash
python3 scripts/setup.py --all
```

That fetches the base checkpoint. Nothing in this directory is required.

## If you want them anyway

They exist so a collaborator can verify the rejections rather than take them on
trust. Published at **<https://huggingface.co/Dylennnn/techjam>** (public, no
sign-in needed):

```bash
python scripts/download_adapters.py --list       # see what is available
python scripts/download_adapters.py --all        # fetch all four (~134 MB)
python scripts/download_adapters.py --adapter local_edit_smoke
```

| Adapter | Experiment | Outcome |
|---|---|---|
| `local_edit_smoke` | Head-only local-edit fine-tune | **Rejected** — held-out AUROC 0.510 → 0.354 |
| `consistency_classification_only` | Ablation baseline, BCE only | Reference arm |
| `consistency_consistency_mse` | + logit-MSE consistency loss | **Rejected** — no gain |
| `consistency_consistency_kl` | + symmetric-KL consistency loss | **Rejected** — no gain |

All are `head_only`: 1,250,561 trainable parameters against 739,121,216 frozen.
**No second LoRA adapter was added** — the existing adapter tensors in the base
checkpoint were reused.

## Running one

Adapters load onto a model restored from the base checkpoint; they are not
standalone models.

```bash
python scripts/run_inference.py \
    --input-dir path/to/images \
    --adapter-dir models/adapters/local_edit_smoke \
    --no-calibration \
    --output outputs/predictions.json
```

`--no-calibration` is not optional here. The shipped Platt calibration and the
frozen **0.69** threshold were fitted for the base checkpoint and do **not**
transfer to an adapted model. Using them together would report a calibrated
probability that is not calibrated for the model producing it.

## Regenerating instead of downloading

Training is seeded (`seed: 42`), so the smoke adapter can be rebuilt locally in
a few minutes:

```bash
./.venv/bin/python scripts/train_local_edit_lora.py --config configs/lora_finetune_smoke.yaml
```

Metrics should land in the same place. Byte-identical tensors are **not**
guaranteed across different hardware (MPS vs CUDA vs CPU).
