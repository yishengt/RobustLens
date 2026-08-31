# RobustLens

Detects AI-generated images — and reports whether that answer survives compression,
blurring, rescaling, noise, colour shifts and cropping.

A detector that only works on pristine images is useless, because almost nothing
online is pristine.

```bash
python3 scripts/setup.py                     # venv, deps, model checkpoint
./.venv/bin/streamlit run app.py             # demo
```

---

## How it works

For every image the system builds **14 degraded versions** — the transformations the
competition brief names — scores all 15, and reports two things: a verdict, and how
much that verdict moved.

```
image ─┬─► score the original
       └─► JPEG 90/70/50/30 · blur σ0.5/1/2 · resize 0.5×/0.25×
           noise σ0.02/0.05/0.10 · colour jitter · centre crop 80%
                          │
              ┌───────────┴───────────┐
        how AI-generated?      how stable is that answer?
              └───────────┬───────────┘
                    verdict, or "Uncertain"
```

If the 15 scores disagree too much the system **withdraws the verdict** rather than
guessing. For a moderation pipeline, "I don't know" beats a confident mistake.

**Model:** a public checkpoint (Bombek1 SigLIP2 + DINOv2) used as a frozen feature
extractor. **740,371,777 parameters — 37% of the 2B limit.** We trained the 1.25M
classifier head; both backbones stayed frozen.

---

## Setup

```bash
python3 scripts/setup.py            # venv + deps + 2.11 GB checkpoint
python3 scripts/setup.py --all      # the above plus a ~2 GB evaluation sample
python3 scripts/setup.py --check    # what's present, what's missing
```

`--check` prints a status table and the exact command that fixes anything missing;
it exits non-zero when something required is absent, so it works in CI.

Everything is idempotent. On Apple Silicon add `--device mps` to any command — about
8.9× faster than CPU.

## Run it

```bash
./.venv/bin/python scripts/run_inference.py \
    --input-dir IMAGES --device mps --output predictions.json
```

Output is one record per image:

```json
[{"image_path": "/abs/path/0000.jpg", "pred": 0.009768}]
```

Add `--detailed-output report.json` for labels, per-transformation scores, confidence
and abstention reasoning.

Verify the parameter limit:

```bash
./.venv/bin/python scripts/count_params.py --checkpoint models/pretrained/pytorch_model.pt
```

---

## What we found

### The base detector is blind to a whole family of generators

Measured on 646 held-out images:

| Generator | Family | Base recall |
|---|---|---:|
| SD 2.1, SDXL, SD 3 | Latent diffusion | **1.000** |
| Midjourney | Commercial | **1.000** |
| GLIDE | Pixel-space diffusion | 0.739 |
| **ADM** | **Pixel-space diffusion** | **0.305** |

It misses seven of every ten ADM images. Latent diffusion decodes through a VAE and
leaves a signature the detector had learned to find; pixel-space models have none.

### Recall collapses under degradation while AUROC hides it

| Metric | Clean | Transformed | Change |
|---|---:|---:|---:|
| AUROC | 0.962 | 0.933 | −0.029 |
| **Recall** | **0.890** | **0.744** | **−0.146** |

Scores slide *downward*, so a compressed fake comes back **confidently authentic**
rather than uncertain. Report AUROC alone and this is invisible.

---

## Results

**A head trained on 4,979 images across 6 generators closes the blind spot:**

| Group | n | Base | Tuned | Δ |
|---|---:|---:|---:|---:|
| ADM recall | 95 | 0.305 | **0.863** | +0.558 |
| All pixel-space | 187 | 0.519 | **0.914** | **+0.396** |
| Authentic false positives | 459 | 0.013 | 0.072 | +0.059 |

**+0.396 recall for +0.059 false positives** — roughly 7:1.

**Abstention transfers where the fine-tune does not:**

| Test set | Abstains | Accuracy | Among answered | Error enrichment |
|---|---:|---:|---:|---:|
| Reference benchmark | 45.5% | 0.985 | **1.000** | 2.20× |
| `laion_matched` | 38.9% | 0.975 | **1.000** | 2.57× |

Every error the system makes is one it already declined to answer.

**What does not work, and we say so.** The same adapter does *not* improve the
DALL·E-only benchmarks — AUROC −0.011 on one, and on the other the base model wins
13 of 14 conditions at matched false-positive rate. Neither contains pixel-space
diffusion, and the base model already scores recall 1.000 on DALL·E 3. We therefore
**ship the base checkpoint** and publish the adapter as a documented negative result.

Full tables with sample sizes and confidence intervals: **[`RESULTS.md`](RESULTS.md)**.

---

## Reproduce

```bash
./.venv/bin/python scripts/build_training_mix.py            # the dataset
./.venv/bin/python scripts/build_augmented_train.py --views 3 --limit 2000
./.venv/bin/python scripts/train_local_edit_lora.py \
    --config configs/robustness_head.yaml --mode head_only --device mps
./.venv/bin/python scripts/run_inference_chunked.py \
    --input-dir IMAGES --detailed-output out.json --device mps
./.venv/bin/python scripts/robustness_table.py --detailed out.json --labels labels.json
./.venv/bin/python scripts/compare_robustness.py --baseline base.json --candidate tuned.json
```

Repository checks:

```bash
./.venv/bin/python -m pytest -q          # 509 passed, 2 skipped, 350 subtests
./.venv/bin/python -m ruff check .
```

---

## Limitations, and what we would do with more time

- **The detector is not ours.** 0.17% of parameters were trained here. The contribution
  is the measurement layer, the dataset and the blind-spot fix.
- **Small samples.** Most evaluations use ~200 images, so per-condition figures carry
  roughly ±0.07 intervals. The pixel-space result (n=646) is the exception.
- **High abstention rate** (38.9–55.0%). This is a triage tool that answers about half
  of what it sees, not a system that decides on everything.
- **Scene leakage in our splits.** The source dataset generates fakes from real images'
  captions, so 52% of scenes appear in more than one split. Re-scoring on scene-disjoint
  images gave a *larger* gain, so the conclusion holds — but the splitter should group
  by scene, and that is the first thing we would fix.
- **Our authentic images are COCO and ImageNet**, both ordinary object photography. The
  tuned head over-flags polished, professional photographs. Adding LAION-style reals is
  the change most likely to make the fine-tune generalise.
- **13.3 s per image.** Fine for review tooling, too slow for platform scale. Batching
  the 15 forward passes would be the obvious win.
- With more compute we would train the backbone LoRA rather than only the head —
  2.2 hours per epoch, and we had a laptop.

---

## Documentation

| Document | Read it for |
|---|---|
| **[`OVERVIEW.md`](OVERVIEW.md)** | The project in two minutes |
| **[`RESULTS.md`](RESULTS.md)** | Every table, with sample sizes and intervals |
| [`PITCH_SCRIPT.md`](PITCH_SCRIPT.md) | Demo and pitch narration |
| [`DEVPOST_SUBMISSION.md`](DEVPOST_SUBMISSION.md) | Submission write-up |
| [`AUDIT.md`](AUDIT.md) | Independent implementation audit |
| [`docs/reference.md`](docs/reference.md) | Full technical reference — pipeline internals, evaluation protocol, every ablation |

> **Scope note.** `docs/reference.md` and [`FINAL_RESULTS.md`](FINAL_RESULTS.md) report
> the earlier SID_Set evaluation at threshold 0.69. They do not conflict with the
> results above — they are a different dataset answering a different question.

## Team

<!-- Add contributions here before submitting. -->

## Licence and attribution

Base checkpoint: [Bombek1/ai-image-detector-siglip-dinov2](https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2).
Training data: GenImage (CC-BY-NC-SA-4.0), Defactify/MS COCOAI, COCO train2017.
The fine-tuned adapter inherits **CC-BY-NC-SA-4.0 — non-commercial, research use only**.
