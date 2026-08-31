# RobustLens — what this project does

Detects AI-generated images, and reports whether that answer survives the things
that happen to images in the real world: compression, blurring, rescaling, noise,
colour shifts and cropping.

A detector that only works on pristine images is useless, because almost nothing
online is pristine.

---

## How it works

For every image, the system builds **14 degraded versions** — the exact
transformations the competition brief names — and scores all 15.

```
image ──┬─► score the original
        └─► JPEG 90/70/50/30 · blur σ0.5/1/2 · resize 0.5×/0.25×
            noise σ0.02/0.05/0.10 · colour jitter · centre crop 80%
                              │
                    score all 14, then ask:
                              │
        ┌─────────────────────┴─────────────────────┐
   how AI-generated?                     how much did that answer move?
        │                                           │
        └──────────────► verdict ◄──────────────────┘
                      or "Uncertain"
```

If the 15 scores disagree too much, the system **withdraws the verdict** rather
than guessing. For a moderation pipeline, "I don't know" beats a confident
mistake.

## The model

A public checkpoint — Bombek1's SigLIP2 + DINOv2 — used as a frozen feature
extractor. **740,371,777 parameters, 37% of the 2B competition limit.**

| Component | Parameters | Trained here |
|---|---:|---|
| SigLIP2 vision tower | 432,206,912 | No — frozen |
| DINOv2 | 306,914,304 | No — frozen |
| Classifier head | 1,250,561 | **Yes** — 0.17% |

Two backbones because AI images fail two ways: implausible *semantics*, which
SigLIP2 catches, and wrong *texture statistics*, which DINOv2 catches.

---

## What we found

### 1. The detector is blind to a whole family of generators

Measured on 646 held-out images:

| Generator family | Model | Base recall |
|---|---|---:|
| Latent diffusion | SD 2.1, SDXL, SD 3 | **1.000** |
| Commercial | Midjourney | **1.000** |
| **Pixel-space diffusion** | **ADM** | **0.305** |
| **Pixel-space diffusion** | **GLIDE** | 0.739 |

It misses **seven out of ten ADM images**.

The cause is architectural. Latent diffusion decodes every image through a VAE,
which leaves a signature the detector learned to find. Pixel-space models have
no VAE, so there is no signature and nothing to look for.

### 2. Recall collapses under degradation while AUROC hides it

| Metric | Clean | Transformed | Change |
|---|---:|---:|---:|
| AUROC | 0.962 | 0.933 | −0.029 |
| **Recall** | **0.890** | **0.744** | **−0.146** |

Every score slides *downward*, so a compressed fake does not come back
"uncertain" — it comes back **confidently authentic**. Report AUROC alone and
this failure is invisible.

---

## What we built

**A training set** of 4,979 images across 6 generators and 3 architecture
families, with every shortcut removed. In the raw sources *every* generated image
was square and almost none of the authentic ones were — a one-line rule would
have scored ~100% without looking at a single generation artifact. Every image is
now centre-cropped square, resized to 384 and re-encoded at one JPEG quality.

**A fine-tuned head** — 1.25M parameters, trained on augmented copies.

**An abstention layer** that withdraws unstable verdicts.

---

## Results

### The fine-tune closes the blind spot

n=646, thresholds fitted on validation and frozen.

| Group | n | Base | Tuned | Δ |
|---|---:|---:|---:|---:|
| ADM recall | 95 | 0.305 | **0.863** | +0.558 |
| GLIDE recall | 92 | 0.739 | **0.967** | +0.228 |
| **All pixel-space** | 187 | **0.519** | **0.914** | **+0.396** |
| Authentic (false positives) | 459 | 0.013 | 0.072 | +0.059 |

**+0.396 recall for +0.059 false positives** — roughly a 7:1 trade.

### The abstention layer works on benchmarks we did not build

| Test set | Abstains | Accuracy | Among answered | Error enrichment |
|---|---:|---:|---:|---:|
| Reference benchmark | 45.5% | 0.985 | **1.000** | **2.20×** |
| `laion_matched` | 38.9% | 0.975 | **1.000** | **2.57×** |
| Our multi-generator set | 55.0% | 0.890 | 0.978 | 1.65× |

Every error the system makes is one it already declined to answer.

---

## What does *not* work, and we say so

**The fine-tune does not improve the DALL·E-only benchmarks.** Verified twice:

| Benchmark | Result |
|---|---|
| Reference (`normalized`) | AUROC −0.011, improved 3/14 |
| `laion_matched` | At matched false-positive rate, base wins 13/14 |

The reason is consistent with everything above: those benchmarks contain **no
pixel-space diffusion**, and the base model already scores recall 1.000 on
DALL·E 3. There is no blind spot there to fix.

So we **ship the base checkpoint** and publish the adapter as a documented
negative result — alongside the patch-scoring, consistency-loss and two earlier
fine-tuning experiments this project also rejected on measurement.

---

## Run it

```bash
python3 scripts/setup.py                        # venv, deps, 2.11 GB checkpoint
./.venv/bin/streamlit run app.py                # demo

./.venv/bin/python scripts/run_inference.py \
    --input-dir IMAGES --device mps \
    --output predictions.json                   # {image_path, pred} per image
```

Verify the parameter limit:

```bash
./.venv/bin/python scripts/count_params.py --checkpoint models/pretrained/pytorch_model.pt
```

---

## Honest limitations

- **The detector is not ours.** 0.17% of parameters were trained here. The
  contribution is the measurement layer, the dataset, and the blind-spot fix.
- **Small samples.** Most evaluations use ~200 images, so per-condition figures
  carry roughly ±0.07 intervals. The pixel-space result (n=646) is the exception.
- **High abstention rate** (38.9–55.0%). This is a triage tool that answers about
  half of what it sees, not a system that decides on everything.
- **52% of scenes straddle our splits** — the source dataset generates fakes from
  real images' captions. Re-scoring on scene-disjoint images gave a *larger*
  gain, so the conclusion holds, but the splitter should group by scene.
- **Our authentic images are COCO and ImageNet**, both ordinary object
  photography. The tuned head over-flags polished, professional photographs.

Full tables with sample sizes and confidence intervals: [`RESULTS.md`](RESULTS.md).
