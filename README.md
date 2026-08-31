# RobustLens

Estimates the likelihood that an image is AI-generated — and reports whether that
answer survives compression, blurring, rescaling, noise, colour shifts and cropping.

```bash
python3 scripts/setup.py                     # venv, deps, model checkpoint
./.venv/bin/streamlit run app.py             # demo
```

---

## Why we built it

We started with something personal: a friend of ours was deepfaked, and the image was
circulated without consent.

Seeing how quickly a fabricated image could move from a private joke to a real
violation of someone's privacy, modesty and dignity made the problem feel much closer
than a typical machine-learning benchmark.

We did not want to build a tool that made accusations with false certainty. We wanted
to ask a more responsible question:

> **How much evidence survives after an image has been edited, compressed, resized,
> cropped, screenshotted, and shared again?**

That question became RobustLens.

## What it does

RobustLens scores the original image, then evaluates it after 14 realistic
transformations — JPEG compression at four qualities, Gaussian blur at three levels,
downscaling and upscaling, Gaussian noise at three levels, colour jitter, and an 80%
centre crop.

The clean and transformed predictions combine into a final calibrated score:

```
p_final = 0.7 · p_clean + 0.3 · mean(p_1 … p_14)
```

The system also measures **prediction consistency** across the transformed versions.
If the score changes significantly, the result is less trustworthy — and for compound
transformation chains RobustLens can **abstain**, returning `Uncertain` rather than
forcing a potentially misleading classification. Abstention only withdraws a claim;
it never reverses the predicted class.

Optional patch analysis produces a heatmap of which regions influenced the score. It
is explainability only — not proof of AI editing, not a segmentation mask, and it
carries zero weight in the probability.

**Model:** the Bombek1 SigLIP2 + DINOv2 LoRA checkpoint — **740,371,777 parameters,
37% of the competition's 2-billion limit**. Two visual backbones feed one shared
classification head and produce a single image-level score. We trained the 1.25M head;
both backbones stayed frozen.

---

## Setup

```bash
python3 scripts/setup.py            # venv + deps + 2.11 GB checkpoint
python3 scripts/setup.py --all      # the above plus a ~2 GB evaluation sample
python3 scripts/setup.py --check    # what's present, what's missing
```

`--check` prints a status table and the exact command that fixes anything missing; it
exits non-zero when something required is absent, so it works in CI. Everything is
idempotent. On Apple Silicon add `--device mps` — about 8.9× faster than CPU.

## Run it

```bash
./.venv/bin/python scripts/run_inference.py \
    --input-dir IMAGES --device mps --output predictions.json
```

```json
[{ "image_path": "images/example.jpg", "pred": 0.84 }]
```

Add `--detailed-output report.json` for labels, per-transformation scores, confidence
and abstention reasoning. Verify the parameter limit:

```bash
./.venv/bin/python scripts/count_params.py --checkpoint models/pretrained/pytorch_model.pt
```

**Built with:** Python, PyTorch, Hugging Face Transformers, timm, PEFT/LoRA, Pillow,
NumPy, SciPy, pyarrow, pandas, Altair, Streamlit, pytest, Ruff. Runs on CPU and Apple
MPS.

---

## What we found

### Robustness is not accuracy on clean images

Repeated transformation pushed scores toward "authentic", creating false-negative
pressure. A degraded fake becomes *confidently wrong* rather than simply uncertain.

| Metric | Clean | Transformed | Change |
|---|---:|---:|---:|
| AUROC | 0.962 | 0.933 | −0.029 |
| **Recall** | **0.890** | **0.744** | **−0.146** |

AUROC barely moves while recall collapses. Report AUROC alone and this is invisible.

### The detector is blind to one generator family

On 646 held-out images:

| Generator | Family | Base recall |
|---|---|---:|
| SD 2.1, SDXL, SD 3 | Latent diffusion | **1.000** |
| Midjourney | Commercial | **1.000** |
| GLIDE | Pixel-space diffusion | 0.739 |
| **ADM** | **Pixel-space diffusion** | **0.305** |

It misses seven of ten ADM images. The base checkpoint was trained on OpenFake, whose
generators are entirely latent-diffusion and commercial models — **not one pixel-space
diffusion model**. Our analysis suggests it learned signals associated with
latent-diffusion pipelines, including traces introduced during VAE decoding. ADM
generates directly in pixels and produces no such signature. The detector was not
learning "AI-generated" in a generator-independent way; it had learned a strong but
incomplete shortcut.

### A confound that nearly faked our own result

In the raw sources, generated images were almost always square and authentic images
usually were not. A model could have learned that **"square means fake"** and produced
near-perfect results for the wrong reason. We detected it, removed it, measured the
residual file-size signal at **AUC 0.685** — and found the same defect in the official
benchmark's specification-faithful configuration.

---

## Results

**A head trained on 4,979 images across six generators closes the blind spot:**

| Group | n | Base | Tuned | Δ |
|---|---:|---:|---:|---:|
| ADM recall | 95 | 0.305 | **0.863** | +0.558 |
| All pixel-space | 187 | 0.519 | **0.914** | **+0.396** |
| Authentic false positives | 459 | 0.013 | 0.072 | +0.059 |

**Abstention works on benchmarks we did not build:**

| Test set | Abstains | Accuracy | Among answered | Error enrichment |
|---|---:|---:|---:|---:|
| Reference benchmark | 45.5% | 0.985 | **1.000** | 2.20× |
| `laion_matched` | 38.9% | 0.975 | **1.000** | 2.57× |

Every error the system makes is one it already declined to answer.

**And what does not work.** The same adapter does *not* improve the DALL·E-only
benchmarks — AUROC −0.011 on one, and on the other the original model won **13 of 14**
conditions at matched false-positive rate. We treat this as evidence of **conditional
specialisation**, not proof that fine-tuning solved generalisation, and we keep the
original checkpoint as the default.

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

```bash
./.venv/bin/python -m pytest -q          # 509 passed, 2 skipped, 350 subtests
./.venv/bin/python -m ruff check .
```

---

## Limitations, and what we would do with more time

- **The detector is not ours.** 0.17% of parameters were trained here. The contribution
  is the measurement layer, the dataset and the blind-spot fix.
- **We are not claiming generalisation.** ADM and GLIDE were both in our training set.
  Whether the improvement transfers to an *unseen* pixel-space model is untested.
- **ADM is still our weakest generator** at 0.863. It is natively 256px and we upscale
  to 384, which may smooth the artifacts we need — rebuilding at native resolution is
  the first thing we would try.
- **Small samples.** Most evaluations use ~200 images, so per-condition figures carry
  roughly ±0.07 intervals. The pixel-space result (n=646) is the exception.
- **High abstention rate** (38.9–55.0%). A triage tool that answers about half of what
  it sees, not a system that decides on everything.
- **Scene leakage in our splits.** The source dataset generates fakes from real images'
  captions, so 52% of scenes appear in more than one split. Re-scoring on scene-disjoint
  images gave a *larger* gain, so the conclusion holds — but the splitter should group
  by scene.
- **Our authentic images are COCO and ImageNet**, both ordinary object photography. The
  tuned head over-flags polished professional photographs.
- **13.3 s per image.** Fine for review tooling, too slow for platform scale.

Also planned: broader generator coverage with strict source separation; reliable edit
masks for localised AI-edit detection; per-objective threshold calibration; larger
abstention benchmarks; 4-, 9- and 16-patch comparison; independent camera-origin and
provenance evidence; and more realistic chained redistribution pipelines.

---

## Documentation

| Document | Read it for |
|---|---|
| **[`OVERVIEW.md`](OVERVIEW.md)** | The project in two minutes |
| **[`RESULTS.md`](RESULTS.md)** | Every table, with sample sizes and intervals |
| [`DEVPOST.md`](DEVPOST.md) | Submission write-up |
| [`PITCH_SCRIPT.md`](PITCH_SCRIPT.md) | Demo and pitch narration |
| [`AUDIT.md`](AUDIT.md) | Independent implementation audit |
| [`docs/reference.md`](docs/reference.md) | Full technical reference — pipeline internals, evaluation protocol, every ablation |

> **Scope note.** `docs/reference.md` and [`FINAL_RESULTS.md`](FINAL_RESULTS.md) report
> the earlier SID_Set evaluation at threshold 0.69. They do not conflict with the
> results above — a different dataset answering a different question.

## Team

<!-- Add contributions here before submitting. -->

## Licence and attribution

Base checkpoint: [Bombek1/ai-image-detector-siglip-dinov2](https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2),
trained on OpenFake. Training data: GenImage (CC-BY-NC-SA-4.0), Defactify/MS COCOAI,
COCO train2017. The fine-tuned adapter inherits **CC-BY-NC-SA-4.0 — non-commercial,
research use only**.

---

RobustLens is not a proof system and does not detect every AI-generated or AI-edited
image. Its most trustworthy role is to provide an evidence-based likelihood, identify
where the detector is weak — especially on ADM — and make it possible to say
**"we do not know"** when the available evidence is insufficient.
