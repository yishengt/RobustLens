# RobustLens — Devpost submission

## Inspiration

We started with something personal: a friend of ours was deepfaked, and the image
was circulated without consent.

What affected us most was how quickly a fabricated image could become a real source
of harm. A deepfake can violate someone's privacy, modesty, dignity, and sense of
safety. It can affect friends, classmates, families, and anyone whose identity is
placed inside an image that never happened. By the time people realise an image is
false, it may already have been downloaded, reposted, and shared beyond their
control.

When we came together for this challenge, we remembered our friend and wanted to
build something that could help potential victims, moderators, and communities
respond earlier. We did not want to create a tool that makes accusations with false
certainty. We wanted to ask a more responsible question:

**How much evidence survives after an image has been edited, compressed, resized,
cropped, screenshotted, and shared again?**

That question became RobustLens.

## What it does

RobustLens estimates the likelihood that an image is AI-generated.

It first analyses the original image, then evaluates the same image after 14
realistic transformations:

- JPEG compression at qualities 90, 70, 50, and 30
- Gaussian blur at σ 0.5, 1.0, and 2.0
- downscaling by 0.5× and 0.25× followed by upscaling
- Gaussian noise at σ 0.02, 0.05, and 0.10
- colour jitter of ±20% brightness, contrast and saturation
- centre cropping that retains 80% of the image

The clean and transformed scores are combined as:

```
final = 0.7 × clean + 0.3 × mean(transformed)
```

The result is calibrated using Platt scaling fitted on clean validation data. Every
threshold is selected on validation data and then **frozen** across all conditions —
re-fitting per condition would conceal exactly the degradation the system exists to
measure.

RobustLens also tests compound transformation chains. When the evidence becomes
unreliable, the system withdraws its claim and returns **Uncertain** rather than
forcing a potentially misleading answer. Abstention never flips an image from
AI-generated to authentic, or from authentic to AI-generated — it only declines.

The project includes optional patch-risk heatmaps. These show which regions
influenced the model's score, but they are not presented as proof of editing or as
segmentation masks, and they carry **zero weight** in the probability.

## How we built it

We built RobustLens as an inference pipeline around the Bombek1 SigLIP2 + DINOv2
LoRA checkpoint — **740,371,777 parameters, 37% of the competition's 2-billion
limit**, verified with a one-command receipt.

The SigLIP2 and DINOv2 branches feed one shared classification head and produce a
single image-level score. Two backbones because AI images fail two ways: implausible
*semantics*, which the language-supervised SigLIP2 catches, and wrong *texture
statistics*, which the self-supervised DINOv2 catches.

The pipeline validates and preprocesses the image; generates seeded transformation
variants; scores the clean and transformed views; calculates prediction consistency
and confidence; applies calibration and the frozen threshold; evaluates abstention
rules; optionally produces patch-risk information; and returns structured JSON.

The required batch output is preserved exactly:

```json
[{ "image_path": "images/example.jpg", "pred": 0.84 }]
```

**Tools:** Python, PyTorch, Hugging Face Transformers, timm, PEFT/LoRA, Pillow,
NumPy, SciPy, pyarrow, pandas, Altair, Streamlit, pytest, Ruff. Runs on CPU and
Apple MPS. Includes a Streamlit interface, command-line inference, reproducible
evaluation scripts, data-quality audits, and 509 automated tests.

## Challenges we ran into

**Degradation does not make the model uncertain — it makes it confidently wrong.**
Under transformation, AUROC barely moved (0.962 → 0.933) while recall collapsed
(0.890 → 0.744, as low as 0.560 under heavy JPEG). Every score slid *downward*, so a
compressed fake does not return "uncertain" — it returns **confidently authentic**.
Report AUROC alone and this failure is invisible. It is the reason the whole project
exists.

**A shortcut in our own data nearly faked the entire result.** When we assembled a
multi-generator training set, *every* generated image was square and almost no
authentic one was — 1024×1024 against 640×480. A one-line rule, "square means fake",
would have scored near 100% without ever looking at a generation artifact, and would
have collapsed the moment anyone resized an image. We centre-cropped everything
square, resized to 384, and re-encoded at one JPEG quality — both classes
identically — then measured what remained rather than assuming it was clean (file
size alone still gives AUC 0.685, which we report).

We then found the **identical defect in the reference benchmark**: in its
spec-faithful configuration every real image is 200×200 and no fake image is, so a
size check alone scores AUC 1.000 with no model at all. We evaluated on the
normalised configuration instead and name the configuration on every number.

**Detecting local AI edits.** When only a small region is changed, the surrounding
authentic pixels dominate the whole-image prediction. We tested patch-based scoring,
but every patch mode performed worse than whole-image-only. We kept patches as
explainability and gave them zero weight in probability and confidence.

**Limited and imperfect data.** SID_Set provides no generator or camera identities
for true unseen-source testing, and available local-edit data lacks reliable masks.
Rather than inventing unsupported results, we added configuration guards that fail
loudly and documented the limitations.

## Accomplishments that we're proud of

**We found a blind spot in the base detector and closed it.** On 646 held-out images
the checkpoint is *perfect* on latent-diffusion and commercial generators and
largely blind to pixel-space diffusion:

| Generator | Family | Base recall |
|---|---|---:|
| SD 2.1, SDXL, SD 3 | Latent diffusion | **1.000** |
| Midjourney | Commercial | **1.000** |
| GLIDE | Pixel-space diffusion | 0.739 |
| **ADM** | **Pixel-space diffusion** | **0.305** |

It misses seven of every ten ADM images. The cause is architectural: latent
diffusion decodes every image through a VAE, leaving a signature the detector had
learned to find. Pixel-space models have no VAE, so there is nothing to find.

A 1.25M-parameter head trained on a 4,979-image, six-generator, confound-normalised
dataset closed it — **ADM 0.305 → 0.863**, all pixel-space **0.519 → 0.914**, for
0.059 of false-positive cost on 459 authentic images. That is roughly seven points
of recall per point of precision.

**Abstention works on benchmarks we did not build.**

| Test set | Abstains | Accuracy | Among answered | Error enrichment |
|---|---:|---:|---:|---:|
| Reference benchmark | 45.5% | 0.985 | **1.000** | 2.20× |
| `laion_matched` | 38.9% | 0.975 | **1.000** | 2.57× |

Every error the system makes is one it already declined to answer.

Also: 14 realistic transformations plus deterministic compound chains; clean-data
calibration with frozen operating thresholds; six conflicting-label pairs detected
and quarantined; zero exact cross-split duplicates; tests for file-format,
resolution, aspect-ratio, compression-history and filename shortcuts; verified
adapter save/reload determinism with the original checkpoint left byte-identical;
the `{image_path, pred}` contract preserved; **509 passing tests** with clean Ruff
and compilation checks.

## What we learned

**Robustness is not one accuracy number.** It includes calibration, confidence
behaviour, false-positive costs, and recognising when the evidence is insufficient.

**A single metric can hide a failure completely.** Our clearest example: on one
benchmark the adapter's recall improved in all 14 transformed conditions, by an
average of 0.114. It looked like a decisive win. At *matched false-positive rate*
the base model won 13 of 14 — every gain was purchasable by simply lowering the
threshold. We did not ship it. Had we stopped at the recall column we would have
reported a result that collapses under the first serious question.

**Ideas must be tested before they enter the decision path.** Patch fusion, a
transformation-consistency loss, and two fine-tuning experiments were each measured
and each rejected or restricted on the evidence:

- **Patch scoring** — every mode scored worse than whole-image-only. Demoted to
  explainability with zero weight.
- **Local-edit fine-tune** — AUROC fell 0.5104 → 0.3542 on a 68-image run. Rejected;
  the original checkpoint was left unchanged.
- **Robustness fine-tune** — genuinely fixes pixel-space diffusion, but does *not*
  improve the DALL·E-only benchmarks (AUROC −0.011 on one; base model wins 13/14 at
  matched FPR on the other). Neither benchmark contains pixel-space diffusion and
  the base model already scores recall 1.000 on DALL·E 3, so there is no blind spot
  there to fix. We ship the base checkpoint and publish the adapter as a documented
  negative result.

**Where the detector remains weakest.** On 72 held-out SID_Set images the original
checkpoint reached clean accuracy 0.8194, AUROC 0.8568, F1 0.8471, FPR 0.0417, worst
transformed accuracy 0.7778. Split by image family:

| Image family | AUROC | Recall |
|---|---:|---:|
| Wholly synthetic | 1.000 | 1.000 |
| Locally tampered | 0.701 | 0.455 |

All 12 held-out false negatives were locally tampered images. A strong whole-image
detector can still fail when only a small part of an authentic image has changed.

**Uncertainty is a useful output.** A detector should not pretend to know the origin
of an image when transformations have destroyed too much evidence.

## What's next for RobustLens

The next version focuses on **locally edited images**, because that is where the
system is weakest — a larger, more diverse dataset with reliable edit masks, and
full-image generation evaluated separately from small inpainting and generative-fill
edits.

We would also **broaden the authentic class**. Ours is COCO and ImageNet, both
ordinary object photography, and the tuned head over-flags polished professional
photographs as a result. Adding LAION-style reals is the single change most likely
to make the fine-tune generalise.

Then: independently validated camera-origin and provenance evidence, so unseen-source
evaluation is meaningful rather than a family-level proxy; larger transformation-chain
validation sets; quantitative comparison of 4-, 9- and 16-patch configurations;
spatial grouping of suspicious patches; stronger calibration under severe
degradation; verified EXIF and C2PA handling; scene-level dataset splitting (52% of
scenes currently straddle splits, because the source generates fakes from real
images' captions); and larger-scale evaluation before any future fine-tuning
decision.

---

RobustLens is not a proof system and does not detect every AI-generated or AI-edited
image. Its most trustworthy role today is to estimate likelihood, show how that
estimate changes under realistic transformations, and make it possible to say
**"we do not know"** when the evidence has degraded too far.
