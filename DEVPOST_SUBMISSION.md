# RobustLens: Transformation-Resistant AI Image Detector

**TikTok TechJam 2026 — Track 5: Robust Detection of AI-Generated Images Under
Real-World Transformations**

---

## The problem

An AI-generated image almost never reaches a moderator in the state the
generator produced it. Between creation and review it is screenshotted,
re-encoded by a messaging app, downscaled by a CDN, cropped to fit a feed,
sharpened by an auto-enhance filter, and re-compressed again on every reshare.

Each step erases exactly the evidence detectors depend on. Generative models
leave their fingerprints in fine, high-frequency structure — the first thing
JPEG quantisation and resampling throw away. A detector tuned on pristine
generator output can look excellent in a benchmark and fail on the same image
after one trip through a social platform.

We measured how badly. Across 12 real photographs, the spectral statistics a
classical forensic detector would rely on move by **41–53%** after a single
0.5× resize, while surviving a light JPEG almost untouched. Detection has to
survive the whole redistribution chain, not one transformation at a time.

Worse, the failure is **directional**. We built a chained-degradation harness
and found that stacking transformations pushes scores toward "authentic" for
*every* origin. A degraded fake does not become uncertain — it becomes
confidently wrong. Confidence is highest exactly where evidence is weakest.
That is the specific failure RobustLens is built to refuse.

---

## The solution

RobustLens scores an image, then interrogates its own answer.

**Whole-image AI detection.** A single strong classifier produces one calibrated
probability that the image is AI-generated. This is the primary signal, and the
system does not pretend to have others.

**Transformation testing.** Every image is re-scored through **14 official
transformations** — JPEG at four qualities, three blur levels, two resize
factors, three noise levels, colour jitter, and a centre crop — all applied to
the original at full resolution and passed through identical preprocessing.

**Prediction consistency.** The spread of those 15 scores becomes a stability
measure. A detector reading genuine generation artefacts keeps saying roughly
the same thing; one latching onto fragile cues swings wildly.

**Probability calibration.** Platt scaling, fitted on **clean validation data
only**, converts the raw model score into a probability that means what it says.
Held-out clean ECE is **0.1721**, and it holds to **0.1868** in the worst
transformation — calibration survives redistribution.

**One fixed threshold.** **0.69**, selected on clean validation data at the
balanced operating point and **frozen across every condition**. It is never
retuned per transformation or per chain; doing so would measure the tuner, not
the detector.

**Evidence-driven abstention.** Probability bands alone cannot catch an image
whose score collapsed because it was degraded rather than because it is
authentic. Rules fitted on chain data withdraw the claim when the score sits too
near the threshold, when stability is low, or when transformed versions
disagree. On held-out chains this abstains on **12.5%** of images and lifts
accuracy among answered cases from **0.750 to 0.810**, with abstained images
**2.67× more error-prone** than the population. Abstention only ever withdraws a
claim; it never flips an image to the opposite class.

**Optional suspicious-region heatmaps.** Patch scoring tiles the image and maps
which regions moved the model. It is explainability only, and carries **zero
weight** in both probability and confidence.

**Honest uncertainty.** The interface distinguishes four states that are easy to
blur: a calibrated probability versus a raw score, and a data-derived threshold
versus an interface default. Where evidence is missing, the weight is dropped
and the rest renormalised — a missing signal never counts as evidence.

---

## Technical implementation

| Component | Detail |
|---|---|
| Detector | **Bombek1 SigLIP2 + DINOv2 LoRA** ensemble |
| Parameters | **740,371,777** — under the 2B competition limit |
| Input | **Image pixels only**. EXIF and filenames are discarded before inference |
| Transformations | **14 official**, seeded and reproducible |
| Hardware | **Apple MPS** and **CPU**; CUDA supported |
| Interface | **Streamlit** demo, cold start 14.7 s |
| Batch mode | JSON inference over a directory |
| Patch analysis | **Explainability only** — zero scoring weight |
| Calibration | Platt scaling, clean validation only |
| Tests | **487 passing** |

The pipeline runs: validation → preprocessing → transformation generation →
classification → patch scoring (optional) → consistency → fusion → calibration →
abstention → confidence → output.

**Fusion** is deliberately simple:

```
final = 0.7 · whole_image + 0.3 · mean(14 transformed versions)
```

Patch evidence is absent from this formula because an ablation showed it made
results worse — not because it was never tried.

**Output contract** (unchanged):

```json
[
  {
    "image_path": "images/example.jpg",
    "pred": 0.84
  }
]
```

A detailed JSON output adds per-transformation scores, consistency, calibration
provenance, abstention reasoning, and patch findings.

---

## Results

All measured in this repository. Full provenance in `FINAL_RESULTS.md`.

**Detection — 72 held-out SID_Set images, threshold 0.69 frozen**

| Metric | Value |
|---|---:|
| Clean accuracy | **0.819** |
| Worst transformed accuracy | **0.778** |
| Largest accuracy drop | **0.042** |
| False positive rate | **0.042** |
| F1 | 0.847 |
| AUROC | 0.857 |

**Calibration**

| Measurement | ECE |
|---|---:|
| Validation (in-sample, the fitting split) | 0.1327 |
| **Clean, held out** | **0.1721** |
| **Worst transformation, held out** | **0.1868** |

**Chain abstention — held out, 24 images**

| Metric | Value |
|---|---:|
| Error enrichment | **2.67×** |
| Accuracy among answered | **0.810** (vs 0.750 with abstention off) |
| Abstention rate | 0.125 |

**Two components were rejected on measurement, not preference:**

- **Patch scoring did not improve performance.** Every patch mode scored worse
  than whole-image-only, with no reduction in false positives. Patch evidence
  carries zero weight in probability and confidence, and survives only as an
  explainability heatmap.
- **The fine-tuned model was rejected.** AUROC fell from 0.510 to **0.354**. The
  original checkpoint remains in production, byte-identical.

**Quality:** **487 tests pass**, ruff clean, compileall clean, Streamlit starts
without errors.

---

## Error analysis

**False positives are rare and mild.** One false positive across 72 held-out
images (FPR 0.042). It was a genuine photograph scoring 0.820 — above threshold,
but not extreme. The system does not systematically flag authentic photography.

**False negatives are concentrated, and that is the real story.** All **12**
held-out false negatives were **locally tampered** images. Split by generation
family, the detector scores AUROC **1.000** on wholly synthetic images and
**0.701** on locally tampered ones, with recall of just **0.455**. The headline
accuracy of 0.819 is carried almost entirely by fully generated images.

**Local tampering is the weakness.** A small edited region is averaged away
across a whole frame that is otherwise a real photograph. This is both the
harder problem and the more consequential one for moderation, and we report it
rather than burying it in an aggregate.

**Transformation chains degrade evidence directionally.** Every chain we tested
drifted scores *downward*, toward "authentic". The deepest chains drifted most
(generation-5: −0.096), and recall fell from 0.625 to 0.500. The concerning part
is not the drop but the direction: degradation manufactures false negatives
while confidence rises.

**Why abstention is useful.** Because of that direction, a degraded fake exits
the uncertain band *out of the bottom* rather than stopping inside it. Fixed
probability bands cannot catch this; rules that look at drift, stability and
agreement can. On held-out chains, declining 12.5% of cases raised accuracy on
the rest from 0.750 to 0.810, and the declined images were 2.67× more
error-prone. Withdrawing a claim is the one action that is always safe when
evidence is thin.

**Why patch evidence was removed from scoring.** Two independent runs found
every patch mode scored *worse* than whole-image-only (F1 −0.019, recall −0.025)
with no reduction in false positives. Adding patch agreement to confidence made
it worse at telling correct predictions from incorrect ones (AUROC 0.768 →
0.742). Gating it to "reliable" patches did not rescue it. The evidence did not
support the weight, so the weight was removed.

**Why the fine-tuned model was not adopted.** A head-only fine-tune on local
edits completed cleanly, saved and reloaded bit-exactly, and left the original
checkpoint untouched — but AUROC fell from 0.510 to 0.354 while nothing
improved. The pre-registered rule required an improvement in local-edit recall
or F1; none appeared, so the adapter was rejected.

> **On sample sizes.** The fine-tuning comparison used **68 training and 20 test
> images**. That is enough to prove the pipeline runs end to end and nowhere near
> enough to conclude anything about fine-tuning as a method. We report it as a
> smoke test. The detection and abstention results rest on 72 and 24 held-out
> images respectively — hackathon scale, indicative, not definitive.

---

## Limitations

We would rather state these than have a judge find them.

- **Local AI-edit detection remains weak.** Recall 0.455, AUROC 0.701 on
  locally tampered images.
- **All 12 held-out false negatives were tampered images.** The failure mode is
  systematic, not random.
- **Moderate-edit and transformed fine-tuning subgroups were empty.** The
  available dataset provides only one edit severity, so severity could not be
  reported separately.
- **The fine-tuning comparison used only 68 training and 20 test images.** No
  conclusion about fine-tuning as a method can be drawn from it.
- **No edit masks were available** for any of the 25,337 dataset images, so
  heatmap overlap against ground-truth edited regions could not be evaluated.
  Heatmaps are qualitative explainability only.
- **True unseen-generator generalisation is not established.** SID_Set publishes
  no per-generator labels, so our `full_synthetic` / `tampered` split is a
  *generation-family proxy*, not a generator holdout. Requesting a
  generator-held-out evaluation without valid labels raises a configuration
  error rather than producing a misleading number.
- **Heatmaps are not proof of AI editing.** A highlighted region is a region
  that influenced the model's score. It is not a segmentation mask.
- **The system cannot reconstruct editing history.** It cannot tell you how many
  times an image was edited, re-uploaded or re-encoded. That information is not
  recoverable from a single image.
- **Threshold 0.69 applies only to the current original calibrated model.** It
  does not transfer to any other checkpoint, including a fine-tuned one.
- **Grad-CAM is unavailable** for this dual-branch architecture. The two
  branches use different input resolutions and token grids, so no single
  attribution map could faithfully represent both. We report it as unavailable
  rather than drawing a misleading map.

RobustLens estimates the likelihood that an image is AI-generated. It is not a
proof system, and it does not detect every AI edit.

---

## Running it

```bash
python3 scripts/setup.py --all      # venv, dependencies, checkpoint, dataset
./.venv/bin/streamlit run app.py    # interactive demo

./.venv/bin/python scripts/run_inference.py \
    --input-dir path/to/images \
    --output outputs/predictions.json
```

`scripts/setup.py --check` reports anything missing with the exact command to
fix it.
