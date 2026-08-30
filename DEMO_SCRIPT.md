# RobustLens — 3-Minute Demo Script

**Total: 3:00.** Timings are cumulative. Do not show Grad-CAM — it is
unavailable for this dual-branch model and the demo says so honestly.

## Before you start

```bash
./.venv/bin/streamlit run app.py
```

Cold start takes ~15 s (2 GB checkpoint). **Launch it before you begin
speaking.** Have three files ready:

| Slot | File | Why |
|---|---|---|
| A | a clean AI-generated image | scores high, stays stable |
| B | a real photograph | scores low, stays stable |
| C | image A after a chained degradation | triggers abstention |

Prepare C in advance:

```bash
./.venv/bin/python -c "
import sys; sys.path.insert(0,'.')
from PIL import Image
from src.evaluation.chains import build_generation_chains, apply_chain
im = Image.open('YOUR_AI_IMAGE.png').convert('RGB')
spec = [c for c in build_generation_chains(1234) if c.generation == 5][0]
apply_chain(im, spec).save('demo_degraded.jpg', quality=95)
print('chain:', spec.operations)
"
```

---

## 0:00 – 0:25 — The problem

> "An AI-generated image almost never reaches a moderator the way the generator
> made it. It gets screenshotted, re-encoded, downscaled, cropped, re-compressed
> on every reshare. Each step erases the fine high-frequency detail detectors
> depend on.
>
> We measured it: after a single 0.5× resize, the spectral statistics a forensic
> detector would use shift by 41 to 53 percent.
>
> And the failure has a *direction*. Stacking transformations pushes scores
> toward 'authentic' — for every kind of image. A degraded fake doesn't become
> uncertain. It becomes confidently wrong. That's what RobustLens is built to
> refuse."

## 0:25 – 0:45 — Upload and whole-image score

Upload **image A**.

> "One 740-million-parameter SigLIP2 plus DINOv2 detector scores the image.
> That's the primary signal — and we don't pretend we have others."

Point at the AI-generated probability.

## 0:45 – 1:05 — Calibration and the fixed threshold

Scroll to **"How to read these numbers."**

> "Two things are easy to blur, so we separate them explicitly. Green on the
> left means this is a *calibrated probability*, not a raw model score. Green on
> the right means the threshold — 0.69 — was *derived from data*, not picked by
> hand.
>
> It was fitted on clean validation data only, and it's frozen. We never retune
> it per transformation. Retuning would measure the tuner, not the detector."

## 1:05 – 1:30 — Transformed predictions

Scroll to **"Predictions for each image version."**

> "The same image goes through 14 official transformations — JPEG at four
> qualities, blur, resize, noise, colour jitter, crop — and every version is
> re-scored through identical preprocessing.
>
> On held-out data, clean accuracy is 0.819 and the worst transformation is
> 0.778. The largest drop is four percentage points."

## 1:30 – 1:45 — Consistency

Point at the transformation-consistency metric.

> "The spread across those 15 scores is the stability measure. A detector
> reading genuine generation artefacts keeps saying the same thing. One latching
> onto fragile cues swings wildly — and that swing is itself evidence."

## 1:45 – 2:15 — Chained degradation and abstention ⭐

Upload **image C** (the degraded version of A).

> "Same image, after five stacked transformations — the redistribution chain.
>
> Notice the verdict: **Uncertain**. Not 'authentic'. The system detected the
> score had collapsed because the image was *degraded*, not because it's real,
> and withdrew the claim.
>
> These rules were fitted on chain data with a pre-registered bar: abstained
> images have to be at least 1.5× more error-prone than average. On held-out
> chains, declining 12.5% of cases raised accuracy on the rest from 0.750 to
> 0.810, and the declined images were 2.67× more error-prone.
>
> We did not lower that bar to get a passing result. An earlier attempt on
> single transformations selected nothing, and we reported that negative."

**If C does not abstain:** say so plainly — *"this one stayed confident; the
rules fire on about one in eight chained cases"* — and move on. Never imply a
result you did not get.

## 2:15 – 2:35 — Patch heatmap, and what it is not

Scroll to **"Patch-level risk map."**

> "Optionally we tile the image and map which regions moved the model."

Read the on-screen caption aloud, verbatim:

> "*A highlighted region is a suspicious region that influenced the model's
> score. It is not proof of AI editing, a segmentation mask, or a reconstruction
> of editing history.*
>
> And it carries **zero weight** in both the probability and the confidence. We
> tested adding it — every patch mode scored *worse* than whole-image-only. So
> it stays as explainability and nothing more.
>
> Coverage is shown, and unmeasured regions are left untinted — never drawn cold,
> because nothing was measured there."

## 2:35 – 2:45 — JSON output

Click **Download detailed JSON**, or show a terminal:

```bash
./.venv/bin/python scripts/run_inference.py \
    --input-dir demo_images --output outputs/predictions.json
```

> "The submission contract is exactly `image_path` and `pred`. The detailed
> output adds per-transformation scores, calibration provenance, and the
> abstention reasoning — every rule that fired and why."

## 2:45 – 3:00 — Limitations and what's next

> "What we'd want a judge to know:
>
> Local AI edits are our weak point — recall 0.455, and **all twelve** held-out
> false negatives were locally tampered images. The headline 0.819 is carried by
> wholly synthetic images.
>
> We tried fine-tuning for it. AUROC fell from 0.510 to 0.354, so we rejected it
> and kept the original checkpoint. We report the rejection.
>
> We have no unseen-generator claim — the dataset has no per-generator labels,
> and asking for that evaluation raises an error rather than a misleading number.
>
> Next: a detector trained on locally-edited examples with real edit masks, and
> confirming the abstention gain beyond 24 images.
>
> RobustLens estimates likelihood. It isn't proof, and it doesn't catch every
> AI edit."

---

## Do not say

- ❌ "It detects AI images" → ✅ "It estimates the likelihood an image is AI-generated"
- ❌ "The heatmap shows where it was edited" → ✅ "The heatmap shows what influenced the score"
- ❌ "Fine-tuning improved it" → ✅ "Fine-tuning made ranking worse, so we rejected it"
- ❌ "It generalises to unseen generators" → ✅ "Not established — no per-generator labels"
- ❌ Do not open the Grad-CAM section — it is unavailable and reports so.

## If something breaks

| Problem | Response |
|---|---|
| Checkpoint missing | `python3 scripts/setup.py --check` prints the exact fix |
| Cold start slow | Expected: ~15 s to load 2 GB. Launch before speaking |
| Image rejected | Validation rejects <32 px, >50 MP, corrupt files, unsupported formats — this is correct behaviour, say so |
| Abstention doesn't fire | Say it fires on ~1 in 8 chained cases. Don't oversell |
