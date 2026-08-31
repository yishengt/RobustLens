# RobustLens — presentation script

**3-minute demo** for the Devpost video. **6-minute pitch** for the live event.
Numbers in `code` should be on screen. Do not round them up.

Every figure here is in `RESULTS.md` with its sample size and confidence interval.

---

## 3-minute demo video

### 0:00 — The problem

> An AI-image detector that only works on pristine images is useless, because
> almost nothing on the internet is pristine.
>
> By the time a picture has been uploaded, recompressed by a platform,
> screenshotted and reposted, the fragile cues most detectors learned are gone.
> So we didn't ask "can we detect AI images?" We asked two harder questions:
> **does the answer survive the real world, and does the detector know when it
> doesn't?**

### 0:25 — What it does

> RobustLens takes an image and builds fourteen degraded versions of it — JPEG at
> four qualities, three blur levels, two downscales, three noise levels, colour
> jitter and a centre crop. Exactly the transformations the brief names.
>
> It scores all fifteen and reports two things: a verdict, and how much that
> verdict moved.

*[Screen: Streamlit, one upload → verdict, heatmap, consistency chart]*

### 0:50 — Finding one: the detector is blind to a whole family of generators

> We started from a strong public detector — Bombek1's SigLIP2 plus DINOv2.
> On our test set it is **perfect** on Stable Diffusion 2.1, SDXL, Stable
> Diffusion 3 and Midjourney. Recall `1.000` on all four.
>
> Then we tested pixel-space diffusion — ADM and GLIDE, models that generate
> directly in pixels instead of through a latent decoder.
>
> ADM recall: **`0.305`**. It misses seven out of ten.
>
> The reason is architectural. Latent diffusion decodes every image through a
> VAE, which stamps a signature the detector had learned to find. Pixel-space
> models have no VAE, so there is no signature — and the detector has nothing
> to look for.

### 1:25 — Finding two: recall collapses while AUROC lies

> Second failure, and it's the one that matters for deployment.
>
> Under degradation, AUROC barely moves — `0.962` to `0.933`. The model still
> ranks images correctly. But recall falls from `0.890` to `0.744`, and as low as
> `0.560` under heavy JPEG.
>
> Every score slides *downward*. So a compressed fake doesn't come back
> "uncertain" — it comes back **confidently authentic**. Report AUROC alone and
> you would never see it.

### 1:50 — What we did

> We built a training set of `4,979` images spanning **six generators** across
> three families, normalised so that no shortcut survives, and trained a
> classifier head on augmented copies.
>
> On `646` held-out images: ADM recall `0.305` to **`0.863`**. Pixel-space
> detection overall, **`0.519` to `0.914`** — for `5.9` points of false-positive
> cost. That's about seven points of recall for every point of precision.

### 2:20 — And the part that works everywhere

> The fine-tune fixes a specific blind spot. The **abstention layer** is the piece
> that generalises.
>
> When the fourteen versions disagree, the system withdraws the verdict instead
> of guessing.
>
> On the competition's own benchmark, accuracy among the images it chooses to
> answer is **`1.000`** — up from `0.985`. The images it declines are **`2.2`
> times** more error-prone than the population. Same result on a second, harder
> benchmark: `0.975` to **`1.000`**, `2.57` times enrichment.
>
> **Every error the system makes is one it already told you not to trust.**

### 2:45 — The honest close

> It answers `54.5%` of images. That's a triage tool, not an oracle.
>
> And our fine-tune does **not** help on DALL·E-only benchmarks — we checked, at
> matched false-positive rate, and reported it. There's no pixel-space blind spot
> there to fix.
>
> Everything we rejected is in the log: three datasets, one architecture, and two
> fine-tuning runs we threw away on measurement.

---

## 6-minute pitch

Use the demo script for the first three minutes, then continue.

### 3:00 — Whose model is this (say it before you're asked)

> The detector is not ours. It's Bombek1's SigLIP2 + DINOv2 checkpoint —
> `740,371,777` parameters, `37%` of the two-billion limit, and we have a
> one-command receipt for that.
>
> What's ours: the robustness protocol, the abstention layer, the training data,
> and a head of `1,250,561` parameters. **0.17% of the model.** Both backbones
> stayed frozen and the base checkpoint is byte-unchanged.
>
> Real detection systems are mostly wrappers. The model is a commodity; knowing
> when not to believe it is the product.

### 3:45 — The shortcut that would have faked our result

> When we assembled the training data, **every generated image was square and
> almost no authentic image was.** 1024 by 1024 against 640 by 480.
>
> A one-line rule — "square means fake" — would have scored near 100% on our own
> data without ever looking at a generation artifact. And it would have collapsed
> the moment anyone resized an image.
>
> So we centre-cropped everything square, resized to 384, re-encoded at one JPEG
> quality — both classes identically. Then we measured what was left: file size
> alone still gives `0.685` AUC, and we report that rather than claim it's clean.

### 4:20 — The same flaw is in the benchmark

> Then we found the identical defect in the benchmark we're scored on. In its
> spec-faithful config, every real image is 200 by 200 and no fake image is. A
> size check alone scores **AUC 1.000** with no model at all.
>
> We evaluated on the normalised config instead, and we name the config on every
> number we report.

### 4:50 — How we checked ourselves

> Our raw recall gain looked like `+0.224`. We didn't report that, because false
> positives rose alongside it.
>
> Three checks. AUROC, which is threshold-free. TPR at matched false-positive
> rate, which equalises the operating point. And a scene-disjoint subset, because
> the source dataset generates fakes from real images' captions — 52% of scenes
> straddled our splits.
>
> The gain survived all three, and on scene-disjoint images it was **larger**, not
> smaller. That's how we know it isn't memorisation.
>
> On a different benchmark the same checks **killed** a result: recall looked
> `+0.114` better across all fourteen conditions, but at matched false-positive
> rate the base model won thirteen of fourteen. So we didn't ship it.

### 5:30 — What's weak

> Sample sizes. Two hundred images per sweep, so per-condition figures carry
> roughly `±0.07` intervals. The pixel-space result is the exception — `646`
> images, tight intervals.
>
> Our headline gains are in-distribution. And the abstention rate is high enough
> that this is a triage system, not a decision system.

### 5:50 — Close

> We treated measurement as the deliverable. We killed our own patch scoring, our
> own consistency loss, and two fine-tuning runs — each on evidence.
>
> What survived: a detector that covers a generator family the original missed,
> and a layer that tells you when not to trust it.

---

## Q&A preparation

**"Did you actually train anything?"**
> The classifier head — 1.25M parameters, best at epoch 3. Not the backbone. The
> base checkpoint is byte-unchanged and we ship a 32 MB adapter.

**"Your fine-tune didn't improve the benchmark."**
> Correct, and we say so. The benchmark is DALL·E 3 only, which the base model
> already detects at recall `1.000`. There's no blind spot there to fix. Our
> improvement is on pixel-space diffusion, which that benchmark contains none of.

**"Isn't 0.99 AUROC suspicious?"**
> On our own set, yes, and we checked. It's in-distribution, and we verified it
> isn't memorisation. The benchmark's `0.999` is the base model's, not ours.

**"Why is your false-positive rate worse?"**
> Because we optimised for missed fakes. On pixel-space images the trade is
> `+0.396` recall for `+0.059` false positives — about seven to one. On DALL·E
> images it's roughly one to one, which is why we don't ship it there.

**"What would you do with more time?"**
> Train the backbone LoRA, not just the head — 2.2 hours per epoch and we had a
> laptop. Add LAION-style reals; our authentic class was COCO and ImageNet, both
> ordinary object photography, and the head over-flags polished photographs.
> And fix scene-level splitting.

---

## Screen cues

| Time | Show |
|---|---|
| 0:25 | Streamlit, one upload → verdict, heatmap, consistency chart |
| 0:50 | Per-generator recall table — four at 1.000, ADM at 0.305 |
| 1:25 | Clean-vs-transformed table, TPR column highlighted |
| 1:50 | ADM before/after, n=646 |
| 2:20 | **An abstained image** — confident clean, unstable degraded |
| 2:45 | Abstention table: 0.985 → 1.000, 2.20× |
| 3:45 | Square-fake vs non-square-real montage |
| 4:20 | The benchmark's `cheat()` function, four lines |

## Do not say

- "Our model" for the detector — say "the detector we built on"
- "99% accurate" without naming the dataset and config
- Any number from the benchmark's `default` config
- A per-condition figure as if it were precise — the intervals are ±0.07
