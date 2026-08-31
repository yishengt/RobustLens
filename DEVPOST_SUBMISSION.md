# RobustLens: Transformation-Resistant AI Image Detector

**TikTok TechJam 2026 — Track 5: Robust Detection of AI-Generated Images Under
Real-World Transformations**

**Team:** Li Ren, Yi Sheng

---

## Inspiration

We started with something personal: a friend of ours was deepfaked, and the
image was circulated without consent. Seeing how quickly a fabricated image
could move from a private joke to a public harm made the problem feel much
closer than a typical machine-learning benchmark.

The damage was not only technical. A deepfake can violate someone’s privacy,
modesty, dignity, and sense of safety. It can affect friends, classmates,
families, and anyone whose identity is placed inside an image that never
happened. By the time people realise that an image is false, it may already
have been downloaded, reposted, or shared with people who never see the
correction.

That experience gave our team a clear reason to build. We wanted RobustLens to
assist potential victims, moderators, and communities by providing an early
signal that an image may be AI-generated or unreliable. We did not want to
promise perfect detection or make accusations from weak evidence. We wanted to
help answer a more responsible question: **how much evidence survives after an
image has been edited, compressed, resized, cropped, screenshotted, and shared
again?**

RobustLens is our attempt to turn that concern into something constructive: a
tool that supports privacy and dignity by making uncertainty visible before a
fabricated image can be trusted or spread further.

## What it does

RobustLens estimates the likelihood that an image is AI-generated. It first
scores the original image, then re-scores it after 14 deterministic
transformations representing common redistribution conditions:

- JPEG compression at qualities 90, 70, 50, and 30;
- Gaussian blur at σ 0.5, 1.0, and 2.0;
- downscaling by 0.5× and 0.25× followed by upscaling;
- Gaussian noise at σ 0.02, 0.05, and 0.10;
- colour jitter; and
- centre cropping that retains 80% of the image.

The clean and transformed predictions are combined as:

$$
p_{final}=0.7p_{clean}+0.3\left(\frac{1}{14}\sum_{i=1}^{14}p_i\right)
$$

The system reports calibrated probability, per-transformation scores,
consistency, confidence, and optional patch-risk heatmaps. Compound
transformation chains can activate a safe abstention response: instead of
forcing a potentially misleading class, RobustLens returns `Uncertain` and
explains why. Abstention never flips the predicted class.

## How we built it

We used the Bombek1 SigLIP2 + DINOv2 LoRA checkpoint, with 740,371,777
parameters, under the competition’s 2-billion-parameter limit. The two visual
backbones feed one shared classification head and produce a single image-level
AI-generation score.

The inference pipeline validates the image, standardises it, generates seeded
transformations, scores each view, calculates consistency, applies the frozen
calibration and threshold, evaluates abstention rules, and emits a structured
result. The public batch interface preserves the required JSON contract:

```json
[
  {
    "image_path": "images/example.jpg",
    "pred": 0.84
  }
]
```

The project includes a Streamlit demo, command-line inference, calibration and
robustness evaluation scripts, deterministic chain evaluation, data-quality
audits, and automated tests. It runs on CPU and Apple MPS; CUDA support is
present but was not verified in the current environment.

## Challenges we ran into

The first challenge was that image degradation does not simply make the model
less certain. In our chain experiments, repeated transformations pushed scores
toward “authentic,” creating false-negative pressure. A degraded fake can look
more confidently genuine instead of merely uncertain.

The second challenge was local AI editing. When only a small region is changed,
the surrounding authentic pixels dominate a whole-image prediction. We tested
patch scoring, but every patch mode performed worse than whole-image-only
scoring. We therefore kept the heatmap as an explanation aid and gave it zero
weight in the final probability and confidence.

The third challenge was evaluating honestly with limited data. SID_Set does not
provide suitable generator or camera identities, and the available local-edit
data has no reliable edit masks. We added configuration guards and reported
unseen-generator and localization results as unestablished rather than
inventing them.

## Accomplishments that we're proud of

- Built an end-to-end inference system around a 740-million-parameter model.
- Tested 14 realistic transformations and deterministic multi-step chains.
- Fitted calibration on clean validation data and evaluated it on held-out and
  transformed images.
- Added chain-based abstention that can withdraw a claim without reversing it.
- Detected and quarantined six conflicting-label pairs affecting 12 files.
- Found zero exact cross-split duplicates in the audited dataset.
- Added format, resolution, aspect-ratio, compression-history, and filename
  shortcut checks.
- Verified adapter save/reload determinism during a separate smoke test while
  leaving the original checkpoint unchanged.
- Rejected patch fusion and fine-tuning when measured results did not support
  them.
- Preserved the required `{image_path, pred}` output contract.
- Reached 487 passing tests, with clean Ruff and compilation checks.

## What we learned

We learned that “robustness” is not a single accuracy number. It includes the
direction of failure, calibration under transformation, false-positive costs,
and knowing when the available evidence is insufficient.

On 72 held-out SID_Set images, clean accuracy was 0.8194 and AUROC was 0.8568.
On the harder locally tampered family, AUROC was 0.7008 and recall was 0.4545.
All 12 held-out false negatives were locally tampered images, while wholly
synthetic images achieved AUROC 1.0000 on the same family analysis.

We also learned that a technically attractive feature is not automatically a
useful one. Patch scores, additional consistency signals, and a small
fine-tuning experiment were all tested. When they failed to improve the
pre-registered metrics, we removed them from the production decision path.

## What's next for RobustLens

The next version should focus on locally edited images, because that is where
the current detector is weakest. We would collect a larger and more diverse
dataset with reliable edit masks and evaluate full-image generation separately
from small inpainting or generative-fill edits.

We would also add independent camera-origin and provenance evidence, test
unseen generators and camera models with datasets that provide valid identities,
expand the chain benchmark beyond the current hackathon-scale sample, and
compare 4-, 9-, and 16-patch configurations using localization and calibration
metrics.

RobustLens is not a proof system. Its most trustworthy role today is to provide
an evidence-based likelihood, show how that likelihood changes under common
transformations, and make it possible to say “we do not know” when the evidence
has degraded too far.

### Development tools

- Python 3.9+
- Visual Studio Code
- Git and GitHub
- Codex-assisted development
- Python command-line scripts
- Streamlit
- pytest
- Ruff
- `compileall`
- Apple MPS for accelerated inference
- CPU-compatible inference
- CUDA-compatible code path, not verified in the current environment
- Hugging Face model hosting for checkpoint retrieval

The shipped system is inference-only. A separate head-only fine-tuning smoke
test was run for investigation, but the resulting adapter was rejected and is
not used by the submitted model.

### Models and APIs used

RobustLens uses the **Bombek1 AI Image Detector SigLIP2 + DINOv2 LoRA**
checkpoint:

- architecture: SigLIP2 and DINOv2 dual-backbone classifier;
- LoRA-adapted branches with a shared classification head;
- parameters: 740,371,777, below the 2-billion-parameter limit;
- checkpoint SHA-256: `caae0c005d8e37e7aa086aa241d1c9445d296ef77649004655c14f5c81130d4b`;
- checkpoint size: approximately 2.11 GB;
- inference hardware verified: Apple MPS and CPU;
- no external commercial detection API is required.

The model produces an image-level AI-generation score. Patch analysis uses the
same model for optional local score visualization. Grad-CAM is unavailable for
this dual-branch architecture because the branches use different input
resolutions and token grids; the application reports that limitation instead
of presenting a misleading attribution map.

### Libraries and frameworks

- PyTorch for model inference;
- Hugging Face Transformers for SigLIP2 processing;
- `timm` for DINOv2 processing;
- PEFT/LoRA support for the checkpoint architecture;
- Pillow and OpenCV for image loading and transformations;
- NumPy and SciPy for numerical and optional frequency-domain calculations;
- scikit-learn for calibration and evaluation metrics;
- pandas for tabular outputs;
- Matplotlib and Seaborn for charts;
- Streamlit for the interactive demo;
- pytest for automated tests;
- Ruff for code quality;
- YAML, JSON, and CSV for configuration and evaluation artifacts.

### Datasets and assets used

| Dataset or asset | Role and current use |
|---|---|
| OpenFake | Recorded as the checkpoint's training source; the submitted pipeline does not retrain the checkpoint. |
| SID_Set | Main evaluation resource. It does not provide enough generator or camera identities for those holdout claims. |
| CIFAKE | Sample inference and smoke-testing resource. |
| escher-vismin local edits | Small smoke-test dataset for the rejected head-only fine-tuning comparison. |
| WildFake | Not used in the current results. |
| COCO val2017 | Not used in the current results. |
| DALL·E Advanced | Not used in the current results. |
| Bombek1 checkpoint | Existing trained model used for inference; the original checkpoint remains unchanged. |
| Calibration and threshold artifacts | Platt calibration parameters and frozen threshold 0.69. |
| Evaluation artifacts | Transformation scores, chain scores, robustness metrics, confidence reports, patch ablations, data-quality audits, and JSON/CSV tables. |

The project does not claim that the provided WildFake, COCO, or DALL·E
demonstration resources were used for training or evaluation. They were not
part of the reported results.

---

## 2. Public Code/GitHub Repository

**Repository:** [github.com/yishengt/RobustLens](https://github.com/yishengt/RobustLens)

The repository contains:

- image validation and preprocessing;
- model loading and whole-image classification;
- deterministic single-image transformations and compound chains;
- probability calibration and fixed-threshold evaluation;
- consistency, confidence, and abstention analysis;
- optional patch-risk heatmaps;
- batch inference and the Streamlit demo;
- robustness, calibration, chain, shortcut, and data-quality scripts;
- automated tests and reproducibility documentation.

### Installation and setup

```bash
python3 scripts/setup.py --all
```

To check the environment without changing anything:

```bash
python3 scripts/setup.py --check
```

### Running the demo

```bash
./.venv/bin/streamlit run app.py
```

The demo validates the upload, runs clean and transformed predictions, shows
calibration and consistency information, optionally displays the patch-risk map,
and allows detailed JSON download.

### Running batch inference

```bash
./.venv/bin/python scripts/run_inference.py \
    --input-dir path/to/images \
    --output outputs/predictions.json
```

The public output contract is unchanged:

```json
[
  {
    "image_path": "images/example.jpg",
    "pred": 0.84
  }
]
```

`pred` is the calibrated final probability that the image is AI-generated.
Detailed output additionally includes transformed-view scores, calibration
provenance, consistency, abstention reasoning, patch findings, and errors.

### Reproducing the reported results

```bash
./.venv/bin/python scripts/evaluate_protocol.py --reuse-scores
./.venv/bin/python scripts/evaluate_calibration_robustness.py
./.venv/bin/python scripts/evaluate_chains.py --reuse-scores
./.venv/bin/python scripts/audit_dataset_quality.py
./.venv/bin/python -m pytest -q
./.venv/bin/ruff check src scripts tests app.py
./.venv/bin/python -m compileall -q src scripts app.py tests
```

The fine-tuning comparison and consistency-loss ablation are smoke-scale
historical experiments. They are not part of the shipped inference model and
should not be interpreted as evidence that fine-tuning improves performance.

---

## 3. Demo Video

The three-minute demonstration covers:

1. upload validation;
2. clean whole-image scoring;
3. calibrated probability and the frozen threshold;
4. predictions after the 14 official transformations;
5. consistency and chain-based abstention;
6. the optional patch-risk heatmap and its limitations; and
7. detailed JSON output.

**Public YouTube link:** *Add the public video URL after upload.*

The video should be publicly visible and use only project-owned or permitted
visual material. It should not present the heatmap as proof of editing, claim
unseen-generator generalisation, or describe the rejected fine-tuned adapter as
the production model.

---

## 4. Robustness Evaluation Summary

All current detection results use the original Bombek1 checkpoint, a clean
validation-fitted Platt calibrator, and the frozen threshold **0.69**. The
held-out test set contains **72 SID_Set images**. The chain abstention result
uses a separate **24-image held-out chain split**.

### Clean versus transformed images

| Condition | Accuracy | Notes |
|---|---:|---|
| Clean | **0.8194** | Held-out test, n=72 |
| Average over 14 transformations | **0.8085** | Same frozen threshold |
| Worst transformation: JPEG quality 50 | **0.7778** | Largest drop: **0.0417** |

Whole-image clean metrics at the same threshold:

| Metric | Value |
|---|---:|
| Accuracy | 0.8194 |
| F1 | 0.8471 |
| Recall | 0.7500 |
| False-positive rate | 0.0417 |
| AUROC | 0.8568 |

### Calibration under transformation

| Measurement | ECE | Split |
|---|---:|---|
| Validation, in-sample | 0.1327 | 48-image fitting split |
| Clean, held out | **0.1721** | 72-image test split |
| Worst transformation, held out | **0.1868** | `noise_s0.02` |

The calibrator was fitted on clean validation scores only. The held-out ECE
increase from clean to the worst tested transformation was 0.0148. This is a
small-sample robustness result, not a production calibration guarantee.

### Chain-based abstention

Three abstention rules were accepted on chain-validation data and frozen without
lowering the pre-registered requirements of at least 1.5× error enrichment and
at most 35% abstention: `borderline_margin 0.02`, `min_consistency 0.50`, and
`min_agreement 0.70`.

| Metric | Held-out chains, abstention off | Held-out chains, abstention on |
|---|---:|---:|
| Images | 24 | 24 |
| Abstention rate | 0.000 | **0.125** |
| Accuracy among answered | 0.750 | **0.810** |
| Error enrichment | — | **2.67×** |

The single-transformation sweep selected no active rule because those
transformations barely moved the score. The chain result is more relevant to
redistribution degradation, but it remains indicative because it is based on
only 24 held-out images.

### Evaluation interpretation

The system is strongest on wholly synthetic images and substantially weaker on
locally tampered images. The chain harness found a directional failure: deeper
transformation chains moved scores toward “authentic,” creating false-negative
pressure. The desired response is decreasing confidence or abstention, not a
confident claim that degradation proves authenticity.

---

## 5. Error Analysis Note

### Representative false positives

There was **one false positive among 72 held-out images** (FPR 0.0417). It was
an authentic photograph with a score of approximately 0.82: above the adopted
threshold but not an extreme score. Possible false-positive sources include
unusual textures, digital artwork, heavy compression, and unusual colour or
lighting patterns.

### Representative false negatives

All **12 held-out false negatives** were locally tampered images. By generation
family:

| Family | AUROC | Recall |
|---|---:|---:|
| Wholly synthetic | **1.000** | **1.000** |
| Locally tampered | **0.701** | **0.455** |

The headline clean accuracy of 0.819 therefore overstates practical performance
for the harder local-edit case. A small AI-edited region can be averaged away by
a whole-image detector, especially after compression or rescaling.

### Patch-analysis trade-off

Patch scoring was tested rather than assumed to help. Every tested patch mode
performed worse than whole-image-only scoring, with no reduction in false
positives. Adding patch agreement to confidence also reduced its ability to
separate correct from incorrect predictions. Patch evidence therefore carries
zero weight in probability and confidence and is retained only as a qualitative
heatmap.

A highlighted region means only that the region influenced the model's score. It
is not proof of AI editing, a segmentation mask, or a reconstruction of editing
history. No edit masks were available for the 25,337 audited files, so region
IoU or localization recall could not be measured.

### Transformation-chain trade-off

On the small chain stress sample, generation-5 transformations produced mean
score drift of **−0.0956** and reduced recall from **0.625 to 0.500**. Generation
10 drifted **−0.0870** with recall also at **0.500**. These results show the
direction of degradation but are not large enough for a population claim.

### Fine-tuning decision

A separate head-only fine-tuning smoke test used 68 training images and 20
held-out test images. The adapter saved and reloaded deterministically, and the
original checkpoint remained byte-identical. However, AUROC fell from **0.5104**
to **0.3542**, while no pre-registered metric improved. The adapter was rejected
and is not used in the submitted system. At this sample size, the experiment
demonstrates that the pipeline runs end to end, not that fine-tuning is harmful
or beneficial as a general method.

### Main limitations and future improvements

- True unseen-generator generalisation is not established because SID_Set does
  not provide suitable generator identities.
- Unseen-camera generalisation is not established, and no camera-origin channel
  is implemented.
- The system does not parse or cryptographically verify EXIF/C2PA provenance.
- It applies transformations for evaluation but does not infer an image's exact
  editing or upload history.
- The current system is not a multi-channel forensic detector; it is one binary
  classifier evaluated across multiple views.
- Fine-tuning, consistency-loss ablations, and post-fine-tuning recalibration
  are not adopted as production results.
- Chain abstention and transformation results need larger validation samples.
- Future work would require locally edited training data with reliable masks,
  independent camera/provenance evidence, larger chain benchmarks, and strict
  unseen-source evaluation.

RobustLens estimates the likelihood that an image is AI-generated. It is not a
proof system and does not detect every AI-generated or AI-edited image.

---

## Team contributions

Li Ren and Yi Sheng contributed to the project research, implementation,
evaluation, documentation, and demonstration.
