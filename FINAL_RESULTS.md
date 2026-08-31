# RobustLens — Final Results

> **Scope.** This document reports the **SID_Set** evaluation at the frozen
> threshold **0.69**. It predates the multi-generator work; for the blind-spot
> finding and the benchmark results see [`RESULTS.md`](RESULTS.md).


Every number here was produced in this repository and can be regenerated from
the commands listed. Nothing is estimated.

**Reading the tables.** Each carries its dataset, sample size, checkpoint,
threshold, split, and status:

| Status | Meaning |
|---|---|
| **Current** | Produced with the shipped checkpoint and the frozen calibrated threshold **0.69** |
| **Historical** | Produced before calibration was fitted, at threshold **0.42**. Kept for provenance. **Not comparable** with current results |
| **Smoke** | Ran end to end on a deliberately tiny sample. Demonstrates the machinery, **not** the science |

**Adopted model throughout:** Bombek1 SigLIP2 + DINOv2 LoRA, 740,371,777
parameters, SHA-256 `caae0c005d8e37e7aa086aa241d1c9445d296ef77649004655c14f5c81130d4b`.
The fine-tuned adapter was **rejected** and is not used anywhere below.

---

## 1. Clean versus transformed performance

| | Value |
|---|---|
| Dataset | SID_Set validation shards |
| Images | 120 scored; **72 held-out test** (48 used for threshold selection) |
| Checkpoint | Bombek1 SigLIP2 + DINOv2 LoRA (original) |
| Threshold | **0.69**, frozen — never retuned per transformation |
| Split | **Held-out test** |
| Status | **Current** |
| Source | `outputs/protocol_calibrated/metrics.json` |

| Condition | Accuracy |
|---|---:|
| Clean | **0.8194** |
| Average across 14 transformations | 0.8085 |
| **Worst transformation** (`jpeg_q50`) | **0.7778** |
| Largest accuracy drop | **0.0417** |

Whole-image-only detection metrics at the same threshold (n=72):

| Metric | Value |
|---|---:|
| Accuracy | 0.8194 |
| F1 | 0.8471 |
| Recall | 0.7500 |
| **FPR** | **0.0417** |
| AUROC | 0.8568 |

Per generation family (each family scored against the shared authentic pool):

| Family | n | Accuracy | AUROC | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| Full synthetic | 50 | 0.9800 | **1.0000** | 1.0000 | 0.0417 |
| Locally tampered | 46 | 0.7174 | **0.7008** | **0.4545** | 0.0417 |

> The headline 0.819 is carried by wholly synthetic images. Locally tampered
> images are close to the weak end, and that gap is the system's main
> limitation.

**Reproduce:** `./.venv/bin/python scripts/evaluate_protocol.py --reuse-scores`

---

## 2. Calibration

| | Value |
|---|---|
| Dataset | SID_Set (same 120 images) |
| Method | Platt scaling, fitted on **clean validation scores only** |
| Checkpoint | Bombek1 (original) |
| Status | **Current** |
| Source | `outputs/calibration_robustness/calibration_robustness.json` |

| Measurement | ECE | Split | n | Note |
|---|---:|---|---:|---|
| Reported in confidence report | **0.1327** | validation — **the fitting split** | 48 | **In-sample.** Optimistic |
| Clean, held out | **0.1721** | held-out test | 72 | +0.0394 optimism vs in-sample |
| Mean over 14 transformations | 0.1480 | held-out test | 72 | No degradation |
| **Worst transformation** (`noise_s0.02`) | **0.1868** | held-out test | 72 | +0.0148 vs clean — within the 0.05 limit |

**Calibration survives transformation.** Fitted only on pristine images, it
stays within 0.015 ECE of clean performance across all 14 transformations.

Reliability on the held-out split (5 bins): the 0.8–1.0 bin holds 33 images at
mean confidence 0.943 and **0.970 actual accuracy**. The 0.4–0.6 bin holds 23
images at mean confidence 0.509 and only **0.304 accuracy** — the mid band is
where the system is unreliable, and where tampered images land.

**Reproduce:** `./.venv/bin/python scripts/evaluate_calibration_robustness.py`

---

## 3. Threshold comparison

| | Value |
|---|---|
| Fitted on | Clean scores of the **validation split only** (48 images) |
| Checkpoint | Bombek1 (original) |
| Status | **Current** |
| Source | `outputs/calibration.json` |

| Operating point | Threshold | Adopted |
|---|---:|---|
| **balanced** (Youden's J) | **0.69** | ✅ **shipped** |
| f1_optimal | 0.53 | no |
| low_false_positive | 0.78 | no |
| high_recall | 0.01 | no |

`balanced` was chosen on measured held-out evidence: `f1_optimal` maximised F1
on the 48-image validation split but transferred worse to the held-out split.

> **Historical:** an earlier threshold of **0.42** appears in `README.md` and in
> `outputs/protocol/`. It predates calibration. Do not compare 0.42 results with
> 0.69 results.

Threshold 0.69 applies **only** to this checkpoint. It was deliberately *not*
reused for the fine-tuned comparison.

---

## 4. Abstention (transformation chains)

| | Value |
|---|---|
| Dataset | SID_Set images through compound chains |
| Images | 48, split **24 chain-validation / 24 chain-test** |
| Checkpoint | Bombek1 (original) |
| Threshold | **0.69**, frozen |
| Status | **Current** |
| Source | `outputs/abstention_chains/abstention_on_chains.json` |

| | Chain validation (n=24) | **Chain held out (n=24)** | Held out, abstention OFF |
|---|---:|---:|---:|
| Abstention rate | 0.292 | **0.125** | 0.000 |
| Accuracy among answered | 0.765 | **0.810** | 0.750 |
| Error enrichment | 1.90× | **2.67×** | — |

**Three rules accepted and frozen:** `borderline_margin 0.02`,
`min_consistency 0.50`, `min_agreement 0.70`.

Pre-registered bars — **≥1.5× error enrichment, ≤35% abstention rate** — were
**not lowered**. A prior sweep on *single* transformations selected nothing
(drift and consistency rules fired on 0.000 of images at every threshold), which
is why fitting moved to chains. Rules that still failed are parked at settings
they cannot fire at rather than tuned to look active.

**Reproduce:** `./.venv/bin/python scripts/select_abstention_on_chains.py`

---

## 5. Patch ablation

| | Value |
|---|---|
| Dataset | SID_Set |
| Images | 60 |
| Checkpoint | Bombek1 (original) |
| Threshold | 0.42 — **historical**; the conclusion is current |
| Split | validation |
| Status | **Historical threshold, current conclusion** |
| Source | `outputs/patch_ablation/ablation.json` |

| Mode | Accuracy | F1 | Recall | FPR | AUROC | passes/img | s/img |
|---|---:|---:|---:|---:|---:|---:|---:|
| **whole-image only** | **0.750** | **0.776** | **0.650** | 0.050 | **0.844** | 0 | 0.56 |
| coarse | 0.733 | 0.758 | 0.625 | 0.050 | 0.838 | 4.0 | 2.49 |

**Every patch mode scored worse than whole-image-only**, with no reduction in
false positives. Patch evidence therefore carries **zero weight** in probability
fusion (`fusion.mode: rgb_transform`).

A separate confidence ablation (`outputs/confidence/`) found that adding patch
agreement to confidence made it **worse** at separating correct from incorrect
predictions (AUROC 0.7684 → 0.7415), so `confidence.patch_agreement_weight` is
`0.0`.

Patch heatmaps remain available as **optional explainability only**.

---

## 6. Fine-tuning comparison

| | Value |
|---|---|
| Dataset | escher-vismin local edits + SID_Set synthetic replay |
| Images | **68 train / 17 validation / 20 held-out test** |
| Checkpoint | Bombek1 original vs head-only fine-tuned adapter |
| Threshold | **0.5** — a calibration fitted for one model does not transfer to another, so 0.69 was **not** reused |
| Split | Held-out test |
| Status | **Smoke** |
| Source | `outputs/finetune_comparison/comparison.json` |

| Metric | Original | Fine-tuned | Δ |
|---|---:|---:|---:|
| Accuracy | 0.6000 | 0.6000 | +0.0000 |
| Balanced accuracy | 0.5000 | 0.5000 | +0.0000 |
| F1 | 0.7500 | 0.7500 | +0.0000 |
| Recall | 1.0000 | 1.0000 | +0.0000 |
| **AUROC** | 0.5104 | **0.3542** | **−0.1562** |
| FPR | 1.0000 | 1.0000 | +0.0000 |
| FNR | 0.0000 | 0.0000 | +0.0000 |

**Decision: the fine-tuned model was NOT adopted.** Ranking quality fell and
nothing improved. Both models collapse to predicting every image AI-generated at
threshold 0.5 on this 20-image set — which is what 68 training images should be
expected to produce. **These numbers demonstrate the pipeline works end to end;
they are not evidence about fine-tuning as a method.**

Integrity checks (all **Current**, all passed):

| Check | Result |
|---|---|
| Adapter save → reload determinism | **0.0** max score difference |
| Adapter changes predictions | Yes, max **0.2329** |
| Original checkpoint SHA-256 | **Unchanged** |
| Second LoRA adapter added | **No** — existing adapter tensors reused |

### Consistency-loss ablation

Same smoke dataset; all variants saw identical paired transformed views, so only
the loss differed. Source: `outputs/consistency_ablation/consistency_ablation.json`.

| Variant | F1 | Recall | FPR | AUROC | Runtime |
|---|---:|---:|---:|---:|---:|
| Classification only | 0.7500 | 1.0000 | 1.0000 | 0.4479 | 250.0 s |
| + logit-MSE consistency | 0.7500 | 1.0000 | 1.0000 | 0.4375 | 122.5 s |
| + symmetric-KL consistency | 0.7500 | 1.0000 | 1.0000 | 0.4479 | 113.5 s |

**Decision: keep classification-only.** Neither variant met the pre-registered
+0.01 gain. The loss code is retained and tested but **disabled by default**. At
68 training images this ablation cannot resolve a real effect.

---

## 7. Runtime

| | Value |
|---|---|
| Hardware | Apple Silicon, **MPS** backend |
| Checkpoint | Bombek1, 740,371,777 parameters |
| Status | **Current** |

| Configuration | Per image | Forward passes |
|---|---:|---:|
| Full protocol (clean + 14 transforms + 12 patches) | **13.34 s** | 27 |
| Whole-image only (batch CLI default, patches off) | **~1.14 s** | 1 |

Batch inference disables patch analysis by default, so submission runs pay one
forward pass per image plus the transformation passes only when enabled.

| Stage | Cost |
|---|---|
| Checkpoint load (once per process) | ~15 s |
| Streamlit cold start | 14.7 s |

---

## Test and quality status

| Check | Result |
|---|---|
| `pytest tests/ -q` | **509 passed**, 2 skipped, 350 subtests |
| `ruff check src scripts tests app.py` | All checks passed |
| `compileall src scripts app.py` | OK |
| Streamlit startup | OK, no exceptions or errors |
| Simple JSON contract | `{image_path, pred}` — unchanged |
