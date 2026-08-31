# RobustLens — all results

Every table below is copy-paste ready (GitHub-flavoured Markdown; Devpost accepts it).
All numbers are measured, not quoted. Reproduction commands are at the bottom.

**Reading rule:** always name the test set. The same system scores 1.000 accuracy on
one and 0.890 on another, and the difference is the data, not the model.

---

## 1. Model composition and competition compliance

Measured from the shipped checkpoint with
`python scripts/count_params.py --checkpoint models/pretrained/pytorch_model.pt`.

| Component | Parameters | Share | Trained here |
|---|---:|---:|---|
| SigLIP2 vision tower (`siglip2-so400m-patch14-384`) | 432,206,912 | 58.4% | No — frozen |
| DINOv2 (`vit_large_patch14_dinov2.lvd142m`) | 306,914,304 | 41.5% | No — frozen |
| Classifier head | 1,250,561 | 0.2% | **Yes** |
| **Total** | **740,371,777** | **100%** | 0.17% trained |
| *of which LoRA tensors inside the towers* | *7,127,040* | — | No |

| Limit | Used | Headroom | Verdict |
|---|---:|---:|---|
| 2,000,000,000 | 740,371,777 | 1,259,628,223 | **PASS — 37.0% of budget** |

---

## 2. Training dataset

4,979 images, 6 generators, 2 authentic sources. Every image normalised to
384×384 JPEG q95. 4,979 unique SHA-256, zero cross-split overlap.

| Source | Class | Dataset | Origin | Train | Val | Test | Total |
|---|---|---|---|---:|---:|---:|---:|
| `coco` | Real | Defactify | COCO **train2017** | 654 | 158 | 188 | 1,000 |
| `imagenet_real` | Real | GenImage | ImageNet | 892 | 202 | 271 | 1,365 |
| `sd21` | Fake | Defactify | Stable Diffusion 2.1 | 290 | 66 | 99 | 455 |
| `sdxl` | Fake | Defactify | SDXL | 301 | 74 | 80 | 455 |
| `sd3` | Fake | Defactify | Stable Diffusion 3 | 301 | 60 | 94 | 455 |
| `ADM` | Fake | GenImage | ADM (guided diffusion) | 295 | 65 | 95 | 455 |
| `glide` | Fake | GenImage | GLIDE | 291 | 72 | 92 | 455 |
| `Midjourney` | Fake | GenImage | Midjourney | 214 | 49 | 76 | 339 |
| **Total** | | | | **3,238** | **746** | **995** | **4,979** |

Class balance: 47.7% / 52.3% authentic-to-synthetic in train, similar in every split.

### Deliberate exclusions

| Excluded | Reason |
|---|---|
| DALL·E 3 | Keeps the reference benchmark a genuine unseen-generator test |
| Defactify class 5 | Card and paper order generators differently; dropped rather than risk it |
| BigGAN | 128×128 native with no paired reals — upscaling would add a new shortcut |
| CIFAKE | 32×32 against a 392×392 model input |
| COCO val2017 | Banned by the brief for training |

---

## 3. Confound audit

The raw sources were trivially separable before normalisation.

| Channel | Before | After |
|---|---|---|
| Aspect ratio | Fakes square, reals not — near-perfect leak | All 384×384 |
| Resolution | 128² – 1024² vs 500×375 / 640×480 | Uniform |
| Format / container | Mixed | All JPEG q95 RGB |
| Cross-split leakage | — | 0 of 4,979 |
| File size (residual) | — | **AUC 0.685** — reported, not eliminated |

### COCO provenance verification

Caption matching against the public COCO annotations.

| Check | Result |
|---|---:|
| Defactify real captions matching **train2017** | **997 / 997 (100%)** |
| Matching **val2017** | 13 (1.3%, all also in train2017) |
| Matching neither | 0 |

**Conclusion:** training reals are COCO train2017; the banned val2017 split is untouched.

---

## 4. The failure this project is built on

Base model, multi-generator test set, n=200, threshold frozen at 0.6804.

| Condition | AUROC | Accuracy | TPR | FPR | ΔAUROC |
|---|---:|---:|---:|---:|---:|
| clean | 0.962 | 0.930 | 0.890 | 0.030 | — |
| color_jitter | 0.959 | 0.910 | 0.860 | 0.040 | −0.003 |
| blur σ0.5 | 0.958 | 0.910 | 0.910 | 0.090 | −0.004 |
| center_crop 80% | 0.958 | 0.895 | 0.810 | 0.020 | −0.004 |
| JPEG q90 | 0.957 | 0.895 | 0.830 | 0.040 | −0.005 |
| resize 0.5× | 0.954 | 0.880 | 0.810 | 0.050 | −0.008 |
| JPEG q70 | 0.941 | 0.860 | 0.750 | 0.030 | −0.021 |
| resize 0.25× | 0.931 | 0.800 | 0.600 | 0.000 | −0.031 |
| noise σ0.02 | 0.929 | 0.835 | 0.860 | 0.190 | −0.032 |
| JPEG q50 | 0.924 | 0.795 | 0.600 | 0.010 | −0.038 |
| blur σ2.0 | 0.922 | 0.825 | 0.670 | 0.020 | −0.040 |
| noise σ0.05 | 0.911 | 0.810 | 0.750 | 0.130 | −0.051 |
| JPEG q30 | 0.902 | 0.775 | **0.560** | 0.010 | −0.060 |
| noise σ0.10 | 0.861 | 0.770 | 0.570 | 0.030 | **−0.101** |

| Metric | Clean | Transformed mean | Change |
|---|---:|---:|---:|
| AUROC | 0.962 | 0.933 | **−0.029** |
| TPR | 0.890 | 0.744 | **−0.146** |

**The finding:** ranking survives, the threshold does not. Scores slide downward, so a
compressed fake is reported as confidently authentic rather than uncertain.

---

## 5. Fine-tune result — multi-generator set

n=200 held out. Each model at its **own** threshold, fitted on a separate 180-image
validation split and frozen across all 15 conditions. Original t=0.7718, tuned t=0.8103.

| Condition | Acc base | Acc tuned | Δ |
|---|---:|---:|---:|
| clean | 0.915 | 0.930 | +0.015 |
| resize 0.25× | 0.755 | 0.905 | **+0.150** |
| JPEG q30 | 0.735 | 0.855 | **+0.120** |
| JPEG q50 | 0.770 | 0.885 | **+0.115** |
| JPEG q70 | 0.840 | 0.925 | +0.085 |
| blur σ2.0 | 0.805 | 0.880 | +0.075 |
| noise σ0.10 | 0.710 | 0.775 | +0.065 |
| JPEG q90 | 0.895 | 0.950 | +0.055 |
| resize 0.5× | 0.880 | 0.935 | +0.055 |
| blur σ1.0 | 0.880 | 0.920 | +0.040 |
| center_crop 80% | 0.900 | 0.940 | +0.040 |
| color_jitter | 0.910 | 0.940 | +0.030 |
| noise σ0.05 | 0.815 | 0.845 | +0.030 |
| blur σ0.5 | 0.900 | 0.925 | +0.025 |
| noise σ0.02 | 0.860 | 0.845 | **−0.015** |

| Summary | Base | Tuned | Δ |
|---|---:|---:|---:|
| Transformed accuracy (mean) | 0.832 | 0.895 | **+0.062** |
| Transformed AUROC (mean) | 0.933 | 0.969 | **+0.036** |
| Worst-case accuracy | 0.710 | 0.775 | +0.065 |
| Mean FPR | 0.033 | 0.132 | **+0.099** |
| Conditions improved | — | — | **13 / 14** |

### Verified three independent ways

| Check | Rules out | Result |
|---|---|---|
| AUROC (threshold-free) | Credit for a threshold shift | **+0.045**, 14/14 |
| TPR at matched FPR | Buying recall with false alarms | **+0.058**, 13/14 |
| Scene-disjoint subset (n=180) | Memorising training scenes | **+0.045**, larger than contaminated |

Paired bootstrap, 2,000 resamples: **+0.0363, 95% CI [+0.0133, +0.0627]** — excludes zero.

---

## 6. Reference benchmark — COCO val2017 + DALL·E 3

`techjam-aigc/wildfake-eval-subset`, **`normalized`** config (both classes 200×200).
Balanced sample n=200. Threshold 0.1127.

### Base model

| Condition | AUROC | Accuracy | TPR | FPR |
|---|---:|---:|---:|---:|
| clean | 0.999 | 0.985 | 1.000 | 0.030 |
| JPEG q90 | 0.999 | 0.985 | 0.990 | 0.020 |
| color_jitter | 0.999 | 0.980 | 0.980 | 0.020 |
| resize 0.5× | 0.998 | 0.980 | 0.980 | 0.020 |
| blur σ0.5 | 0.998 | 0.960 | 0.990 | 0.070 |
| center_crop 80% | 0.997 | 0.975 | 0.960 | 0.010 |
| JPEG q70 | 0.996 | 0.965 | 0.960 | 0.030 |
| noise σ0.02 | 0.996 | 0.950 | 0.980 | 0.080 |
| JPEG q50 | 0.995 | 0.960 | 0.920 | 0.000 |
| noise σ0.05 | 0.991 | 0.950 | 0.940 | 0.040 |
| JPEG q30 | 0.989 | 0.945 | 0.920 | 0.030 |
| blur σ1.0 | 0.989 | 0.940 | 0.980 | 0.100 |
| noise σ0.10 | 0.974 | 0.910 | 0.870 | 0.050 |
| resize 0.25× | 0.969 | 0.880 | 0.760 | 0.000 |
| blur σ2.0 | 0.945 | 0.850 | 0.910 | 0.210 |

DALL·E 3 clean recall: **1.000**. COCO val2017 clean FPR: 0.030.

### Adapter vs base

| Summary | Base | Tuned | Δ |
|---|---:|---:|---:|
| Transformed AUROC (mean) | 0.988 | 0.977 | **−0.011** |
| Transformed accuracy (mean) | 0.945 | 0.927 | −0.018 |
| Worst-case accuracy | 0.850 | 0.790 | −0.060 |
| Conditions improved (AUROC) | — | — | **3 / 14** |

Paired bootstrap: **−0.0108, 95% CI [−0.0204, −0.0027]** — excludes zero.
**Verdict: do not adopt.**

---

## 7. `laion_matched` — the harder configuration

3,826 LAION-5B reals + 3,826 DALL·E 3, both natively ≥1024px, served at 512×512.
Balanced sample n=198. Threshold 0.9518.

### Base model — the failure mode is present here

| Condition | AUROC | Accuracy | TPR |
|---|---:|---:|---:|
| clean | 0.996 | 0.975 | **1.000** |
| center_crop 80% | 0.999 | 0.934 | 0.867 |
| JPEG q50 | 0.997 | 0.934 | 0.867 |
| resize 0.25× | 0.997 | 0.838 | **0.673** |
| JPEG q70 | 0.997 | 0.949 | 0.908 |
| noise σ0.02 | 0.997 | 0.960 | 0.980 |
| blur σ1.0 | 0.997 | 0.955 | 0.929 |
| blur σ0.5 | 0.997 | 0.970 | 0.990 |
| JPEG q90 | 0.996 | 0.965 | 0.980 |
| resize 0.5× | 0.995 | 0.949 | 0.908 |
| color_jitter | 0.995 | 0.955 | 0.980 |
| blur σ2.0 | 0.994 | 0.904 | 0.806 |
| JPEG q30 | 0.994 | 0.914 | 0.837 |
| noise σ0.05 | 0.993 | 0.939 | 0.898 |
| noise σ0.10 | 0.976 | 0.833 | **0.673** |

AUROC holds at ~0.99 while recall falls to 0.673 — the same failure as §4.

### Adapter vs base — a threshold shift, not an improvement

| Metric | Result | Interpretation |
|---|---|---|
| TPR at fixed threshold | **+0.114 mean, 14/14** | Looks like a large win |
| AUROC | **−0.006 mean, 0/14** | Separation got *worse* |
| FPR | +0.080 to +0.270 | The recall was bought |
| **TPR at matched FPR (0.05)** | **−0.028 mean, 0/14 improved, 13 worse** | **Base model wins** |

**Verdict: do not adopt.** Every recall gain is obtainable by lowering the threshold on
the base model, which yields more recall for the same false-positive rate.

---

## 8. Abstention — the result that transfers

Selective prediction. The system withdraws a verdict when the 14 transformed versions
disagree. Base model, no adapter.

| Test set | n | Abstains | Accuracy overall | Accuracy among answered | Error enrichment |
|---|---:|---:|---:|---:|---:|
| **Reference benchmark** | 200 | 45.5% | 0.985 | **1.000** | **2.20×** |
| **`laion_matched`** | 198 | 38.9% | 0.975 | **1.000** | **2.57×** |
| Multi-generator (ours) | 200 | 55.0% | 0.890 | **0.978** | 1.65× |

Pre-registered bar for error enrichment is 1.5× — cleared on all three, and highest on
the two benchmarks we did not build.

**Rules fired** (reference benchmark): `low_transformation_consistency` 91,
`low_agreement_between_versions` 19, `borderline_probability` 2.

**Caveat to state:** the abstention rate breaches the repo's own pre-registered 35%
ceiling. The honest framing is *"answers 54.5% of images with 100% accuracy"*, which
suits triage — auto-clear the confident, route the rest to review — not a system that
must decide on everything.

---

## 9. Summary scoreboard

| Contribution | Multi-generator | Reference benchmark | `laion_matched` |
|---|---|---|---|
| Fine-tuned adapter | ✅ +0.036 AUROC, 14/14 | ❌ −0.011, 3/14 | ❌ −0.028 at matched FPR |
| **Abstention layer** | ✅ 0.890 → 0.978 | ✅ **0.985 → 1.000** | ✅ **0.975 → 1.000** |

**The fine-tune does not generalise. The abstention layer does.**

---

## 10. Reproduce

```bash
python3 scripts/setup.py
python3 scripts/count_params.py --checkpoint models/pretrained/pytorch_model.pt
python3 scripts/build_training_mix.py
python3 scripts/build_augmented_train.py --views 3 --limit 2000
python3 scripts/train_local_edit_lora.py --config configs/robustness_head.yaml --mode head_only --device mps
python3 scripts/run_inference_chunked.py --input-dir IMAGES --detailed-output out.json --device mps
python3 scripts/robustness_table.py --detailed out.json --labels labels.json
python3 scripts/compare_robustness.py --baseline base.json --candidate tuned.json --labels labels.json
```

## Caveats that belong with any number above

- **n≈200 per evaluation.** Aggregate claims are supported; individual per-condition
  figures have 95% CIs around ±0.07 and should be read as "where gains concentrate",
  not as fourteen separate measurements.
- **Multi-generator results are in-distribution** — same generators, sources and
  normalisation as training.
- **52% of Defactify scenes straddle splits** (fakes are generated from real images'
  captions). Re-scoring on scene-disjoint images gave a *larger* gain, so the
  conclusion holds, but the splitter should group by scene.
- **The detector is not ours.** 0.17% of parameters were trained here.
