# BDC Disaster Dataset Deep EDA & Forensic Analysis

This report is based on the complete dataset present in the working directory: every image was inventoried and decoded; all files were cryptographically hashed; image geometry, quality, format, and selected metadata were measured; exact and perceptual duplicate graphs were built; evidence-selected samples were inspected visually; and frozen-representation diagnostics were run. No image was modified, moved, deleted, or relabeled, and no TEST target was inferred.

Throughout this report:

- **MEASURED** means directly computed from or observed in this dataset.
- **INTERPRETATION** means a plausible explanation of measured evidence.
- **HYPOTHESIS** means a proposition that still requires a controlled modeling experiment.

## 1. Executive Summary

The dataset contains **17,932 readable images**: 17,482 TRAIN and 450 TEST, occupying 9.673 GB (9.01 GiB), plus `TRAIN/Solution.csv`. The actual TRAIN target space is the expected 3 disasters × 3 severity levels = 9 joint combinations. `Solution.csv`, however, asks for **two predictions per TEST image**—one disaster type and one severity—not one joint-class row.

The highest-value findings are:

1. **MEASURED — disaster is easy; severity is the real problem.** A scene-aware frozen SigLIP2 probe reached 99.15% disaster accuracy but only 72.00% global severity accuracy. Conditional severity probe accuracy varied enormously: 58.78% for BANJIR, 75.67% for GEMPA BUMI, and 92.00% for KEBAKARAN. Visual review agrees: flood severity is poorly defined by visible damage, quake `SEDANG` and `BERAT` overlap strongly, while fire has the clearest severity progression.
2. **MEASURED — duplicate/scene leakage is substantial.** SHA-256 found 1,292 exact-duplicate groups involving 2,837 images (15.82% of all images). A conservative perceptual graph involved 3,774 images (21.05%), with 357 cross-label pairs and 23 TRAIN–TEST pairs. There are 105 exact TRAIN duplicate groups spanning different severity labels. Ordinary random image-level validation would therefore be optimistic.
3. **MEASURED — source and geometry artifacts are unusually label-correlated.** Disaster × resolution-bucket Cramér's V is 0.884, joint label × resolution bucket is 0.639, and joint label × extension is 0.413. For example, 1,631 of 1,746 `KEBAKARAN/RINGAN` images are 400×400. A Xiaomi device family contributes 291 files, 246 of them `GEMPA BUMI/RINGAN`. These signals diagnose collection bias; they should not become deliberately exploited features.
4. **MEASURED — TRAIN and TEST differ, but not as wholly separate domains.** TEST has larger median pixel count, lower compressed bits per pixel, a different orientation mixture, and is distinguishable from TRAIN in frozen embedding space (grouped domain-probe AUC 0.865 with SigLIP2). Nonetheless, systematic visual inspection found the same broad phone, news, aerial, documentation, screenshot, and professional-photo families in both splits. The evidence supports **moderate source-mixture shift**, not a completely unsupported TEST domain.
5. **MEASURED — naive square stretching is unjustified.** Aspect ratio ranges from 0.3005 to 6.7767; TRAIN contains 3,909 unique resolutions, and exact-square images are themselves label/source concentrated. Severity often depends on spatial extent and context. Use EXIF-aware RGB decoding and aspect-ratio-preserving resize/pad or shape bucketing; test 224 versus 384 before spending on higher resolution.
6. **MEASURED — SigLIP2 is a strong primary representation.** Frozen `google/siglip2-base-patch16-224` achieved scene-excluded 5-NN agreement of 99.19% for disaster, 78.48% for severity, and 78.11% for the 9-way joint label on a balanced 2,700-image TRAIN sample. Its severity structure is weakest for flood and strongest for fire, matching visual findings.
7. **MEASURED LIMITATION — DINOv3 was not available locally and its official checkpoints are gated.** It was not evaluated and no DINOv3 result is fabricated. A same-sample frozen DINOv2-base baseline was measured only as evidence about the self-supervised representation direction: it was slightly weaker on disaster and joint 5-NN, slightly stronger on the global linear severity probe and fire probe, and made meaningfully different errors. This supports testing a computationally sensible DINOv3-small/base checkpoint later, not jumping to 7B.
8. **INTERPRETATION — severity is not one clean global ordinal axis.** Ordinality is unsupported for BANJIR, partially supported for GEMPA BUMI, and best supported for KEBAKARAN. A disaster-conditioned severity design is better motivated than a single global ordinal objective.
9. **RECOMMENDED DECISION — validate with five scene-aware folds.** Use a fixed exact+strict-near component ID as the group and joint label as the stratification target. Report the official pooled row-level Micro F1 (which equals row accuracy under the observed schema), plus disaster accuracy, severity accuracy, joint exact-match accuracy, per-class F1, and three confusion views.
10. **RECOMMENDED MODELING DIRECTION — start simple and aligned with the target.** Prioritize SigLIP2 with shared features and separate disaster/severity heads. Then test disaster-specific severity heads or a small auxiliary 9-class head. Do not start with class balancing, global ordinal loss, aggressive crops, or DINOv3-7B.

## 2. Competition Target and Metric Interpretation

### Actual schema

**MEASURED.** `TRAIN/Solution.csv` uses semicolon separation and has columns `ID;Target`. It contains 900 blank target rows in this order:

- `1_jenis`, `1_kerusakan`, …
- through `450_jenis`, `450_kerusakan`.

This is **separate-output encoding**: one disaster-type output (`jenis`) and one severity output (`kerusakan`) per TEST image. It is not a conventional single column containing one of nine joint classes. The directory labels nevertheless define exactly nine TRAIN combinations; there is no filesystem or target-schema evidence for 12 classes.

### What Micro F1 means here

For conventional single-label prediction on each of the 900 rows, pooled Micro F1 is mathematically equal to row-level accuracy:

`Micro F1 = correct target rows / 900`

Because the schema contains exactly one disaster row and one severity row for each of 450 images:

`row accuracy = (disaster accuracy + severity accuracy) / 2`.

This is **not** the same as per-image 9-class joint exact-match accuracy: an image with correct disaster but wrong severity receives one correct row rather than zero correct images. The scorer implementation was not present, so this conclusion is conditional on the official evaluator applying ordinary single-label Micro F1 to the `Solution.csv` rows—the schema strongly supports that reading.

**Implication.** The official objective weights aggregate disaster and severity decisions equally, despite disaster being much easier. Joint accuracy remains a valuable diagnostic and an optional auxiliary objective, but it should not replace the actual row-level metric during model selection.

## 3. Dataset Inventory and Health

### Verified size and split inventory

| Item | Count / size |
| --- | ---: |
| TRAIN images | 17,482 |
| TEST images | 450 |
| All images | 17,932 |
| Additional data file | 1 (`TRAIN/Solution.csv`) |
| All regular files | 17,933 |
| Total bytes | 9,672,740,884 B |
| Total binary size | 9.01 GiB |
| TRAIN directory including CSV | 9,494,498,480 B |
| TEST directory | 178,242,404 B |

### Decoding and format integrity

**MEASURED.** Every one of the 17,932 images decoded successfully. There are **0 corrupt images, 0 unreadable images, and 0 zero-byte files**.

| Detected content format | Images | Share |
| --- | ---: | ---: |
| JPEG | 12,119 | 67.58% |
| PNG | 5,800 | 32.34% |
| WEBP | 11 | 0.06% |
| MPO | 2 | 0.01% |

Filename extensions are `.jpg` 10,974, `.png` 5,776, `.jpeg` 1,122, and `.jfif` 60. There are 99 extension/content-format mismatches. Sixty are the benign `.jfif`→JPEG naming case; the remaining 39 are materially misleading: 25 `.jpg` files contain PNG, 11 `.jpg` files contain WEBP, two `.jpg` files contain MPO, and one `.png` contains JPEG.

| Decoded image mode | Images |
| --- | ---: |
| RGB | 17,104 |
| RGBA | 825 |
| P | 1 |
| CMYK | 1 |
| L | 1 |

**Finding → Evidence → Implication → Recommended Action.** Suffix-based decoders are unsafe because 39 filenames materially disagree with their contents, while robust content-aware decoding succeeded on all files. Use a library decoder that identifies content, apply EXIF orientation, explicitly convert/composite all modes to RGB, and log rather than silently skip future decode failures. No present image needs deletion.

## 4. Class Distribution and Imbalance

### Joint 9-class distribution

| Disaster | Severity | Count | TRAIN share |
| --- | --- | ---: | ---: |
| BANJIR | KERUSAKAN BERAT | 1,968 | 11.26% |
| BANJIR | KERUSAKAN RINGAN | 2,018 | 11.54% |
| BANJIR | KERUSAKAN SEDANG | 1,971 | 11.27% |
| GEMPA BUMI | KERUSAKAN BERAT | 1,623 | 9.28% |
| GEMPA BUMI | KERUSAKAN RINGAN | 1,393 | 7.97% |
| GEMPA BUMI | KERUSAKAN SEDANG | 2,730 | 15.62% |
| KEBAKARAN | KERUSAKAN BERAT | 2,025 | 11.58% |
| KEBAKARAN | KERUSAKAN RINGAN | 1,746 | 9.99% |
| KEBAKARAN | KERUSAKAN SEDANG | 2,008 | 11.49% |

The smallest joint class is `GEMPA BUMI/RINGAN` (1,393); the largest is `GEMPA BUMI/SEDANG` (2,730), a maximum/minimum ratio of **1.960**.

### Marginals

| Disaster | Count | Share |
| --- | ---: | ---: |
| BANJIR | 5,957 | 34.08% |
| GEMPA BUMI | 5,746 | 32.87% |
| KEBAKARAN | 5,779 | 33.06% |

| Severity | Count | Share |
| --- | ---: | ---: |
| KERUSAKAN BERAT | 5,616 | 32.12% |
| KERUSAKAN RINGAN | 5,157 | 29.50% |
| KERUSAKAN SEDANG | 6,709 | 38.38% |

Disaster marginals are essentially balanced. Severity has a 1.301 largest/smallest ratio. The localized joint imbalance is best described as **mild to moderate**, not severe. Disaster and severity association is weak overall (Cramér's V 0.0965), though GEMPA contributes the joint extremes.

**Finding → Evidence → Implication → Recommended Action.** Unequal class counts exist, but the official Micro-F1/accuracy objective rewards every target row equally and the imbalance is modest. Begin with ordinary cross entropy and natural-frequency sampling. Do not default to class weights, focal loss, over/undersampling, or a balanced sampler. Assess them only if scene-aware out-of-fold results show a repeatable gain in official row accuracy rather than merely improved macro diagnostics.

## 5. Image Geometry

### Overall geometry

| Statistic | TRAIN | TEST |
| --- | ---: | ---: |
| Median width × height | 788.5 × 512 | 800 × 600 |
| Median pixel count | 360,000 | 480,000 |
| Pixel-count p05 / p95 | 62,500 / 2,464,000 | 62,500 / 4,557,997.5 |
| Median file size | 173,747 B | 123,177.5 B |
| Median compressed bits/pixel | 3.323 | 1.981 |
| Aspect-ratio p05 / p95 | 0.750 / 1.782 | 0.731 / 1.779 |
| Unique resolutions | 3,909 | 201 |

The complete aspect-ratio range is **0.3005 to 6.7767**. The smallest image is `TEST/159.jpeg` at 127×113 (14,351 pixels). The largest is `TRAIN/GEMPA BUMI/KERUSAKAN RINGAN/921.jpg` at 9232×5588 (51,588,416 pixels).

| Exact orientation (`width` vs `height`) | TRAIN | TEST |
| --- | ---: | ---: |
| Landscape | 10,618 (60.74%) | 319 (70.89%) |
| Portrait | 1,498 (8.57%) | 88 (19.56%) |
| Square | 5,366 (30.69%) | 43 (9.56%) |

TRAIN's leading resolution modes are 400×400 (2,071), 800×450 (1,628), 512×512 (1,383), 1280×720 (1,352), 256×256 (903), and 250×250 (865). TEST's are 800×600 (94), 250×250 (41), 1024×768 (28), 1024×683 (21), and 3060×4080 (18).

### Geometry is correlated with collection and label

| Association | Cramér's V |
| --- | ---: |
| Disaster × resolution bucket | 0.8835 |
| Joint label × resolution bucket | 0.6387 |
| Severity × resolution bucket | 0.5387 |
| Split × resolution bucket | 0.3477 |
| Joint label × extension | 0.4128 |
| Disaster × extension | 0.3182 |
| Severity × extension | 0.2994 |
| Split × extension | 0.0885 |

The strongest examples are source-template concentrations: 1,631/1,746 `KEBAKARAN/RINGAN` images are 400×400; `KEBAKARAN/SEDANG` has 1,761/2,008 images at either 256×256 or 512×512; and `KEBAKARAN/BERAT` contains 865 images at 250×250 and 525 at 512×512. BANJIR is overwhelmingly landscape.

**INTERPRETATION.** Resolution is partly a proxy for how each class was collected, edited, or scraped rather than a causal severity feature. A random split will preserve these shortcuts and may make validation look better than deployment on a shifted source mixture.

**Finding → Evidence → Implication → Recommended Action.** Geometry is diverse, class-correlated, and sometimes essential to judging affected extent. Do not stretch every image into a square and do not use aggressive random crops. Preserve aspect ratio with padding or shape buckets; compare moderate 224 and 384 input scales; and evaluate a native/flexible-aspect SigLIP2/NaFlex direction only after the fixed-resolution baseline. Multi-resolution training above 384 is a later experiment, not an EDA-derived requirement.

## 6. Image Quality

Quality features were computed from consistently decoded RGB thumbnails, with Lanczos downsampling constrained to 256 pixels. They are useful comparative proxies, not camera-calibrated physical measurements.

| Median property | TRAIN | TEST |
| --- | ---: | ---: |
| Luminance | 0.4660 | 0.4454 |
| Luminance contrast (SD) | 0.2085 | 0.2134 |
| Saturation | 0.2236 | 0.2235 |
| Entropy | 7.403 bits | 7.361 bits |
| Underexposed-pixel fraction | 1.02% | 0.82% |
| Overexposed-pixel fraction | 0.59% | 0.81% |
| Laplacian sharpness proxy | 0.01078 | 0.00942 |
| High-frequency/noise proxy | 0.03318 | 0.03103 |

The marginal TRAIN–TEST quality shift is small to modest: TEST is slightly darker and softer, while saturation is essentially unchanged. Compression/source differences are clearer: TEST has lower median compressed bits per pixel despite larger median pixel count.

Class-conditional regimes are much stronger:

- `BANJIR/RINGAN` mean luminance is 0.518 with saturation 0.119 and median 14.15 compressed bpp; `BANJIR/BERAT` is darker (0.455) and much more compressed (6.44 bpp).
- `GEMPA BUMI/RINGAN` mean luminance is 0.538 versus 0.449 for `SEDANG` and 0.465 for `BERAT`.
- `KEBAKARAN/RINGAN` is exceptionally dark (mean luminance 0.324), underexposed (mean fraction 0.183), low-entropy (6.455), and soft (median sharpness 0.00253). `KEBAKARAN/BERAT` is brighter (0.377) and more saturated (0.465); `SEDANG` is brighter still (0.429) but less saturated (0.316).

**INTERPRETATION.** Much of this apparent class signal is entangled with source genre: controlled-fire imagery, low-resolution fire collections, quake phone photos, flood webcams, and professionally edited/news imagery. Brightness or sharpness differences should not be treated as definitions of severity.

**Recommended action.** Use backbone-specific normalization after RGB conversion. Test only mild brightness/contrast and JPEG perturbation to improve robustness to the measured regimes. Strong blur, noise, or color transforms would amplify artifacts that already exist and risk erasing damage cues.

## 7. Source Domains and Domain Shift

### Metadata evidence

Only 893 images (4.98%) expose at least one selected EXIF field, so metadata cannot fully enumerate domains. Among those files, Orientation appears in 869, DateTime in 849, Software in 634, and Make/Model in 440. There are 848 parseable timestamps ranging from 2005-08-15 to 2023-11-05.

Several source fingerprints are strongly label-concentrated:

- Xiaomi `M2102K1G`: 291 files—246 `GEMPA BUMI/RINGAN`, 17 `GEMPA BUMI/SEDANG`, three `GEMPA BUMI/BERAT`, and 25 TEST.
- Windows Photo Editor: 293 files—112 `GEMPA BUMI/RINGAN`, 104 `SEDANG`, 44 `BERAT`, and 33 TEST.
- GIMP 2.10.32: 186 files—69 `BANJIR/SEDANG`, 68 `BANJIR/BERAT`, 31 `BANJIR/RINGAN`, and 18 TEST.

Available devices include phones, Nikon/Canon cameras, and DJI equipment.

### Visual and embedding evidence

Systematic visual inspection identified multiple real source families rather than one homogeneous photographic domain:

- BANJIR: marked webcams, river monitoring, rescue/field documentation, phone/news photos, urban and rural inundation, aerial/drone views, and satellite-like imagery.
- GEMPA BUMI: phone/street documentation, close-up cracks, drone collapse views, stock/news watermarks, collages/infographics/screenshots, and some illustration-like imagery.
- KEBAKARAN: phone/news images, professional wildfire photography, surveillance-like imagery, controlled-fire photos, burned aftermath, and already-rotated/noisy edits.
- TEST contains the same broad families, though not in the same proportions.

A grouped frozen-embedding split probe distinguishes TRAIN from TEST with ROC AUC 0.865 for SigLIP2 and 0.848 for the DINOv2 auxiliary baseline. This cannot be explained by a single low-level statistic alone.

**INTERPRETATION.** Domain shift is visible and measurable, especially as a change in source mixture and content framing. It is not evidence that all TEST images lie outside TRAIN support. Source-label correlation can inflate a random validation score because the same device/template/collection style appears on both sides.

**Recommended action.** Diagnose performance by source proxies and geometry buckets, but never feed filename, file-size, EXIF device, editor, or timestamp deliberately as competition features. Scene grouping is essential; aggressive source grouping is a sensitivity analysis because over-grouping could create folds unlike TEST.

## 8. Exact Duplicates

All 17,932 images were hashed with SHA-256. Decoded-pixel hashes agreed with byte-hash grouping for this dataset.

| Exact-duplicate result | Measured value |
| --- | ---: |
| SHA-256 duplicate groups | 1,292 |
| Images in duplicate groups | 2,837 (15.82%) |
| Redundant copies beyond one/group | 1,545 |
| Duplicate pairs | 1,882 |
| Groups confined to one observed split/label category | 1,182 |
| Groups crossing an observed TRAIN label or split category | 110 |
| Largest group sizes | 7, 7, 7, 6, 6, 6, 5, 5, 4, 4 |
| TRAIN groups crossing severity | 105 |
| Image occurrences in those conflict groups | 223 |
| Exact TRAIN–TEST overlap groups | 5 |
| Images in those overlap groups | 10 |

The five exact TRAIN–TEST overlaps are:

- `TEST/128.jpg` ↔ `TRAIN/BANJIR/KERUSAKAN SEDANG/332.jpg`
- `TEST/137.jpg` ↔ `TRAIN/BANJIR/KERUSAKAN SEDANG/1743.jpg`
- `TEST/152.jpeg` ↔ `TRAIN/GEMPA BUMI/KERUSAKAN RINGAN/601.jpeg`
- `TEST/179.jpeg` ↔ `TRAIN/GEMPA BUMI/KERUSAKAN BERAT/439.jpeg`
- `TEST/35.jpg` ↔ `TRAIN/BANJIR/KERUSAKAN BERAT/727.jpg`

Most cross-label exact conflicts are within quake severity: 54 `BERAT`↔`SEDANG`, 34 `RINGAN`↔`SEDANG`, 12 spanning all three severities, and three `BERAT`↔`RINGAN` groups in the reviewed breakdown. One exact group crosses disaster as well: `BANJIR/RINGAN/220.jpg` equals `GEMPA BUMI/SEDANG/1204.jpg`, an overlaid aerial “worst earthquakes” scene. A severe-collapse image is identically present at `GEMPA BUMI/BERAT/1335.jpg`, `RINGAN/1120.jpg`, and `SEDANG/2564.jpg`. Fire also has a direct conflict: `KEBAKARAN/BERAT/1993.jpg` equals `KEBAKARAN/SEDANG/1947.jpg`.

**INTERPRETATION.** Exact conflicting images prove annotation inconsistency or an image-level label definition that cannot be recovered from pixels alone. They do not identify which label is correct.

**Recommended action.** Keep all members of an exact group in one fold. Flag conflict groups for audit and sensitivity analysis; do not automatically relabel or delete them. Do not use the five TEST overlaps to claim generalization or to infer other TEST labels.

## 9. Near-Duplicates and Repeated Scenes

The scalable screen used 64-bit perceptual hash, difference hash, and average hash, with shared 16-bit hash-band candidate generation rather than an all-pairs 161-million comparison. A conservative edge required `pHash ≤ 1`, `dHash ≤ 2`, and `aHash ≤ 4`; connected components define strict repeated-image groups.

| Perceptual result | Pairs | Components | Involved images |
| --- | ---: | ---: | ---: |
| Identical triple-hash signature | 2,313 | 1,463 | 3,263 |
| Strict near-duplicate graph | 3,974 | 1,550 | 3,774 (21.05%) |

The strict graph contains 357 cross-label pairs and 23 TRAIN–TEST pairs. Its largest component sizes are 41, 21, 18, 17, 14, 14, 14, 13, 12, and 12. It captures resized/recompressed images, PNG/JPEG resaves, marked webcam repetitions, image bursts, and visually repeated scenes.

**Manual verification.** The 24 highest-risk and 24 random strict pairs were visually reviewed; all 48/48 were genuine identical or repeated-scene variants. With zero observed false positives, a one-sided 95% upper bound on the false-positive rate is approximately 6%. This establishes high precision, not perfect recall. Crops, screenshots, stronger transforms, sequence frames, and different viewpoints can be missed when they share no hash band.

**INTERPRETATION.** At least one fifth of the dataset participates in very conservative repeated-image structure. This is a lower bound on broader event/scene repetition, but it would be incorrect to call all 3,774 images redundant or all connected components single real-world events.

**Recommended action.** Use exact+strict-near components as the default validation group. Treat broader embedding scene clusters as an optional sensitivity split, with manual checks, rather than merging aggressively by default.

## 10. Leakage and Validation Risk

An ordinary stratified random split is not trustworthy as the primary estimate:

- 15.82% of all images are in exact duplicate groups.
- 21.05% are in conservative perceptual components.
- Repeated webcams, resaves, and bursts can place nearly the same visual evidence in training and validation.
- 105 exact TRAIN groups contain conflicting severity labels; even a group kept within one class can still leak source identity.
- Strong resolution/device/editor correlations make same-source random folds easier than a shifted TEST mixture.

On the balanced frozen SigLIP2 sample, the observed numerical inflation from non-grouped diagnostics was small: random versus scene-aware 5-NN was 78.70% versus 78.48% for severity and 78.33% versus 78.11% for joint labels; grouped versus random probes differed by at most 0.22 percentage point. This does **not** prove random full-data validation is safe: only 61 of the 2,700 sampled TRAIN paths merged into another sampled strict component, frozen features cannot memorize fine-tuning examples, and the exhaustive full graph contains much more repeated structure. The hard overlap counts establish risk; the size of final-model optimism remains a modeling-stage measurement.

**Tradeoff.** Grouping every broad embedding cluster would reduce leakage further, but may conflate semantically common scenes—such as unrelated building collapses or fire photographs—and distort folds away from TEST. Exact and strict perceptual groups have much stronger evidence of shared origin.

**Concrete recommendation.** Use five-fold joint-label `StratifiedGroupKFold`, where the group is the connected component of exact and strict-near edges. Singletons each receive their own group. Keep every conflicting component intact. Then run one sensitivity audit on a manually validated broader scene grouping or source-stratified slice. Section 22 specifies the full protocol.

## 11. Filename and Metadata Bias

**MEASURED.** Every image stem is numeric, and numbering restarts within class folders. Filename alone therefore exposes folder-construction/order information if a pipeline retains the original path. Extensions, resolutions, editors, and devices are label-associated, as quantified in Sections 5 and 7. Ninety-nine extension/content pairs disagree, further showing that suffix is a collection artifact rather than reliable image semantics.

No single EXIF field covers enough images to explain the dataset, but the concentrated device/editor examples are too strong to ignore as a validation diagnostic. Timestamp coverage is sparse and spans 18 years; it should not be treated as a stable temporal domain variable.

**INTERPRETATION.** Numeric ranges and encoding choices may identify a scraped batch or augmentation/export run. A model can also absorb these artifacts visually through fixed resolution, borders, watermarks, compression, or editing signatures even when explicit metadata is removed.

**Recommended action.** Build manifests with opaque IDs for training; exclude path, stem, extension, file size, EXIF, timestamp, device, and software from model inputs. After each fold is created, audit distributions of numeric-stem ranges, resolution modes, extension, and known EXIF sources. These checks diagnose biased validation; they are not features to exploit.

## 12. Visual Semantics of Severity

### Inspection design

**MEASURED.** Visual analysis covered 246 unique images (1.37% of the full set): 196 core samples selected by fixed-seed class sampling plus geometry/aspect extremes and 19 TEST examples, 50 additional filename-stratified samples, and four high-risk conflict cases. All nine class contact sheets, TEST representatives, exact-conflict sheets, strict-near sheets, embedding disagreements, and high-distance TEST cases were reviewed. Selection targeted representatives, extremes, conflicts, and hard cases rather than pretending a small random gallery described every image.

The counts below describe only reviewed images and must not be extrapolated as exact full-dataset prevalence. Frozen-representation results in Sections 13–14 supply broader quantitative evidence.

### BANJIR

#### RINGAN versus SEDANG

Of 20 core `RINGAN` images, roughly nine showed clear inundation/rescue, nine showed water or high-water context without visible damage, and two were weak or uninterpretable. Some `RINGAN` examples (`2018`, `1930`) visually show severe aerial flooding. Among 19 `SEDANG` images, about five showed broad inundation, seven were river/water context without structural damage, and seven appeared dry, off-topic, or non-flood. Conversely, `SEDANG/285`, `144`, `273`, `259`, and `228` show severe city flooding.

**INTERPRETATION.** The label does not consistently encode water extent, water depth, visible structural damage, or rescue intensity. In many frames, scale/reference information needed to estimate depth is absent.

#### SEDANG versus BERAT

Among 20 `BERAT` images, about nine showed clearly affected settlements/people, ten showed river/high-water scenes without clear damage, and one showed no visible flood. Convincing severe examples include `121`, `352`, `845`, and `1883`; weak examples include marked river/webcam or dry-road images such as `1476.png`, `1940.png`, and `493.jpg`. Marked webcam families recur across severities—for example, `BERAT/1940.png`, `RINGAN/1048.png`, and `SEDANG/1896.png`—making source identity easier than severity.

#### RINGAN versus BERAT

The endpoints overlap visibly: some `RINGAN` aerial inundation looks more extensive than `BERAT` river-monitoring frames. Frozen SigLIP2 conditional severity supports this difficulty: scene-aware 5-NN accuracy is 64.89%, and the grouped linear probe is 58.78% on a balanced flood sample.

**BANJIR conclusion — MEASURED + INTERPRETATION.** Flood recognition is clear, but severity is visually inconsistent and often unobservable from a single frame. The most plausible cues are inundated area, water depth relative to people/vehicles/buildings, affected infrastructure, and rescue context; the dataset does not apply them consistently. Flood severity is the principal hard/noisy subproblem.

### GEMPA BUMI

#### RINGAN versus SEDANG

Eighteen of 20 reviewed `RINGAN` images showed intact structures or minor cracks, so `RINGAN` has a recognizable core. Two were major-destruction outliers (`921.jpg`, `1392.jpg`). In contrast, about 18 of 20 `SEDANG` images showed collapse, rubble, or major destruction; examples include `1947`, `1015.PNG`, `683.jpeg`, `2503`, `671.png`, and `2152`.

#### SEDANG versus BERAT

`SEDANG` often looks conventionally severe. Of 19 reviewed `BERAT` images, 12 were severe while seven showed localized/minor damage, intact context, or uncertainty. `BERAT/248`, `1502`, and `667` are weak/ambiguous, and `1529` appears illustration-like. Exact duplicates directly connect `SEDANG` and `BERAT` in 54 groups.

#### RINGAN versus BERAT

The typical endpoints differ—small cracks/intact buildings versus large collapse—but the outliers and exact conflicts violate this distinction. SigLIP2's conditional grouped probe reaches 75.67%, with most remaining confusion between `SEDANG` and `BERAT`.

**GEMPA BUMI conclusion — MEASURED + INTERPRETATION.** Severity appears to represent structural damage extent, collapse, and rubble, but the boundary between `SEDANG` and `BERAT` is not consistently applied. `RINGAN` is usually distinctive; the two higher levels form a broad overlapping destruction regime.

### KEBAKARAN

#### RINGAN versus SEDANG

All 20 reviewed `RINGAN` images contained visible flame, usually a controlled/isolated burner, fireplace, container, campfire, or small fire rather than disaster-scale damage. Seven of 20 also had heavy noise, rotation, or baked transforms. Eighteen of 19 `SEDANG` images showed an active vehicle, vegetation, or structure fire. `SEDANG/1032.jpeg`, a group photo with no fire, is a strong inconsistency.

#### SEDANG versus BERAT

All 19 reviewed `BERAT` images showed extensive active fire/smoke, suppression, or burned aftermath. `SEDANG` covers active but more localized incidents; the boundary remains somewhat subjective when framing hides affected extent.

#### RINGAN versus BERAT

The endpoint semantics are comparatively clear: controlled/small flame versus large destructive incident or extensive aftermath. SigLIP2 conditional severity 5-NN reaches 95.56%, and the grouped probe reaches 92.00%.

**KEBAKARAN conclusion — MEASURED + INTERPRETATION.** Fire has the strongest severity structure. Relevant cues include affected area/object, flame and smoke extent, suppression response, and destruction/aftermath. Source genre remains a shortcut: the `RINGAN` controlled-fire collection is visually and geometrically distinctive.

### Cross-disaster conclusion

Severity is **disaster-specific rather than globally comparable**. “Heavy” flood is about extent/depth and affected context; “heavy” quake is about structural collapse; “heavy” fire is about scale/destruction/aftermath. One global severity head can share statistical strength, but a single global semantic axis is too restrictive without disaster conditioning.

## 13. Severity Ordinality

Conceptual order was encoded as `RINGAN=0`, `SEDANG=1`, `BERAT=2`. Ordinality was assessed using visual review, scene-grouped probe confusions, and cosine distances between severity centroids within each disaster. Centroid distances describe geometry, not direction along a single line.

### SigLIP2 evidence

| Scope | Mean absolute severity-step error | Probe errors | Adjacent among errors | Non-adjacent among errors | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Global | 0.3344 | 756 | 80.56% | 19.44% | Partially Supported |
| BANJIR | 0.4878 | 371 | 81.67% | 18.33% | Unsupported |
| GEMPA BUMI | 0.2978 | 219 | 77.63% | 22.37% | Partially Supported |
| KEBAKARAN | 0.0978 | 72 | 77.78% | 22.22% | Supported, with caveat |

For BANJIR, centroid distances are `RINGAN–SEDANG=0.01581`, `SEDANG–BERAT=0.01286`, and `RINGAN–BERAT=0.03725`: endpoints are farther apart, but all margins are tiny and classification is weak. This geometric hint is outweighed by visual inconsistency and extensive three-way confusion.

For GEMPA BUMI, distances are `RINGAN–SEDANG=0.14445`, `SEDANG–BERAT=0.01419`, and `RINGAN–BERAT=0.09539`. `SEDANG` and `BERAT` are nearly coincident; the arrangement is not a clean linear order, although `RINGAN` is usually distinguishable.

For KEBAKARAN, distances are `RINGAN–SEDANG=0.18914`, `SEDANG–BERAT=0.05918`, and `RINGAN–BERAT=0.16348`. Severity is highly predictable and visual endpoints are meaningful, but the centroid triangle is not one-dimensional—likely because controlled flame, active incident, and aftermath are also categorical source/content regimes.

The DINOv2 auxiliary probe gives the same broad conclusion: global mean step error 0.3222; BANJIR 0.4844, GEMPA 0.3233, and KEBAKARAN 0.0800. Its adjacent-error shares are 78.66%, 81.84%, 79.25%, and 83.87%, respectively.

**Statistical caution.** Adjacent errors are more numerous than endpoint errors even under many non-ordinal confusion patterns because there are two adjacent class pairs and one endpoint pair. The observed ~78–84% adjacent share is supportive but not proof.

**Decision.** Do not impose one global ordinal loss. A fire-specific ordinal auxiliary objective is justified for experimentation; a quake-specific objective may be tested cautiously; a flood ordinal objective is currently unsupported.

## 14. Foundation Representation Analysis

### Protocol and scope

**MEASURED.** Frozen representations were computed on a fixed balanced sample: 300 TRAIN images from each of nine joint classes (2,700 TRAIN total) plus all 450 TEST images, seed 20260901. All embeddings are 768-dimensional and L2-normalized. For 5-NN, every image in the query's strict perceptual component was excluded. Linear probes used five-fold `StratifiedGroupKFold` with the same component groups. This is representation EDA, not final training.

The balanced sample makes cross-class comparisons fair but is not the natural TRAIN distribution. Results are not leaderboard estimates. CPU-only throughput was measured on the available Intel Core Ultra 7 155H; no GPU was available.

### SigLIP / SigLIP2

Checkpoint: frozen `google/siglip2-base-patch16-224`, using its native processor at 224 pixels. Total measured inference time was 725.43 seconds for 3,150 images (4.34 images/s).

| Diagnostic | Scene-excluded 5-NN accuracy | Mean 5-neighbor purity | Grouped linear-probe accuracy | Nearest-centroid accuracy |
| --- | ---: | ---: | ---: | ---: |
| Disaster | 99.19% | 98.90% | 99.15% | 98.00% |
| Global severity | 78.48% | 73.46% | 72.00% | 64.96% |
| Joint 9-class | 78.11% | 73.07% | 75.04% | 69.52% |
| BANJIR severity | 64.89% | 58.96% | 58.78% | 53.44% |
| GEMPA BUMI severity | 75.56% | 68.24% | 75.67% | 70.67% |
| KEBAKARAN severity | 95.56% | 93.76% | 92.00% | 88.11% |

Joint-label cosine silhouette is only 0.0746 despite high disaster accuracy, because within-disaster severity clouds overlap. This is useful evidence against treating a 2-D cluster plot as proof of nine clean classes.

**Interpretation.** SigLIP2 is highly suitable for disaster semantics and already captures useful severity evidence. Its failure pattern aligns with manual semantics rather than looking like a generic weak feature extractor: fire is clean, quake is intermediate, and flood is difficult.

**Geometry caveat.** This checkpoint uses fixed 224 input and therefore cannot test whether preserving native aspect ratio or fine local damage at higher resolution improves severity. The dataset's geometry makes 384 and a SigLIP2 NaFlex/native-aspect option plausible future experiments, not measured winners.

### DINOv3

**MEASURED LIMITATION.** No DINOv3 checkpoint was cached locally. Official DINOv3 checkpoints require authentication, contact-data sharing, and license acceptance, and the available environment had neither pre-authorized credentials nor a GPU. The official [DINOv3 repository](https://github.com/facebookresearch/dinov3) also recommends CUDA for use, while the [Transformers DINOv3 documentation](https://huggingface.co/docs/transformers/en/model_doc/dinov3) confirms the supported architecture family. DINOv3 was therefore **not run**.

The official [DINOv3 ViT-S/16 model page](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) lists a 21.6M-parameter small checkpoint. Official model listings scale through approximately 86M (ViT-B), 300M (ViT-L), 840M (ViT-H+), and 6.716B (ViT-7B). Given the current CPU-only throughput and the fact that a small/base checkpoint can answer the representation question, 7B is not a rational first experiment.

#### Auxiliary DINOv2 baseline—not DINOv3

To avoid leaving the self-supervised representation question completely unmeasured, cached frozen `facebook/dinov2-base` was run on exactly the same paths and folds. It took 694.64 seconds (4.53 images/s). These numbers **must not be described as DINOv3 results**.

| Diagnostic | DINOv2 scene-excluded 5-NN | DINOv2 neighbor purity | DINOv2 grouped probe | DINOv2 centroid accuracy |
| --- | ---: | ---: | ---: | ---: |
| Disaster | 98.48% | 98.07% | 98.70% | 97.89% |
| Global severity | 75.59% | 72.01% | 73.44% | 68.63% |
| Joint 9-class | 75.44% | 71.58% | 74.30% | 69.85% |
| BANJIR severity | 61.78% | 58.13% | 59.00% | 56.89% |
| GEMPA BUMI severity | 73.22% | 66.22% | 73.22% | 68.56% |
| KEBAKARAN severity | 94.11% | 93.09% | 93.11% | 87.44% |

Joint-label cosine silhouette is 0.0490. DINOv2 is slightly faster in this CPU pass (~4.4%), but one run is not a hardware-general throughput benchmark.

### Direct Comparison

Because DINOv3 was unavailable, the table makes the missing comparison explicit. DINOv2 is shown only as an auxiliary directional baseline.

| Criterion | SigLIP2-B/16-224 | DINOv3 | DINOv2-B/14 auxiliary | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Disaster 5-NN | 99.19% | Not evaluated | 98.48% | Both measured reps make disaster nearly trivial; SigLIP2 leads. |
| Global severity 5-NN | 78.48% | Not evaluated | 75.59% | SigLIP2 neighborhoods are purer. |
| Global severity probe | 72.00% | Not evaluated | 73.44% | DINOv2 is slightly more linearly usable globally. |
| BANJIR severity 5-NN / probe | 64.89% / 58.78% | Not evaluated | 61.78% / 59.00% | Both struggle; neither resolves label semantics. |
| GEMPA severity 5-NN / probe | 75.56% / 75.67% | Not evaluated | 73.22% / 73.22% | SigLIP2 leads. |
| KEBAKARAN severity 5-NN / probe | 95.56% / 92.00% | Not evaluated | 94.11% / 93.11% | Both strong; DINOv2 probe slightly leads. |
| Joint 9-class 5-NN / probe | 78.11% / 75.04% | Not evaluated | 75.44% / 74.30% | SigLIP2 leads overall. |
| Joint silhouette | 0.0746 | Not evaluated | 0.0490 | SigLIP2 has stronger but still overlapping structure. |
| TRAIN–TEST grouped domain AUC | 0.865 | Not evaluated | 0.848 | Both expose source-mixture shift. |
| Cluster-distribution JS divergence | 0.247 nats | Not evaluated | 0.216 nats | Moderate shift; clustering depends on representation. |
| CPU throughput | 4.34 img/s | Not evaluated | 4.53 img/s | Similar base-model cost here. |

Cosine distances are representation-specific and should not be compared numerically across columns. Within SigLIP2, TEST→sampled-TRAIN nearest distance has median 0.1090, p90 0.1870, p99 0.2786, maximum 0.3068. Scene-excluded TRAIN leave-one-out distance is lower: median 0.0710, p90 0.1569, p99 0.2382, maximum 0.4383. The nearest support used only 2,700 balanced TRAIN samples, not all 17,482 images.

The measured within/between-class centroid diagnostics add a complementary view. “Within” is mean point-to-own-centroid cosine distance; “between” is mean pairwise class-centroid distance. Their ratio is descriptive rather than a calibrated separability score.

| Task | SigLIP2 within / between / ratio | DINOv2 within / between / ratio |
| --- | ---: | ---: |
| Disaster | 0.1785 / 0.1722 / 0.965 | 0.5243 / 0.7966 / 1.519 |
| Global severity | 0.2177 / 0.0361 / 0.166 | 0.6405 / 0.2917 / 0.455 |
| Joint 9-class | 0.1552 / 0.1834 / 1.182 | 0.4591 / 0.7229 / 1.575 |

The ratios cannot rank the two representations because their embedding geometries differ. Within each representation, however, severity centroids are much closer relative to class spread than disaster/joint centroids, independently confirming that severity—not disaster identity—is the overlapping factor.

Cross-class five-neighbor rates (`1 − purity`) are, for SigLIP2 versus DINOv2: disaster 1.10% versus 1.93%, global severity 26.54% versus 27.99%, joint 26.93% versus 28.42%, BANJIR severity 41.04% versus 41.87%, GEMPA severity 31.76% versus 33.78%, and KEBAKARAN severity 6.24% versus 6.91%.

#### Error complementarity

Same-fold raw linear probes were compared image by image. They are a complementarity diagnostic; the scene-aware scores above remain the trustworthy absolute estimates.

| Task | SigLIP2 only correct | DINOv2 only correct | Both wrong | Oracle “either correct” | Prediction disagreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disaster | 0.89% | 0.30% | 0.52% | 99.48% | 1.22% |
| Global severity | 6.52% | 8.33% | 19.52% | 80.48% | 17.74% |
| Joint | 5.93% | 5.30% | 19.89% | 80.11% | 14.56% |
| BANJIR severity | 8.44% | 7.89% | 33.78% | 66.22% | 24.22% |
| GEMPA severity | 7.11% | 5.67% | 20.00% | 80.00% | 14.22% |
| KEBAKARAN severity | 2.33% | 3.33% | 4.33% | 95.67% | 6.11% |

The oracle is an upper bound that no practical ensemble automatically achieves. Still, the two representations make different severity mistakes, especially for BANJIR, supporting later probability ensembling or feature fusion if independently fine-tuned models preserve this diversity.

#### Answers to the required representation questions

1. **Does SigLIP/SigLIP2 appear suitable?** Yes. Evidence is strong for disaster and moderate-to-strong for joint/severity representation, with known flood limitations.
2. **Does DINOv3 appear better for any part?** Unknown: DINOv3 was not evaluated. DINOv2's slightly better global/fire linear probes make a DINOv3 experiment plausible, not proven superior.
3. **Which better separates disaster type?** Among measured models, SigLIP2. DINOv3 remains unresolved.
4. **Which better separates severity?** Mixed among measured models: SigLIP2 has better 5-NN/global neighborhood structure and quake severity; DINOv2 has a 1.44-point global probe and 1.11-point fire-probe advantage.
5. **Does quality differ by disaster?** Decisively: both are weakest on BANJIR, intermediate on GEMPA BUMI, and strongest on KEBAKARAN.
6. **Is severity approximately ordinal?** Only partially and conditionally; see Section 13.
7. **Are SigLIP and DINOv3 complementary?** Unknown directly. SigLIP2 and DINOv2 are measurably complementary on severity, motivating the DINOv3 test.
8. **Is DINOv3 ViT-7B worth testing later?** Not initially. First establish a gain with ViT-S/B under identical folds; move larger only if scaling is informative and adequate GPU compute becomes available.
9. **Which representation should be prioritized?** SigLIP2 is the measured primary. DINOv3-small/base is the recommended secondary experiment once legitimately accessible.
10. **What remains unresolved until fine-tuning?** Resolution benefit, NaFlex benefit, augmentation response, DINOv3 performance, robustness under full natural class frequencies, ensemble gain, and whether conditional heads outperform a simple two-head model.

## 15. Label Noise, Ambiguity, and Hard Samples

No image was relabeled. The categories below describe evidence strength, not corrected ground truth.

### Likely mislabeled

- `TRAIN/KEBAKARAN/KERUSAKAN SEDANG/1032.jpeg`: reviewed as a group photo with no visible fire.
- The exact cross-disaster pair `BANJIR/RINGAN/220.jpg` = `GEMPA BUMI/SEDANG/1204.jpg`: identical pixels cannot validly express two different disaster targets under an image-only task.
- Exact three-severity quake groups, such as `BERAT/1335.jpg` = `RINGAN/1120.jpg` = `SEDANG/2564.jpg`, prove at least two inconsistent annotations even though the correct one is not determined.
- `GEMPA BUMI/RINGAN/1392.jpg` shows major destruction inconsistent with the dominant `RINGAN` semantics; this is strong visual evidence but weaker than an exact conflict.

### Ambiguous

- Many BANJIR scenes show water or a river without scale, depth, affected structures, or damage extent.
- Most `GEMPA BUMI/SEDANG`↔`BERAT` conflicts: both labels often depict large collapse/rubble.
- Fire images framed tightly around flame can hide affected area, making `SEDANG` versus `BERAT` subjective.

### Outlier

- `GEMPA BUMI/BERAT/1529.jpg` is illustration-like rather than ordinary field photography.
- Collages, infographics, screenshots, satellite-like views, marked webcams, and strongly transformed images are unusual domains but not automatically mislabeled.
- High representation distance or centroid disagreement alone is not label-noise proof. For example, `BANJIR/BERAT/1583.png` and `BANJIR/SEDANG/973.png` are plausible flood webcams with ambiguous severity; `GEMPA BUMI/SEDANG/2604.jpg` is a close-up cracked wall plausibly near the `RINGAN` boundary; `KEBAKARAN/SEDANG/44.png` is rotated/blurred yet visibly a valid fire.

### Hard but valid

- Subtle cracks or local building damage.
- Fire aftermath with no active flame.
- Flood response/cleanup frames with no visible active inundation.
- Wide views where damage is real but occupies few pixels.

**Recommended action.** Maintain an audit flag, not a replacement label. Compare baseline training with (a) all data, (b) exact-conflict groups downweighted or excluded from training only, and (c) a small manually adjudicated high-confidence subset if competition rules permit. Never tune the policy on one fold.

## 16. TRAIN vs TEST Distribution Shift

### Measured low-level shift

- TEST median pixel count is 480,000 versus TRAIN's 360,000, and TEST p95 is 4.56M versus 2.46M.
- Exact-square prevalence falls from 30.69% in TRAIN to 9.56% in TEST; portrait prevalence rises from 8.57% to 19.56%.
- Split × resolution-bucket Cramér's V is 0.348.
- TEST is slightly darker and softer, but contrast, saturation, and entropy medians are close.
- TEST median compressed bpp is 1.981 versus 3.323 in TRAIN, despite its higher median resolution.

### Measured embedding shift

Using all 450 TEST images and the balanced 2,700-image TRAIN embedding sample:

- SigLIP2 grouped TRAIN-vs-TEST probe AUC: 0.8653.
- DINOv2 grouped probe AUC: 0.8479.
- SigLIP2 cluster-distribution Jensen–Shannon divergence: 0.2474 nats.
- DINOv2 cluster JS divergence: 0.2156 nats.
- SigLIP2 TEST nearest-distance p90/p99: 0.1870/0.2786 versus TRAIN scene-excluded leave-one-out 0.1569/0.2382.

The highest-distance TEST examples under one or both embeddings were manually checked. They include flood cleanup with no visible water (`TEST/114.jpg`), rainy field documentation without visible damage (`14.jpg`), a stylized text-overlaid campfire (`398.jpg`), a race-car engine fire (`433.jpg`), low-resolution burned aftermath (`446.jpg`), and winter inundation (`75.jpg`). These are recognizable disaster-adjacent content in atypical genres or contexts, not evidence for fabricated labels.

### Overlap

Five TEST images are exact duplicates of TRAIN images; the strict perceptual graph contains 23 TRAIN–TEST pairs. These are leakage facts, not permission to infer TEST labels or evidence that the remainder of TEST is easy.

**Conclusion.** TRAIN is not perfectly representative of TEST. Evidence strength is **moderate to strong** for a changed source/geometry mixture, but weak for an entirely new semantic domain. Because the nearest-neighbor support calculation used a 2,700-image subset, it may overstate distance relative to the full TRAIN set. Validation should therefore preserve scene independence while checking fold-level source/geometry coverage rather than trying to make folds maximally adversarial.

## 17. Flat vs Hierarchical vs Multi-Task Implications

| Structure | Evidence-based assessment |
| --- | --- |
| Flat 9-class | Useful baseline/auxiliary because joint embeddings are coherent (SigLIP2 75.04% grouped probe), but not directly aligned with the two-row target and cannot give partial credit naturally. |
| Hierarchical hard routing | Disaster is ~99% separable and severity is disaster-specific, so conditioning is attractive. Hard routing, however, propagates the rare disaster error to severity. |
| Shared multi-task two-head | Best first structure: directly matches `jenis` and `kerusakan`, shares data, and optimizes the equally weighted row objective. A single global severity head may underfit disaster-specific semantics. |
| Hybrid auxiliary 9-class | Plausible after the two-head baseline; the joint head can regularize disaster–severity combinations while official heads remain primary. Added complexity needs a measured gain. |
| Disaster-specific severity heads | Strongly motivated follow-up because conditional difficulty is 58.78%, 75.67%, and 92.00% for flood, quake, and fire. Soft routing or selecting by known TRAIN disaster avoids forcing one global boundary. |

**Recommended modeling structure.** Start with a shared SigLIP2 encoder and separate disaster/severity heads. Next compare a hybrid with three disaster-specific severity heads, trained with the true disaster during TRAIN and probability-weighted/soft-routed at inference. Add a low-weight 9-class auxiliary head only if it improves scene-aware row Micro F1. Do not begin with a deep cascade or nine independent specialists.

## 18. Recommended EDA for the Competition Notebook

Only decision-relevant visuals should appear in the future notebook.

| Visualization | What it shows | Why it matters / decision supported |
| --- | --- | --- |
| Disaster, severity, and 9-class count bars | Near-balanced disasters, modest severity skew, localized quake imbalance | Justifies ordinary CE/natural sampling rather than reflex balancing. |
| Resolution/aspect panel by joint class plus TRAIN/TEST | Modes, long tails, square-source concentration, split shift | Justifies aspect-preserving preprocessing and resolution experiments. |
| Evidence-selected 3×3 severity gallery | Medoids/typical, boundary, and outlier images for each disaster | Shows that severity semantics differ by disaster; motivates conditional heads. |
| Exact/strict duplicate component summary | Group counts, size distribution, conflicts, and TRAIN–TEST overlap | Makes random-split leakage concrete; supports grouped CV. |
| Source-bias panel | Selected webcam/news/phone/drone/editor examples plus resolution/device concentrations | Explains shortcut risk and why metadata is diagnostic-only. |
| Scene-aware frozen-representation table and confusion matrices | Disaster versus conditional severity performance | Establishes where the task is hard and why SigLIP2 is primary. |
| TRAIN–TEST geometry and embedding-distance comparison | Shift without assigning TEST labels | Supports robust source coverage and cautious model selection. |
| Ordinal error matrix by disaster | Adjacent versus endpoint mistakes | Shows why global ordinal loss is unjustified and fire-specific testing is reasonable. |

UMAP/PCA may be included only as a labeled supporting view with the quantitative kNN/probe results beside it. Decorative RGB histograms, generic augmentation illustrations, and an unlabeled UMAP are filler rather than evidence.

## 19. Recommended Preprocessing

| Finding | Evidence | Implication | Recommended Action |
| --- | --- | --- | --- |
| Decoding is healthy but filenames are unreliable | 17,932/17,932 decode; 39 materially misleading extension/content pairs | Suffix-specific loaders can fail or choose wrong code paths | Decode by file content with a mature image library; retain a failure log even though the current failure count is zero. |
| Orientation and modes are heterogeneous | EXIF Orientation on 869 files; 825 RGBA plus P, CMYK, and L outliers | Silent orientation/mode differences create inconsistent inputs | Apply EXIF transpose first, then convert to three-channel RGB. Composite alpha onto one documented neutral/background value rather than dropping it accidentally. Transparency prevalence inside the RGBA files was not measured, so compare backgrounds only if artifacts appear. |
| Aspect ratio is broad and label-correlated | Range 0.3005–6.7767; 3,909 TRAIN resolutions; 30.69% TRAIN square versus 9.56% TEST | Square stretching distorts objects; centered crops can remove damage extent; padding style itself can become a shortcut | Preserve aspect ratio using resize+pad or shape buckets. Use a constant/reflect policy consistently and include pad masks only if the backbone supports them. Audit performance by aspect bucket. |
| Fine detail may matter, but source shortcuts are strong | Tiny cracks/rubble are severity cues; fixed 224 SigLIP2 is already strong; TEST has a heavier high-resolution tail | Higher resolution may help quake severity but can also expose watermarks/compression and raise cost | Establish 224, then compare 384 under identical folds. Try 512/native resolution only if 384 improves quake/flood severity enough to justify cost. |
| Backbone preprocessing is model-specific | SigLIP2 diagnostics used its native processor; no evidence supports a universal normalization | Swapping in generic ImageNet normalization can invalidate pretrained feature scaling | Use each checkpoint's official processor/normalization. Keep spatial policy separable so aspect-preserving variants can be tested fairly. |
| Interpolation interacts with small damage | Images span 127×113 to 9232×5588 | Poor downsampling can erase cracks or alias rubble | Use the checkpoint default initially; compare bicubic/Lanczos only within the resolution experiment, not as an uncontrolled difference. |
| Exact and near repetitions overweight collections | 15.82% exact-group involvement; 21.05% strict-near involvement; 105 exact conflict groups | Keeping every copy can overemphasize a webcam/export batch, but deletion may remove TEST-like frequency | Group for validation unconditionally. For training, compare all images against one representative/weight per same-label component; keep conflict groups flagged. Do not delete source files or auto-relabel. |
| No current corruption problem | Zero corrupt/unreadable/zero-byte images | Complicated repair logic has no evidence base | Keep a simple runtime exception/logger; do not build a repair pipeline. |

For fixed-size models, a practical first path is: content-aware decode → EXIF transpose → deterministic RGB/alpha handling → aspect-preserving resize → pad to model input → official normalization. Random spatial transforms belong in augmentation, not the deterministic preprocessing contract.

## 20. Recommended Augmentation

Severity depends on context and affected extent, so augmentation strength should be selected on **severity accuracy and official row Micro F1**, not disaster accuracy alone.

| Augmentation | Assessment | Dataset-specific rationale |
| --- | --- | --- |
| Horizontal flip | **Worth Testing** | Disaster semantics are generally left/right invariant, but text, watermarks, road layouts, and edited sources make it less obviously safe than in natural-object datasets. Test p≈0.5 versus none. |
| Vertical flip | **Potentially Harmful** | Gravity, waterline, flames/smoke, buildings, and horizon orientation are semantic. |
| Rotation | **Worth Testing only mildly** | Small camera tilt is realistic and some fire images are already rotated; large rotations make buildings, water, and smoke physically implausible. Restrict initial tests to about ±5°. |
| RandomResizedCrop | **Potentially Harmful when aggressive; Worth Testing when conservative** | Cropping can remove the very extent/context needed for severity. If used, keep a high retained area (for example 0.80–1.00) and mild ratio range; compare with resize+pad. |
| Brightness adjustment | **Recommended, mild** | Illumination varies and class means differ strongly; modest perturbation can weaken source shortcuts. Strong changes can alter fire visibility and flood cues. |
| Contrast adjustment | **Recommended, mild** | Measured contrast is broad and TEST is slightly different; small changes are plausible. Preserve cracks/rubble and water boundaries. |
| Color jitter | **Worth Testing, weak** | Device/source color varies, but flame color, smoke, water, and burned vegetation carry semantics. Avoid aggressive hue shifts. |
| Blur | **Probably Unnecessary initially** | The dataset already contains soft images and TEST median sharpness is lower; more blur can erase cracks and distant damage. A very low-probability mild blur is a robustness ablation only. |
| Added noise | **Probably Unnecessary / Potentially Harmful** | Fire `RINGAN` already contains heavy noise/transforms; synthetic noise risks reinforcing rather than correcting this source regime and destroys texture. |
| Perspective transform | **Potentially Harmful beyond very mild settings** | Viewpoint diversity already exists; strong warp changes structural geometry used to judge quake damage. |
| Affine transform | **Worth Testing only mildly** | Small translation/scale can reduce framing reliance; shear/large scale behaves like destructive crop/warp. |
| JPEG/compression augmentation | **Worth Testing** | TRAIN/TEST compressed-bpp medians differ and resaved variants are common. Apply mildly and include PNG-origin images without pretending their source format changed causally. |
| RandAugment | **Probably Unnecessary initially** | Its policy can combine harmful rotations, solarization, crop, and color shifts; measured targeted transforms are easier to audit. |
| MixUp | **Potentially Harmful initially** | Blended scenes have no coherent disaster extent or ordinal severity. A low-alpha trial is lower priority only if label smoothing is insufficient. |
| CutMix | **Potentially Harmful** | Pasting a localized flame/collapse into another context creates an invalid severity target and directly corrupts affected-area cues. |
| Random Erasing | **Potentially Harmful** | It can hide the small crack, flame, vehicle, waterline, or collapsed region that determines severity. |

No augmentation earns “safe” status merely because it is common in image classification. Use a small factorial ablation—geometry policy, mild photometric perturbation, compression, and flip—rather than a large automatic policy search at the start.

## 21. Recommended Loss and Sampling Strategy

### Default

Use ordinary cross entropy for the disaster head and severity head. With equal numbers of official `jenis` and `kerusakan` rows, equal head weighting is the natural starting point. If a joint auxiliary head is added, give it a smaller tuned coefficient so it cannot silently replace the official objective.

### Options assessed

| Method | Recommendation | Evidence-based reason |
| --- | --- | --- |
| Standard cross entropy | **Recommended baseline** | Marginal classes are balanced/mildly skewed; it aligns with row accuracy and gives a clean benchmark. |
| Label smoothing | **Worth Testing modestly** | Exact conflicts and visual ambiguity are real. Compare zero against a small value; excessive smoothing can hurt the clean fire/disaster classes. |
| Class weights | **Not recommended by default** | Joint max/min is 1.96 and severity max/min 1.30; weighting optimizes a different tradeoff from Micro-F1/accuracy. |
| Focal loss | **Probably unnecessary** | The hard examples include irreducible label conflict, not merely easy-example domination. Focal loss may emphasize noise. |
| Oversampling / balanced sampler | **Not recommended by default** | It would over-repeat the smallest quake class and potentially amplify duplicated source clusters. |
| Undersampling | **Potentially harmful** | It discards useful diversity for no metric-driven reason. |

Use per-class F1 and balanced accuracy to understand failures, but accept a balancing method only if it improves the official pooled out-of-fold row score consistently. For disaster-specific severity heads, report natural-frequency and per-disaster results; do not equalize disasters implicitly without accounting for the official image frequency.

## 22. Recommended Validation Strategy

### Primary fold construction

1. Build a graph over TRAIN using all SHA-256 equality edges plus strict perceptual edges (`pHash≤1`, `dHash≤2`, `aHash≤4`).
2. Assign every connected component one immutable `scene_group`; each singleton is its own group.
3. Use **five-fold `StratifiedGroupKFold`**, stratifying on the nine joint disaster×severity labels and grouping on `scene_group`.
4. Freeze the primary split with seed **20260901** for all model comparisons. Use seeds **3407** and **9173** only as secondary split-sensitivity audits for finalists, not to shop for a favorable fold.
5. Keep all members of cross-label components together. Record how each such group is allocated; never split it to improve stratification.

Five folds are a compromise: each validation set remains large enough for nine classes and conditional severity diagnostics, while four-fifths training data retains source diversity. Ten folds would multiply expensive fine-tuning without resolving the group ambiguity; a single holdout would have high source-mixture variance.

### Fold acceptance checks

Before training, verify for every fold:

- joint, disaster, and severity counts/proportions;
- component counts and largest component size;
- number of exact/near and conflicting components;
- resolution/orientation/extension distributions;
- known EXIF device/editor concentrations where present;
- brightness, sharpness, and compression summaries;
- no exact or strict-near edge crossing train/validation.

Do not use TEST labels—none are available—or inferred TEST classes to redesign folds. TEST geometry/embedding coverage can be reported as an external stress diagnostic only.

### Sensitivity validation

Run one broader scene/source-aware audit after a baseline is fixed: manually validate a sample of embedding-linked scene clusters or hold out a concentrated source family such as a webcam/device batch. This answers whether the model depends on collection artifacts. It should complement, not automatically replace, the primary split because overly broad clustering can make validation less representative of TEST.

### Metrics

Calculate the official score by concatenating the one disaster and one severity prediction per image and computing pooled Micro F1. Under this single-label schema it equals:

`(number of correct disaster predictions + number of correct severity predictions) / (2 × number of images)`.

Also report:

- disaster accuracy and 3×3 confusion matrix;
- severity accuracy and 3×3 confusion matrix;
- exact joint-label accuracy and 9×9 confusion matrix;
- per-class precision/recall/F1 for both heads and nine joint combinations;
- conditional severity accuracy/confusion for BANJIR, GEMPA BUMI, and KEBAKARAN;
- fold mean, standard deviation, and pooled out-of-fold score.

These diagnostics explain errors; they do not replace Micro F1 for model selection.

## 23. Recommended Modeling Experiments

The list is deliberately short and ranked by information gain, expected value, and cost.

### Priority 1 — SigLIP2 two-head baseline

- **Hypothesis.** A shared SigLIP2 encoder with separate disaster and severity heads will transfer the strong frozen structure and align directly with the official two-row target.
- **Evidence.** Frozen scene-aware disaster probe is 99.15%, severity probe 72.00%, and joint probe 75.04%; no final fine-tuning has yet been performed.
- **Experiment.** First fit lightweight heads on frozen embeddings; then fine-tune `siglip2-base-patch16-224` conservatively using the fixed five folds, ordinary CE, aspect-preserving inputs, and minimal augmentation.
- **Success criterion.** Fine-tuning improves pooled out-of-fold row Micro F1 over frozen heads consistently across folds without sacrificing the already near-ceiling disaster head or increasing source-bucket variance.

### Priority 2 — Input geometry and resolution

- **Hypothesis.** Aspect preservation and 384 resolution improve quake/flood severity by retaining spatial extent and local damage, while naive crops or square stretching hurt it.
- **Evidence.** Aspect range is 0.3005–6.7767, TEST geometry shifts, and damage cues can be small/contextual; the current representation test used fixed 224 only.
- **Experiment.** On the same SigLIP2 training recipe, compare 224 resize+pad, 384 resize+pad, and conservative crop; if available, add a comparable SigLIP2 NaFlex/native-aspect configuration.
- **Success criterion.** A repeatable severity and pooled-row gain—especially BANJIR/GEMPA—large enough to justify measured compute, with no degradation on portrait/extreme-ratio or low-resolution buckets.

### Priority 3 — Duplicate/conflict training policy

- **Hypothesis.** Preventing repeated same-label components from dominating batches and reducing influence of irreconcilable conflict groups will improve scene-generalization, even if random-split scores would fall.
- **Evidence.** 15.82% exact and 21.05% strict-near involvement; 105 exact TRAIN conflict groups.
- **Experiment.** Compare all images, component-balanced weights, and same-label component representatives; separately test flagging/downweighting conflict components. Keep folds identical.
- **Success criterion.** Higher scene-aware pooled row score or lower fold/source variance without losing performance on TEST-like geometry/source buckets. A random-fold-only gain does not count.

### Priority 4 — Disaster-conditioned severity

- **Hypothesis.** Soft-routed disaster-specific severity heads will outperform one global severity boundary because severity semantics and difficulty differ sharply by disaster.
- **Evidence.** SigLIP2 grouped conditional probe is 58.78% BANJIR, 75.67% GEMPA, and 92.00% KEBAKARAN; visual cues are disaster-specific while disaster prediction is ~99%.
- **Experiment.** Compare the two-head baseline against three conditional severity heads with soft routing. Optionally add a low-weight joint 9-class auxiliary head in a separate ablation.
- **Success criterion.** Higher pooled row Micro F1 and severity accuracy across folds, with gains in flood/quake not offset by route-error or fire degradation.

### Priority 5 — DINOv3-small/base representation

- **Hypothesis.** A self-supervised DINOv3 encoder may add structural/texture sensitivity and complementary severity errors.
- **Evidence.** Same-path DINOv2 makes different severity predictions (17.74% global disagreement) and slightly leads SigLIP2 on the global and fire linear probes. DINOv3 itself is unmeasured.
- **Experiment.** After legitimate checkpoint access and suitable GPU availability, evaluate ViT-S/16 or ViT-B/16 frozen and fine-tuned under exactly the SigLIP2 folds, input policy, and metrics.
- **Success criterion.** It beats SigLIP2 on conditional severity or supplies complementary out-of-fold errors that improve a simple calibrated blend. If small/base shows no signal, do not escalate to 7B.

### Priority 6 — Calibrated representation ensemble

- **Hypothesis.** SigLIP2 and a validated DINO-family model can reduce severity error through probability diversity.
- **Evidence.** The frozen SigLIP2/DINOv2 oracle “either correct” is 80.48% for severity versus individual raw probes near 72–74%, although this oracle is unattainable directly.
- **Experiment.** Blend fold-calibrated probabilities from independently trained finalists; compare global weights against disaster-conditional weights. Feature fusion is only warranted if probability blending leaves a repeatable gap.
- **Success criterion.** Consistent pooled out-of-fold improvement on fixed predictions, followed by confirmation on secondary seeds, without using TEST-derived pseudo-labels.

**Deferred.** Fire-specific ordinal auxiliary loss and mild test-time augmentation are lower-cost follow-ups after the primary structure is established. Global ordinal training, retrieval blending, and 7B scaling are not current priorities.

## 24. What We Should NOT Do

- Do not use an ordinary random image split as the primary validation; duplicate and scene leakage is measured and substantial.
- Do not split an exact/strict-near component across folds, including cross-label components.
- Do not infer TEST labels from the five exact or 23 strict perceptual overlaps, and do not present them as generalization evidence.
- Do not exploit filenames, numeric ranges, EXIF devices/editors/timestamps, extensions, file size, or resolution deliberately. Their correlation diagnoses collection bias.
- Do not assume 12 classes; the actual folders contain nine combinations and the output has two targets per image.
- Do not equate official row Micro F1 with 9-class joint accuracy.
- Do not square-stretch all images or use aggressive RandomResizedCrop, CutMix, erasing, rotation, perspective, noise, or blur before proving that severity survives.
- Do not remove every duplicate or automatically choose a label for conflict groups. Frequency can represent real source prevalence, and conflicting pixels do not reveal the correct annotation.
- Do not apply class weights, focal loss, over/undersampling, or a balanced sampler merely because the largest joint class is 1.96× the smallest.
- Do not impose a single global ordinal loss; BANJIR evidence is unsupported and GEMPA's high severities overlap.
- Do not use UMAP alone as proof of separability or domain shift.
- Do not describe DINOv2 measurements as DINOv3, and do not claim a SigLIP2-versus-DINOv3 winner without running DINOv3.
- Do not download/run DINOv3 ViT-7B first. A small/base checkpoint answers the useful question at a fraction of the cost.
- Do not treat frozen probes, kNN, or oracle ensembles as leaderboard-performance estimates.
- Do not build a complex hard-routing hierarchy before the target-aligned two-head baseline.

## 25. Final Decision Table

| Decision Area | Recommended Direction | Evidence Strength |
| --- | --- | --- |
| Primary representation | SigLIP2 base; start 224, compare 384 | Strong |
| Secondary representation | DINOv3 ViT-S/B once accessible; DINOv2 is only supporting directional evidence | Moderate hypothesis / DINOv3 unmeasured |
| Potential complementarity | Probability ensemble after independent grouped-CV fine-tuning | Moderate |
| Input geometry | EXIF-aware, aspect-preserving resize+pad or buckets; no square stretch | Strong |
| Preprocessing | Content-aware decode, deterministic RGB/alpha handling, official backbone normalization | Strong |
| Augmentation | Mild brightness/contrast; test flip and JPEG; conservative spatial transforms | Moderate |
| Class balancing | Natural sampling and standard CE first | Strong |
| Label smoothing | Small controlled ablation | Moderate |
| Duplicate policy | Always group in CV; compare component weighting/dedup only during training | Strong for grouping, Moderate for training policy |
| Validation | Five-fold joint-stratified exact+strict-near `StratifiedGroupKFold`, seed 20260901 | Strong |
| Official metric | Pooled two-row Micro F1 = row accuracy under observed schema | Strong, conditional on ordinary scorer implementation |
| Modeling structure | Shared disaster/severity heads first; soft disaster-specific severity next | Strong / Moderate |
| Ordinal objective | Fire-specific test only; no global ordinal default | Moderate |
| DINOv3 7B experiment | Defer unless small/base establishes scaling value and GPU budget exists | Strong |
| TRAIN–TEST shift response | Robust source coverage and bucket diagnostics; no TEST label inference | Moderate to Strong |

## 26. Limitations and Open Questions

1. **DINOv3 remains unmeasured.** Access was gated and no checkpoint/GPU was available. DINOv2 is not a substitute for the requested family comparison.
2. **Frozen analysis used a balanced 2,700-image TRAIN sample.** All 450 TEST images were embedded, but not all 17,482 TRAIN images. kNN support distances may therefore overstate TEST novelty, and natural-frequency probe performance may differ.
3. **Perceptual duplicate recall is deliberately incomplete.** The strict graph is high precision, manually verified on 48 pairs, but stronger crops, screenshots, multi-view events, or video sequences can be missed.
4. **Source-domain labels do not exist.** Domain conclusions combine metadata, geometry, quality, embedding shift, and visual inspection. They do not provide exhaustive phone/news/drone/event assignments.
5. **Visual inspection is evidence-selected, not exhaustive.** It covered 246 images and high-risk groups. Reviewed proportions are descriptive of the sample, not full-dataset prevalence estimates.
6. **Ground-truth intent is undocumented.** Exact conflicts prove inconsistency but cannot tell which severity definition annotators intended, particularly for flood and quake.
7. **The official scoring code was absent.** The Micro-F1 equivalence follows mathematically from conventional one-label evaluation of the observed 900-row schema; a nonstandard hidden scorer could differ.
8. **No final model was trained.** Frozen kNN/probe rankings do not guarantee fine-tuning rankings, augmentation response, calibration, or leaderboard performance.
9. **NaFlex/native-aspect value is unresolved.** Geometry makes it relevant, but no comparable cached checkpoint was evaluated.
10. **Broader event grouping remains open.** Exact+strict-near components are a defensible minimum. Whether different views of the same event materially inflate validation should be tested with a manually audited sensitivity split.

The evidence is sufficient to begin modeling with a trustworthy baseline: SigLIP2, aspect-preserving inputs, simple two-head CE, and scene-grouped five-fold validation. The next modeling stage should treat severity—especially BANJIR—as the main research question, not spend its first budget making already-easy disaster recognition more elaborate.
