# 🏆 BDC Statistics Explore 2026: Disaster Image Classification

**1st place** solution for the image track of the [Big Data Competition Statistics Explore 2026](https://www.kaggle.com/competitions/big-data-competition-statistics-explore-2026/leaderboard).

| Leaderboard | Micro F1  | Rank   |
| ----------- | --------- | ------ |
| Public      | 0.933     | #2     |
| **Private** | **0.904** | **#1** |

**Team (Institut Teknologi Bandung):** Riantama Putra, Farrel Athalla Putra, Ayu Nagita Cahyaning Wulan

## Task

Each disaster photo receives two labels:

- **`jenis`** (disaster type): Flood, Earthquake, Fire
- **`kerusakan`** (damage severity): Light, Moderate, Heavy

The training set has 17,482 images across 9 type × severity classes, and the test set has 450 images. The metric is micro F1 over 900 rows, which is equal to the mean of type accuracy and severity accuracy. Type accuracy is already around 99.6%, so nearly all remaining gains come from severity.

## Approach

**Backbone:** DINOv3 ViT-7B/16 (`facebook/dinov3-vit7b16-pretrain-lvd1689m`) at 384px in bf16. Only the last 6 of 40 transformer blocks are fine-tuned, which is 1.0B of 6.7B parameters.

**1. Leak-free validation.** About 38% of images have a near-duplicate, mostly frames from fixed river cameras. Scenes are grouped by difference hash, then extended with rotation- and mirror-invariant matching (polar-Fourier descriptors verified by NCC). Folds are built with `StratifiedGroupKFold`. A naive random split would leak **45%** of the validation rows.

**2. Severity-specific head.** Damage evidence is local, ordinal, and means something different for each disaster type.

| Problem | Solution |
| ------- | -------- |
| Mean pooling dilutes local damage | Four feature streams: CLS, mean, gated attention pool, top-10% patches |
| "Heavy" means different things per disaster type | **FiLM** conditioning of severity features on predicted type probabilities |
| Light < Moderate < Heavy is ordered | **CORN** ordinal head with **SORD** soft targets |
| Type × severity interaction | Joint 9-class head |

**3. Label noise.** About 19% of images belong to scene groups with conflicting severity labels, and 91% of those conflicts are only one rank apart. Training uses soft targets and label smoothing instead of one-hot labels.

**4. Evidence-preserving augmentation.** MixUp is used without CutMix, since CutMix can cut out the damaged region. Hue jitter is capped at ±0.02 because color is the strongest cue for disaster type. Gaussian noise is added to every class, because one class carries 10× more sensor noise than the rest and would otherwise become a shortcut.

**5. Decoding.** Head blend weights and per-class log-probability offsets are tuned on validation to reduce bias toward the Moderate class.

## Validation Results (fold 0, grouped)

| Type Accuracy | Severity Accuracy | Micro F1 |
| ------------- | ----------------- | -------- |
| 0.9963        | 0.7812            | 0.8888   |

The theoretical ceiling for severity accuracy (majority label per scene group) is 0.927.

## Repository Structure

```
notebooks/04_dinov3_severity_enhanced.ipynb     # full pipeline: EDA, validation, training, submission
results/submission_dinov3_vit7b16_severity.csv  # final submission
docs/presentation.pdf                           # competition presentation slides
```

## Usage

- **Requirements:** an 80 GB GPU (A100/H100), a Hugging Face token with access to DINOv3, and Kaggle API credentials.
- **Training:** about 2.8 hours for 8 epochs on an A100 80GB. Set `HF_TOKEN` and `KAGGLE_API_TOKEN` in the *Kredensial* cell or as environment variables, then run all cells.
- **Inference only:** set `RUN_MODE = 'INFERENCE'` and point `MODEL_PATH` to `dinov3_vit7b16_severity.pt`. The weights are 2 GB and are not stored in this repository.

Submission label codes: `jenis` 1/2/3 = Flood/Earthquake/Fire, `kerusakan` 1/2/3 = Heavy/Light/Moderate.
