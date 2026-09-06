# Architecture

This repository is organized around four explicit boundaries. Keeping these boundaries separate is the main defense against data leakage and serving the wrong model artifact.

## 1. Data ingestion

Input follows the native MVTec AD category structure under `data/raw`:

```text
data/raw/
└── <category>/
    ├── train/good/*.png
    ├── test/good/*.png
    ├── test/<defect_type>/*.png
    └── ground_truth/<defect_type>/*_mask.png
```

`src/data/` owns path discovery and preprocessing. Image size and ImageNet normalization are serialized into each model artifact so inference uses exactly the same transform.

## 2. Offline model building

`src/training/trainer.py` is the only model-building workflow.

```text
train/good
   │
   ├── memory split ──> frozen ResNet18 layer2+layer3 ──> patch embeddings
   │                                                   │
   │                                                   └─> greedy coreset
   │                                                        │
   │                                                        └─> memory_bank.npy
   │
   └── held-out normal calibration ──> NN distances ──> smoothed anomaly maps
                                                       │
                                                       ├─> review threshold
                                                       ├─> fail/image threshold
                                                       └─> pixel threshold
```

Defect test images are not used here.

## 3. Report-only evaluation

`src/evaluation/evaluator.py` loads a frozen artifact and scores the MVTec `test/` split. It never modifies or recomputes thresholds.

Outputs include:
- image AUROC / average precision;
- pixel AUROC / average precision / AUPRO;
- thresholded operational QC metrics such as recall, specificity, false reject rate and false accept rate.

Ground-truth masks are required for every defect test image. Missing masks fail fast instead of silently becoming empty masks.

## 4. Online inspection

`src/inference/` loads one category artifact, applies its serialized preprocessing, performs 1-NN search against the memory bank, creates an anomaly map, and emits a PASS / REVIEW / FAIL decision.

`src/api/` is only a serving layer. It does not own model-building logic.

## Artifact contract

Canonical layout:

```text
models/
├── bottle/
│   ├── config.json
│   └── memory_bank.npy
└── cable/
    ├── config.json
    └── memory_bank.npy
```

The resolver never silently falls back from one requested category to another. Legacy root-level artifacts remain readable only when their declared category matches the request.

## Dependency direction

```text
src/data
   ↓
src/model ← src/training
   ↓          ↓
src/inference
   ↓
src/evaluation     src/api
```

Training produces artifacts. Inference consumes artifacts. Evaluation and API both consume inference; they do not change the artifact.
