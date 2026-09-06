# MVTec Industrial Anomaly Detection

A production-oriented **one-class visual anomaly detection** project built around a PatchCore-style pipeline. The system learns only from normal product images, builds a compact patch memory bank, calibrates operating thresholds on held-out normal data, evaluates on the untouched MVTec AD test split, and serves category-specific inspection through FastAPI.

> This is a PatchCore-style engineering implementation, not a claim of exact reproduction of the original PatchCore paper.

![Defect inspection example](reports/sample_outputs/inspection_defect_sample.png)

## What this project solves

Industrial defect datasets are usually imbalanced and new defect patterns can appear after deployment. Instead of training a supervised classifier on known defect labels, this project learns the visual distribution of **normal** products and flags patches that are far from the normal memory bank.

The repository is structured as a real ML system rather than a single notebook:

```text
raw data
   ↓
normal-only model building
   ↓
category-scoped immutable artifact
   ↓
report-only test evaluation
   ↓
online inspection API
```

## System boundaries

```text
                         OFFLINE MODEL BUILDING
┌───────────────────────────────────────────────────────────────────┐
│ MVTec train/good                                                  │
│       │                                                           │
│       ├─ memory split ─> frozen ResNet18 ─> patch embeddings      │
│       │                                      │                    │
│       │                                      └─> greedy coreset   │
│       │                                             │             │
│       │                                             ▼             │
│       │                                      memory_bank.npy      │
│       │                                                           │
│       └─ calibration split ─> NN anomaly maps ─> P95/P99 policy   │
│                                                     │             │
│                                                     ▼             │
│                                      models/<category>/config.json│
└───────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                       FROZEN MODEL ARTIFACT
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
     REPORT-ONLY EVALUATION                ONLINE INSPECTION
 MVTec test + ground truth             image/API request
               │                                 │
               ▼                                 ▼
 reports/<category>/                 anomaly score + heatmap
 test_metrics.json                   PASS / REVIEW / FAIL
```

### Leakage policy

| Data | Purpose | Used to choose model/threshold? |
|---|---|---:|
| `train/good` memory split | memory bank | Yes |
| `train/good` calibration split | thresholds | Yes |
| `test/good` | final metrics | No |
| `test/<defect_type>` | final metrics | No |
| `ground_truth/<defect_type>` | localization metrics | No |

The test set is **report-only**. Evaluation loads an already calibrated artifact and never retunes thresholds.

## Pipeline

### 1. Data preparation

Expected MVTec layout:

```text
data/raw/
└── bottle/
    ├── train/good/
    ├── test/good/
    ├── test/broken_large/
    ├── test/broken_small/
    └── ground_truth/
```

`src/data/` owns path discovery and preprocessing. The preprocessing configuration is serialized into the artifact and reused during inference and evaluation.

### 2. Normal representation

Images are resized to 224×224 by default and normalized with ImageNet statistics. A frozen ImageNet ResNet18 extracts `layer2` and `layer3` feature maps. The deeper map is resized to the shallower spatial resolution and concatenated to form patch embeddings.

### 3. Memory bank

All normal patch embeddings can be large, so a greedy k-center coreset is selected. The compact original-space embeddings become the nearest-neighbor memory bank.

### 4. Normal-only calibration

A deterministic held-out subset of `train/good` is scored against the memory bank:

- P95 normal image score → `review_threshold`;
- P99 normal image score → `fail_threshold`;
- P99 normal heatmap values → `pixel_threshold`.

These thresholds are saved with the model artifact.

### 5. Inference

A new image follows the exact artifact preprocessing, is embedded by the same frozen backbone, scored by nearest-neighbor distance, smoothed into an anomaly map and converted into an operational decision:

```text
score < review_threshold      -> PASS
review <= score < fail       -> REVIEW
score >= fail_threshold      -> FAIL
```

Localization also reports anomalous area ratio, peak score and a heatmap overlay.

## Repository structure

```text
.
├── src/
│   ├── data/             # dataset contract + preprocessing
│   ├── model/            # feature extractor, coreset, memory bank, artifacts
│   ├── training/         # normal-only build + calibration
│   ├── inference/        # scoring, localization, decisions
│   ├── evaluation/       # report-only metrics
│   ├── api/              # FastAPI serving layer
│   ├── train.py          # training CLI
│   ├── evaluate.py       # evaluation CLI
│   └── pipeline.py       # explicit train/evaluate/all orchestrator
├── models/<category>/    # generated model artifacts
├── reports/<category>/   # generated evaluation reports
├── docs/
│   ├── ARCHITECTURE.md
│   └── DATA_FLOW.md
├── tests/
├── scripts/
├── Dockerfile
└── Makefile
```

## Quick start

```bash
git clone https://github.com/haminhthong/Mvtec-Anomaly-Detection.git
cd Mvtec-Anomaly-Detection
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

Build and evaluate one category:

```bash
python -m src.pipeline --stage all --category bottle
```

Or keep the stages explicit:

```bash
python -m src.pipeline --stage train --category bottle
python -m src.pipeline --stage evaluate --category bottle
```

Makefile equivalents:

```bash
make train CATEGORY=bottle
make evaluate CATEGORY=bottle
make pipeline CATEGORY=bottle
```

### Custom paths

```bash
python -m src.pipeline \
  --stage all \
  --category bottle \
  --data-root data/raw \
  --model-root models \
  --report-root reports
```

## Artifact contract

New runs write category-scoped artifacts only:

```text
models/
└── bottle/
    ├── config.json
    └── memory_bank.npy
```

`config.json` stores the category, version, preprocessing, split policy, thresholds, coreset metadata and runtime versions. The resolver refuses to silently load another category when the requested artifact is missing.

Legacy root-level artifacts are still readable when their declared category matches, but category directories take priority.

## Evaluation

The current committed `bottle` report records:

| Metric | Value |
|---|---:|
| Image AUROC | 1.0000 |
| Image Average Precision | 1.0000 |
| Pixel AUROC | 0.9818 |
| Pixel Average Precision | 0.7157 |
| AUPRO @ 0.3 | 0.9410 |
| Defect Recall at calibrated threshold | 1.0000 |
| False Reject Rate | 0.0000 |
| False Accept Rate | 0.0000 |

These values describe the committed bottle artifact/test report; they are not guaranteed for other categories or real production cameras. Re-run evaluation after building a new artifact.

## API serving

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The serving layer uses `ModelRegistry` to resolve category-scoped artifacts and lazy-load detectors. Model building is deliberately outside the API process.

## Engineering choices

- **One-class learning:** defect labels are not required for training.
- **Frozen backbone:** training is feature extraction + memory construction, not gradient fine-tuning.
- **Category isolation:** `models/<category>/` prevents accidental cross-category artifact overwrite.
- **Frozen test policy:** no threshold search on MVTec test data.
- **Fail-fast evaluation:** missing defect masks are treated as data errors, not empty masks.
- **Config-driven resize:** pixel metrics follow the artifact image size instead of hard-coded 224×224.
- **Reproducibility:** seed, preprocessing, calibration and runtime metadata are stored with the artifact.

## Tests

```bash
pytest -q
```

Tests cover pipeline components and artifact-resolution safety. CI is configured under `.github/workflows/ci.yml`.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — component responsibilities and artifact lifecycle.
- [`docs/DATA_FLOW.md`](docs/DATA_FLOW.md) — exact dataset roles, leakage policy and end-to-end data flow.
- [`RESEARCH_REPORT.md`](RESEARCH_REPORT.md) — research-oriented notes and background.

## Production limitations

MVTec AD is a benchmark, not a substitute for factory validation. Before production use, revalidate on the target camera, lens, lighting, line speed, product variation and business cost of false accepts/rejects. The normal-only P95/P99 thresholds are an explicit operating policy and should be recalibrated with representative site data.

## License

Source code is released under the repository MIT License. MVTec AD has its own dataset license and usage terms; review them separately before commercial use.
