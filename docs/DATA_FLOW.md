# Data Flow and Leakage Policy

## Dataset roles

| Source | Used for | Allowed to affect model artifact? |
|---|---|---:|
| `train/good` memory split | normal representation + memory bank | Yes |
| `train/good` calibration split | review/fail/pixel thresholds | Yes |
| `test/good` | final reporting | No |
| `test/<defect_type>` | final reporting | No |
| `ground_truth/<defect_type>` | localization metrics only | No |

The repository implements one-class anomaly detection. No defect image or defect mask is required to build the memory bank or choose thresholds.

## End-to-end data flow

```text
MVTec AD archive
      │
      ▼
data/raw/<category>
      │
      ├──────────── train/good ─────────────────────────────────────┐
      │                                                            │
      │                       deterministic split                    │
      │                       seed = 42 by default                   │
      │                                                            │
      │               ┌────────────────────┐    ┌──────────────────┐ │
      │               │ memory images      │    │ calibration      │ │
      │               └─────────┬──────────┘    └────────┬─────────┘ │
      │                         │                        │           │
      │                         ▼                        ▼           │
      │                  feature extractor        frozen artifact   │
      │                         │                  scoring loop      │
      │                         ▼                        │           │
      │                  full patch memory              ▼           │
      │                         │                 normal score dist. │
      │                         ▼                        │           │
      │                  coreset selection              ▼           │
      │                         │                   thresholds       │
      │                         └────────────┬───────────┘           │
      │                                      ▼                       │
      │                            models/<category>/                 │
      │                            ├── memory_bank.npy                │
      │                            └── config.json                    │
      │                                      │                       │
      └──────── test + ground_truth ─────────┼─────── report-only ──┘
                                             ▼
                                  reports/<category>/test_metrics.json
```

## Reproducibility contract

Each `config.json` stores:
- category and model version;
- seed;
- backbone and feature layers;
- preprocessing image size / mean / std;
- split sizes and calibration policy;
- all calibrated thresholds;
- coreset size;
- runtime library versions.

This makes an inference result traceable to the artifact that produced it.

## Operational interpretation

The thresholds are policy thresholds calibrated from normal data, not universal guarantees. In a real factory deployment they should be revalidated on site-specific data, camera setup, lighting, acceptable false reject rate, and defect risk tolerance before being used for production decisions.
