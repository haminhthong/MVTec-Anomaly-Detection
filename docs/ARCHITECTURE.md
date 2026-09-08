# Kiến trúc hệ thống phát hiện bất thường ảnh công nghiệp

## 1. Overview & Architectural Philosophy

Repository triển khai hệ thống one-class PatchCore-style lấy cảm hứng từ PatchCore (Roth et al., CVPR 2022). Hệ thống kiểm tra ảnh linh kiện công nghiệp bằng normal reference, không yêu cầu defect labels ở bước build model.

### Core Architectural Principles

1. **Explicit Pipeline Boundaries**: The codebase is strictly organized into 4 decoupled canonical pipelines: Data, Model Building, Evaluation, and Serving.
2. **Artifact-Centric Lifecycle**:
   - Training produces a self-contained, immutable artifact bundle under `models/releases/<category>-v<version>/` and updates `models/production.json`.
   - Inference and Evaluation strictly consume frozen artifacts.
   - The REST API is a lightweight HTTP transport layer with zero embedded ML math.
3. **Strict Category Isolation**: Models are strictly partitioned by category. The `ModelRegistry` rejects cross-category fallback (e.g., requesting `cable` will never fall back to `bottle`).
4. **Anti-Leakage & Operational Calibration**: `NormalReferenceManifest` chỉ chứa train/good; `LockedEvaluationManifest` chỉ được đọc ở final evaluation. Policy V1 chỉ có normal-only AUTO_PASS và HUMAN_REVIEW.

---

## 2. The 4 Canonical Pipelines

```mermaid
flowchart TD
    subgraph DataPipeline["1. DATA PIPELINE"]
        R1["Raw Category Images (data/raw/<category>)"] --> V1["validate_reference_category()"]
        V1 --> DM["NormalReferenceManifest (train/good)"]
        R1 --> V2["validate_locked_evaluation()"]
        V2 --> LOCK["LockedEvaluationManifest (test + masks)"]
    end

    subgraph ModelBuilding["2. MODEL BUILDING PIPELINE"]
        DM --> S1["split Reference / Dev / Calibration"]
        S1 --> SM["split_manifest.json (Reproducibility)"]
        S1 --> FE["FeatureExtractor (ResNet18 layer2 + layer3)"]
        FE --> CS["Greedy K-Center Coreset Selection (64D Projection -> 384D Memory)"]
        CS --> MB["MemoryBank (1-NN Euclidean Search Index)"]
        S1 --> CAL["Held-out Normal Calibration"]
        MB & CAL --> TP["ThresholdPolicy (P99 heuristic AUTO_PASS, P99 pixel)"]
        TP & MB & SM --> ART["ModelArtifact (immutable release)"]
    end

    subgraph EvaluationPipeline["3. EVALUATION PIPELINE (REPORT-ONLY)"]
        ART -.-> EV["evaluate_category()"]
        LOCK --> EV
        EV --> T1["Tier 1: Detection (Image AUROC, Image AP)"]
        EV --> T2["Tier 2: Localization (Pixel AUROC, Pixel AP, AUPRO@0.3)"]
        EV --> T3["Operational QC: normal auto-pass/review và defect escape"]
        T1 & T2 & T3 --> REP["reports/<category>/test_metrics.json & reports/benchmark.csv"]
    end

    subgraph ServingPipeline["4. SERVING PIPELINE"]
        HTTP["HTTP REST Request (POST /inspect or /inspect/batch)"] --> APP["FastAPI App (No ML Logic)"]
        APP --> REG["ModelRegistry (Strict Category Resolution)"]
        REG --> DET["AnomalyDetector (inspect / inspect_batch)"]
        DET --> RES["Standardized Inspection JSON Response"]
    end
```

---

## 3. Component Breakdown

### 3.1 Data Pipeline (`src/data/`)
- `validation.py`: Tách validator reference-only khỏi validator locked evaluation. Reference chỉ kiểm tra `train/good`; evaluation mới kiểm tra test và ground-truth mask.
   - `dataset.py`: `ImageFolderDataset` đọc path trực tiếp từ manifest.
   - `transforms.py`: `PreprocessingConfig` định nghĩa kích thước ảnh, vector normalize và torchvision transform.

### 3.2 Model Building Pipeline (`src/training/` & `src/model/`)
- `trainer.py`: Điều phối build model offline; không backpropagation, chỉ forward qua backbone frozen, tạo memory bank bằng coreset và calibration threshold.
- `calibration.py`: Tách Reference / Dev / Calibration bằng seed cố định và tính P99 heuristic normal-only cho AUTO_PASS cùng pixel threshold.
- `artifacts.py`: Định nghĩa `ModelArtifact`, `ThresholdPolicy`, `SplitManifest` và `ModelMetadata`.

### 3.3 Evaluation Pipeline (`src/evaluation/`)
- `evaluator.py`: Engine đánh giá **report-only**, không sửa hoặc tune threshold theo test.
- `metrics.py`: Tính ba nhóm Detection, Localization và Operational QC.
- `aupro.py`: Tính AUPRO tới `max_fpr=0.3` cho localization.

### 3.4 Serving Pipeline (`src/inference/` & `src/api/`)
- `detector.py`: Runtime inspection engine. Supports both single-image inspection and high-throughput batched inference (`inspect_batch`).
- `registry.py`: Resolve category hoặc line_id qua production pointer, không fallback sang category khác; thiếu mapping thì báo lỗi.
- `app.py`: FastAPI server với `/health`, `/health/live`, `/health/ready`, `/models`, `/inspect` và `/inspect/batch`; không chứa ML math.
- `schemas.py`: Pydantic contract cho response API.
