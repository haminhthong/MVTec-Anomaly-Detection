# Data Flow Specification

This document details the transformation of data tensors, representations, and contracts across each stage of the system.

---

## 1. Offline Model Building Data Flow

```
Input Normal Training Images (N images, e.g., 209 images for bottle)
       ↓  (split_normal_paths, 80% / 20%, seed=42)
Memory Set: 167 images                     Calibration Set: 42 images
       ↓                                                 ↓
Resize to config.preprocessing.image_size (224x224)       ↓ (same preprocessing)
ToTensor + ImageNet Normalize                            ↓
Input Tensor: [Batch, 3, 224, 224]                       ↓
       ↓                                                 ↓
Forward pass through frozen FeatureExtractor             ↓
- Layer 2 feature map: [Batch, 128, 28, 28]              ↓
- Layer 3 feature map: [Batch, 256, 14, 14]              ↓
- Bilinear upsampling of Layer 3 to [Batch, 256, 28, 28] ↓
- Channel concatenation: [Batch, 384, 28, 28]            ↓
- Reshape to patch embeddings: [Batch * 784, 384]        ↓
       ↓                                                 ↓
Accumulated Full Memory: [130,928 patches, 384D]         ↓
       ↓                                                 ↓
Random Projection (Johnson-Lindenstrauss)               ↓
Projected features: [130,928, 64D]                       ↓
       ↓                                                 ↓
Greedy K-Center Selection (coreset_fraction=0.05, K=1,000)
       ↓                                                 ↓
Selected Indices: [1,000 integers]                       ↓
       ↓                                                 ↓
Slice original features at selected indices              ↓
Compact Memory Bank: [1,000, 384D]                       ↓
       ↓                                                 ↓
Fitted NearestNeighbors (1-NN, metric='euclidean') ←──────┘
       ↓
For each calibration image:
  - Extract patches [784, 384D]
  - Query 1-NN distances -> reshape to [28, 28]
  - Gaussian Smoothing (sigma=1.0) -> Smoothed Heatmap [28, 28]
  - Image score: 99th percentile of smoothed heatmap
       ↓
Compute ThresholdPolicy:
  - review_threshold = Quantile(normal_scores, 0.95)  -> ~2.6130
  - fail_threshold   = Quantile(normal_scores, 0.99)  -> ~2.8442
  - pixel_threshold  = Quantile(all_normal_pixels, 0.99) -> ~2.5079
       ↓
Serialize to models/<category>/:
  - config.json (ModelArtifact metadata, ThresholdPolicy, PreprocessingConfig)
  - memory_bank.npy ([1000, 384] float32 array)
  - split_manifest.json (reproducible file lists and hashes)
```

---

## 2. Serving Pipeline Data Flow (Inference)

```
Incoming Image (PNG / JPEG / WebP / TIFF)
       ↓
Validation & Decompression Bomb Protection (PIL Image.open)
       ↓
Preprocessing Transform:
- Resize to target (e.g., 224x224)
- ToTensor & Normalize: [1, 3, 224, 224]
       ↓
Backbone Forward Pass (FeatureExtractor):
- Spatial feature maps aligned and concatenated -> [784, 384D]
       ↓
Nearest-Neighbor Query against MemoryBank:
- Distances to nearest coreset patches: [784]
- Reshape to spatial grid: [28, 28]
       ↓
Heatmap Smoothing:
- Gaussian Filter (sigma=1.0) -> Smoothed Anomaly Heatmap [28, 28]
       ↓
Image Scoring & Defect Localization:
- anomaly_score = Percentile(smoothed_heatmap, 99.0)
- peak_score = Max(smoothed_heatmap)
- anomalous_area_ratio = Count(smoothed_heatmap >= pixel_threshold) / Total_Pixels
       ↓
Operational Decision Engine (ThresholdPolicy):
- If anomaly_score < review_threshold:  decision = "PASS", severity = "PASS"
- Else if anomaly_score < fail_threshold: decision = "REVIEW", severity = "REVIEW"
- Else: decision = "FAIL"
    - If area_ratio >= 0.05 or peak_score >= 1.5 * fail_threshold: severity = "FAIL_MAJOR"
    - Else: severity = "FAIL_MINOR"
       ↓
Visualization (Optional):
- Jet-like colormap generation & alpha blending on resized input -> Base64 PNG data URI
       ↓
Output JSON Response (InspectionResponse)
```

---

## 3. High-Throughput Batch Inspection Flow

```
N Images uploaded to POST /inspect/batch
       ↓
Batch Preprocessing -> Stacked Tensor: [N, 3, 224, 224]
       ↓
Single Forward Pass: FeatureExtractor(batch_tensor) -> [N * 784, 384D]
       ↓
Single 1-NN Query against MemoryBank: [N * 784] distances
       ↓
Reshape to [N, 28, 28]
       ↓
Vectorized Heatmap Smoothing & Percentile Computation for each slice
       ↓
Assembled BatchInspectionResponse with N structured results (Throughput: ~10 FPS on CPU, ~80+ FPS on GPU)
```
