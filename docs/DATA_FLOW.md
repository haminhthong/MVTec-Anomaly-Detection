# Luồng dữ liệu và tensor

Tài liệu mô tả transformation của tensor, manifest và contract qua từng pipeline.

---

## 1. Offline Model Building Data Flow

```
Input Normal Training Images (N ảnh từ train/good)
       ↓  (Reference / Dev / Calibration split, seed=42)
Reference Set                         Dev Set + Calibration Set
       ↓                                                 ↓
Resize theo config.preprocessing.image_size (mặc định 224x224) ↓ (cùng preprocessing)
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
Accumulated Full Memory: [N_reference * H * W patches, D] ↓
       ↓                                                 ↓
Random Projection (Johnson-Lindenstrauss)               ↓
Projected features: [N_reference * H * W, 64D]            ↓
       ↓                                                 ↓
Greedy K-Center Selection (coreset_size; mặc định K=1,000)
       ↓                                                 ↓
Selected Indices: [K integers]                            ↓
       ↓                                                 ↓
Slice original features at selected indices              ↓
Compact Memory Bank: [K, D]                               ↓
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
  - auto_pass_threshold = Quantile(normal_scores, 0.99)  (heuristic normal-only)
  - pixel_threshold     = Quantile(all_normal_pixels, 0.99)
       ↓
Serialize to models/releases/<category>-v<version>/:
  - config.json (ModelArtifact metadata, ThresholdPolicy, PreprocessingConfig)
   - memory_bank.npy ([K, D] float32 array)
  - split_manifest.json (reproducible file lists and hashes)
```

---

## 2. Serving Pipeline Data Flow (Inference)

```
Incoming Image (PNG / JPEG / WebP / TIFF; format thực tế do PIL hỗ trợ)
       ↓
Validation & Decompression Bomb Protection (PIL Image.open)
       ↓
Preprocessing Transform:
- Resize theo preprocessing config (mặc định 224x224)
- ToTensor & Normalize: [1, 3, H, W]
       ↓
Backbone Forward Pass (FeatureExtractor):
- Spatial feature maps aligned và nối lại -> [H_patch * W_patch, D]
       ↓
Nearest-Neighbor Query against MemoryBank:
- Distances tới coreset patch gần nhất: [H_patch * W_patch]
- Reshape về spatial grid: [H_patch, W_patch]
       ↓
Heatmap Smoothing:
- Gaussian Filter theo artifact (mặc định sigma=1.0)
       ↓
Image Scoring & Defect Localization:
- anomaly_score = Percentile(smoothed_heatmap, 99.0)
- peak_score = Max(smoothed_heatmap)
- anomalous_area_ratio = Count(smoothed_heatmap >= pixel_threshold) / Total_Pixels
       ↓
Capture Quality Gate:
- Invalid resolution / blur / exposure / ROI -> decision = "RECAPTURE_REQUIRED"
       ↓
Operational Decision Engine (ThresholdPolicy):
- If anomaly_score < auto_pass_threshold: decision = "AUTO_PASS"
- Else: decision = "HUMAN_REVIEW"
- area_ratio và peak_score chỉ là evidence, không phải severity major/minor.
       ↓
Visualization (Optional):
- Jet-like colormap generation & alpha blending on resized input -> Base64 PNG data URI
       ↓
Output JSON Response (InspectionResponse)
```

---

## 3. High-Throughput Batch Inspection Flow

```
N ảnh upload tới POST /inspect/batch
       ↓
Quality gate từng ảnh
       ↓
Chỉ ảnh hợp lệ được preprocess -> Stacked Tensor: [N_valid, 3, H, W]
       ↓
Single Forward Pass: FeatureExtractor(batch_tensor) -> [N_valid * H_patch * W_patch, D]
       ↓
Single 1-NN Query against MemoryBank: [N_valid * H_patch * W_patch] distances
       ↓
Reshape từng ảnh về [H_patch, W_patch]
       ↓
Vectorized Heatmap Smoothing & Percentile Computation for each slice
       ↓
Ghép BatchInspectionResponse giữ đúng thứ tự input; ảnh không đạt quality trả RECAPTURE_REQUIRED.
Throughput phải đo bằng scripts/benchmark_inference.py theo hardware cụ thể.
```
